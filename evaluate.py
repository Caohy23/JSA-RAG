# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lib2to3.pgen2.tokenize import generate_tokens
import os
import time
from collections import defaultdict
import logging
import random
import numpy as np
import torch
import torch.cuda
import torch.distributed as dist
import torch.nn.functional as F
from src import dist_utils, slurm, util
from src.index_io import load_or_initialize_index, save_embeddings_and_index
from src.model_io import create_checkpoint_directories, load_or_initialize_atlas_model
from src.options import get_options
from src.tasks import get_task
from rag import encode_passages, _to_cuda
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"



def recall(passages, ground_truths):
    total = len(ground_truths)
    passages = [p.lower() for p in passages]
    ground_truths = [g.lower() for g in ground_truths]
    hits = 0
    for g in ground_truths:
        hit_flag = False
        for p in passages:
            if g in p:
                hit_flag = True
                break
        hits += float(hit_flag)
    return hits/total

def _get_eval_data_iterator(opt, data_path, task):
    data_iterator = task.data_iterator(data_path, opt.global_rank, opt.world_size, opt=opt, is_eval=True)
    data_iterator = filter(None, map(task.process, data_iterator))
    data_iterator = list(task.batch_iterator(data_iterator, opt.eval_batch_size))

    if dist.is_initialized():
        len_data = torch.tensor(len(data_iterator), device=torch.device("cuda"))
        dist.all_reduce(len_data, torch.distributed.ReduceOp.MAX)
        dist.barrier()
        if len(data_iterator) < len_data.item():
            data_iterator.extend([{} for _ in range(len_data.item() - len(data_iterator))])

    return data_iterator


@torch.no_grad()
def run_retrieval_only(model, index, opt, data_path, step=None):
    model.eval()
    metrics = defaultdict(lambda: [])
    dataset_wpred = []
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    generator_tokenizer = unwrapped_model.generator_tokenizer

    task = get_task(opt, generator_tokenizer)
    data_iterator = _get_eval_data_iterator(opt, data_path, task)

    for i, batch in enumerate(data_iterator):
        query = batch.get("query", [""])
        answers = batch.get("target", [""])
        batch_metadata = batch.get("metadata")
        query_enc = model.retriever_tokenize(query)
        retrieved_passages, _, _, _ = unwrapped_model.retrieve(
            index,
            opt.n_context,
            query,
            query_enc["input_ids"].cuda(),
            query_enc["attention_mask"].cuda(),
            batch_metadata=batch_metadata,
            filtering_fun=task.filter,
        )
        # If example is a padding example then skip step
        if (len(query) == 0) or (len(query[0]) == 0):
            continue
        for k in range(len(retrieved_passages)):
            if opt.write_results:
                gold = [answers[k]] if not "answers" in batch else batch["answers"][k]
                ex = {"query": query[k], "answers": gold, "passages": retrieved_passages[k]}
                if batch_metadata is not None:
                    ex["metadata"] = batch_metadata[k]
                if "id" in batch:
                    ex["id"] = batch["id"][k]
                dataset_wpred.append(ex)

    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = f"{dataset_name}-step-{step}"
        util.save_distributed_dataset(dataset_wpred, dataset_name, opt)

    return metrics


@torch.no_grad()
def evaluate(model, index, opt, data_path,logger,step=None):
    logger = logging.getLogger(__name__)
    model.eval()
    # print('******************************************')
    # print('device', next(model.parameters()).device)
    metrics = defaultdict(lambda: [])
    ret_cover = []
    dataset_wpred = []
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    generator_tokenizer = unwrapped_model.generator_tokenizer
    pad_token_id = unwrapped_model.generator_tokenizer.pad_token_id
    eos_token_id = unwrapped_model.generator_tokenizer.eos_token_id
    bos_token_id = unwrapped_model.generator_tokenizer.bos_token_id
    #test
    # test_input = generator_tokenizer(['My favourite condiment is'], return_tensors="pt").to("cuda")
    # print(test_input)
    # with torch.no_grad():
    #     outputs = unwrapped_model.generator.generate(
    #         input_ids=test_input["input_ids"],
    #         attention_mask=test_input['attention_mask'],
    #         max_length=unwrapped_model.opt.generation_max_length,
    #         min_length=unwrapped_model.opt.generation_min_length,
    #         return_dict_in_generate=False,
    #         output_scores=True,
    #         length_penalty=opt.generation_length_penalty
    #     )
    # print(unwrapped_model.generator_tokenizer.batch_decode(outputs)[0])
    # assert 1==0
    #test
    task = get_task(opt, generator_tokenizer)
    data_iterator = _get_eval_data_iterator(opt, data_path, task)
    if dist.get_rank()==0:
        iters = tqdm(enumerate(data_iterator))
        print('Total eval data:', len(data_iterator)*opt.eval_batch_size)
    else:
        iters = enumerate(data_iterator)
    

    for i, batch in iters:
        if i==len(data_iterator)*opt.eval_batch_size:
            break
        if 'passages' not in batch:
            continue
        query = batch.get("query", [""])
        answers = batch.get("target", [""])
        # if "woCNA" in data_path:
        #     if answers == "CANNOTANSWER" :
        #         continue
        batch_metadata = batch.get("metadata")
        target_tokens = batch.get("target_tokens")
        bsz, topk = len(query), opt.n_context
        #print(query)
        query_to_retrieve = query
        if unwrapped_model.opt.decoder_only:
            query_enc, decoder_input_ids = unwrapped_model.retriever_tokenize(query), None
        else:
            query_enc, labels, decoder_input_ids = unwrapped_model.tokenize(query_to_retrieve, answers, target_tokens)
        if not opt.use_file_passages:
            query_ids_retriever = query_enc["input_ids"].cuda()
            query_mask_retriever = query_enc["attention_mask"].cuda()
            retrieved_passages, _, _, _ = unwrapped_model.retrieve(
                index,
                topk,
                query_to_retrieve,
                query_ids_retriever,
                query_mask_retriever,
                batch_metadata=batch_metadata,
                filtering_fun=task.filter,
            )
            query_emb = unwrapped_model.retriever(**query_enc, is_passages=False)
            fstr = opt.retriever_format
            retriever_passages = [[fstr.format(**p) for p in example] for example in retrieved_passages]
            retriever_tok = encode_passages(retriever_passages, unwrapped_model.retriever_tokenizer,
                min(opt.text_maxlength, 512),
            )
            retriever_tokens = _to_cuda(retriever_tok) # (B, K, L)
            retriever_tokens = {k:v.reshape(-1, v.size(-1)) for k,v in retriever_tokens.items()} # (B*K, L)
            passage_emb = unwrapped_model.retriever(**retriever_tokens, is_passages=True) # (B*K, H)
            passage_emb = passage_emb.view(bsz, -1, passage_emb.size(-1)) # (B, K, H)

            ret_scores = torch.einsum("id, ijd->ij", [query_emb, passage_emb])
        else:
            retrieved_passages = [p[: 100] for p in batch["passages"]]

            query_emb = unwrapped_model.retriever(**query_enc, is_passages=False)
            fstr = opt.retriever_format
            retriever_passages = [[fstr.format(**p) for p in example] for example in retrieved_passages]
            retriever_tok = encode_passages(retriever_passages, unwrapped_model.retriever_tokenizer,
                min(opt.text_maxlength, 512),
            )
            retriever_tokens = _to_cuda(retriever_tok) # (B, K, L)
            retriever_tokens = {k:v.reshape(-1, v.size(-1)) for k,v in retriever_tokens.items()} # (B*K, L)
            passage_emb = unwrapped_model.retriever(**retriever_tokens, is_passages=True) # (B*K, H)
            passage_emb = passage_emb.view(bsz, -1, passage_emb.size(-1)) # (B, K, H)

            ret_scores = torch.einsum("id, ijd->ij", [query_emb, passage_emb])
            _, top_retrieved_doc_ids = ret_scores.topk(topk)
            ret_scores = torch.gather(ret_scores, 1, top_retrieved_doc_ids)
            retrieved_passages = [[retrieved_passages[i][j] for j in top_retrieved_doc_ids[i]] for i in range(bsz)]
        # calculate retriever metrics
        for b in range(len(retrieved_passages)):
            ps = [p['text'] for p in  retrieved_passages[b]]
            gt = [answers[b]]
            ret_cover.append(recall(ps, gt))

        # If example is a padding example then skip step
        if (len(query) == 0) or (len(query[0]) == 0):
            continue
        
        if unwrapped_model.opt.decoder_only:
            if opt.use_gradient_checkpoint_generator:
                unwrapped_model.generator.gradient_checkpointing_enable()
            input_tok, labels, attention_mask, _ = unwrapped_model.tokenize_casual(query, retrieved_passages, answers)
            eval_loss = 0
            if "eval_loss" in task.metrics:
                with torch.no_grad():
                    generator_output = unwrapped_model.generator(
                        input_ids=input_tok,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                    )
                    eval_loss, logits = generator_output[0].cpu().item(), generator_output[1]


            generate_inputs, attention_mask = unwrapped_model.tokenize_casual4gen(query, retrieved_passages) # (B*K, L)/if concat,(B,L)
            input_len = generate_inputs.size(1)
            rank = dist.get_rank()  # 获取当前进程的rank
            # if rank == 0:
            # if rank ==0:
            #     print(f"before_gen_input:{unwrapped_model.generator_tokenizer.decode(generate_inputs[0], skip_special_tokens=False)}")
            #     print(f"before_gen_inputids:{generate_inputs[0]}")
                
            
            # print(f"generation_test:{unwrapped_model.generator_tokenizer.decode(output[0],skip_special_tokens=False)}")
            # assert 1==0
            with torch.no_grad():
                #print(generate_inputs.dtype)
                outputs = unwrapped_model.generator.generate(
                    input_ids = generate_inputs,
                    attention_mask = attention_mask,
                    max_new_tokens=256,
                    return_dict_in_generate=True,
                    output_scores=True,
                    forced_bos_token_id=None,
                    length_penalty=opt.generation_length_penalty,
                    do_sample=False,
                    bos_token_id=bos_token_id,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )
            # if rank ==0:
            #     print(f"output_sequence:{outputs.sequences[0]}")
            #     # print(f"outputs1:{outputs.sequences[0]}")
            #     # print(f"output_test:{unwrapped_model.generator_tokenizer.decode(outputs.sequences[0])}")
            #     print(f"output_passage:{unwrapped_model.generator_tokenizer.decode(outputs.sequences[0],skip_special_tokens=False)}")
            # #     assert 1==0
            if opt.use_gradient_checkpoint_generator:
                unwrapped_model.generator.gradient_checkpointing_disable()
        else:
            generator_tokens, _ = unwrapped_model.tokenize_passages(query, retrieved_passages)
            if "eval_loss" in task.metrics:
                eval_loss, logits = unwrapped_model.compute_generator_loss_and_logits(generator_tokens, decoder_input_ids, labels)
            # if not opt.fid_training:
            #     generator_tokens = {k: v.view(bsz*topk, -1).unsqueeze(1) for k,v in generator_tokens.items()} # (B*K, 1, L)
            outputs = unwrapped_model.generate(
                generator_tokens, query, choices=batch["choices"] if "choices" in batch else None
            )
            
        generation, scores = outputs.sequences, outputs.scores # scores are before softmax
        #print(outputs.scores)
        output_sequence = generation
        if opt.decoder_only:
            generation = generation[:, input_len:] # (B*K, L_out)
        #for i, _ in enumerate(generation):
        #     if rank == 0:
        #         print(f"output_sequence:{unwrapped_model.generator_tokenizer.decode(output_sequence[i])}")
        #         print(f"output_answer:{unwrapped_model.generator_tokenizer.decode(generation[i])}")
        # assert 1==0
        metrics["eval_loss"].append(eval_loss)
        generation ,orin_gen= unwrapped_model.method_generate(generation=generation, scores=scores, bsz=bsz, topk=topk,ret_scores=ret_scores)
        print(generation.shape)
        #metrics caculate
        for k, g in enumerate(generation):
            if opt.decoder_prompt_format is not None:
                query_ids = generator_tokenizer.encode(
                    opt.decoder_prompt_format.format_map({"query": query[k]}), add_special_tokens=False
                )
                p = g[len(query_ids) + 1 :]
            else :
                p = g
            gen_list = []
            for j, og in enumerate(orin_gen[k]):
                ori = generator_tokenizer.decode(og, skip_special_tokens=True)
                gen_list.append(ori)
            #gen = generator_tokenizer.decode(g, skip_special_tokens=True)
            pred = generator_tokenizer.decode(p, skip_special_tokens=True)
            rank = dist.get_rank()  # 获取当前进程的rank
            print(answers)
            gold = [answers[k]] if not "answers" in batch else batch["answers"][k]
            # if rank == 0:
            #     #print(f"generationg:{g}")
            #     if opt.concat_doc:
            #         logger.info(f"pred:{pred}\ngold:{gold}")
            #     else:
            #         logger.info(f"topk_gen:{gen_list}\npred:{pred}\ngold:{gold}")

            sample_metrics = task.evaluation(pred, gold)

            for key, value in sample_metrics.items():
                metrics[key].append(value)

            if opt.write_results:
                ex = {"query": query[k], "answers": gold, "generation": pred}
                if not opt.dont_write_passages:
                    ex["passages"] = retrieved_passages[k]
                if batch_metadata is not None:
                    ex["metadata"] = batch_metadata[k]
                if opt.task == "multiple_choice":
                    ex["choice_logits"] = task.get_choice_logits(logits[k])
                if "id" in batch:
                    ex["id"] = batch["id"][k]
                dataset_wpred.append(ex)
    dist_utils.barrier()
    metrics, dataset_wpred = task.evaluation_postprocessing(metrics, dataset_wpred)
    metrics = util.avg_dist_dict(task.metrics, metrics)
    metrics = {key: value if key == "eval_loss" else 100 * value for key, value in metrics.items()}
    # metrics[f'coverage{opt.n_context}'] = sum(ret_cover)/len(ret_cover)
    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = f"{dataset_name}-step-{step}"
        util.save_distributed_dataset(dataset_wpred, dataset_name, opt)

    return metrics


if __name__ == "__main__":
    options = get_options()
    opt = options.parse()
    print(os.environ["WORLD_SIZE"])
    opt.local_rank = int(os.getenv('LOCAL_RANK', 0))
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    checkpoint_path, saved_index_path = create_checkpoint_directories(opt)

    logger = util.init_logger(opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, "run.log"))
    if opt.is_main:
        options.print_options(opt)

    logger.info(f"world size: {dist_utils.get_world_size()}")

    index, passages = load_or_initialize_index(opt)
    model, _, _, _, _, opt, step = load_or_initialize_atlas_model(opt, eval_only=True)

    logger.info("Start Evaluation")
    dist_utils.barrier()

    if opt.rebuild_index:
        model.build_index(index, passages, opt.per_gpu_embedder_batch_size, logger)

    if not opt.use_file_passages and opt.load_index_path is None:
        indexing_start = time.time()
        model.build_index(index, passages, opt.per_gpu_embedder_batch_size, logger)

    for data_path in opt.eval_data:
        dataset_name = os.path.basename(data_path)
        logger.info(f"Start Evaluation on {data_path}")
        if opt.retrieve_only:
            run_retrieval_only(model, index, opt, data_path, step)
        else:
            metrics = evaluate(model, index, opt, data_path, step, logger)
            log_message = f"Dataset: {dataset_name}"
            for k, v in metrics.items():
                log_message += f" | {v:.3f} {k}"
            logger.info(log_message)
    if opt.save_index_path is not None:
        save_embeddings_and_index(index, opt)