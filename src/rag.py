# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# In 2024/7/24, Cao Hongyu changed the original "atlas" to "rag".
from cProfile import label
import copy
import logging
import math
import psutil
import torch.distributed as dist
import time
from functools import reduce
from turtle import pos, register_shape
from typing import List, Optional, Union
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src import dist_utils
from src.retrievers import EMBEDDINGS_DIM, UntiedDualEncoderRetriever
#from async
import sys
sys.path.append(r"/home/bingxing2/home/scx7124/nlp_workspace/caohy/tasi_rag_platform-master")
from rebuildgrpc.async_init_build_client import run_retrieve
from rebuildgrpc.async_init_build_client import run_build
import torch
import asyncio
import threading
import time 
from post import call_retrieve_api
#from evaluate
from src import dist_utils, slurm, util
from tqdm import tqdm

logger = logging.getLogger(__name__)
IGNORE_INDEX: int = -100
BERT_MAX_SEQ_LENGTH: int = 512

class OptionalNoGrad(torch.no_grad):
    def __init__(self, condition):
        self.condition = condition
    
    def __enter__(self):
        if self.condition:
            super().__enter__()

def encode_passages(batch, tokenizer, max_length):
    bsz = len(batch)
    n = max([len(example) for example in batch])
    batch = [example + [""] * (n - len(example)) for example in batch]
    batch = reduce(lambda a, b: a + b, batch)
    # lengths = [len(tokenizer.encode(sent)) for sent in batch]
    # max_length = min(max_length, max(lengths))
    tokens = tokenizer(
        batch,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        truncation=True,
    )
    tokens = {k: v.view(bsz, n, -1) for k, v in tokens.items()}
    return tokens


class RAG(nn.Module):
    def __init__(self, opt, generator, retriever, generator_tokenizer, retriever_tokenizer,retriever_passage_tokenizer = None):
        super(RAG, self).__init__()

        self.generator = generator
        self.retriever = retriever
        self.generator_tokenizer = generator_tokenizer
        self.retriever_tokenizer = retriever_tokenizer
        self.retriever_passage_tokenizer = retriever_tokenizer if retriever_passage_tokenizer == None else retriever_passage_tokenizer
        self.post_retriever = None
        self.opt = opt
        self.test = False
        self.generator_ALL_TOKENS = list(self.generator_tokenizer.vocab.values())
        if self.opt.fix_encoder:
            self.retriever.requires_grad_(False)
        self.eps = 1e-10
        self.kl_beta = 1
        if opt.gold_score_mode in ['vrag', 'jsa'] and not opt.simplify_JSA:
            # need to initialize a posterior retriever
            if self.opt.decouple_encoder:
                # the passage encoder is the same as prior retriever, but the query encoder is independent
                post_passage_encoder = self.retriever.passage_retriever
                post_query_encoder = copy.deepcopy(self.retriever.query_retriever)
                self.post_retriever = UntiedDualEncoderRetriever(opt, post_query_encoder, post_passage_encoder)
            else:
                self.post_retriever = copy.deepcopy(self.retriever)

    def _get_fp16_retriever_copy(self, posterior=False):
        if hasattr(self.retriever, "module"):
            retriever_to_copy = self.retriever.module if not posterior else self.post_retriever.module
        else:
            retriever_to_copy = self.retriever if not posterior else self.post_retriever
        return copy.deepcopy(retriever_to_copy).half().eval()

    @torch.no_grad()
    def build_index(self, index, passages, gpu_embedder_batch_size, logger=None):
        n_batch = math.ceil(len(passages) / gpu_embedder_batch_size)
        retrieverfp16 = self._get_fp16_retriever_copy()
        logger.info(f"{len(passages)} passages ready for encode")
        total = 0
        for i in range(n_batch):
            batch = passages[i * gpu_embedder_batch_size : (i + 1) * gpu_embedder_batch_size]
            batch = [self.opt.retriever_format.format(**example) for example in batch]
            batch_enc = self.retriever_passage_tokenizer(
                batch,
                padding="longest",
                return_tensors="pt",
                max_length=self.opt.text_maxlength,
                truncation=True,
            )

            embeddings = retrieverfp16(**_to_cuda(batch_enc), is_passages=True)
            index.embeddings[:, total : total + len(embeddings)] = embeddings.T
            total += len(embeddings)
            if i % 50 == 0 and i > 0:
                logger.info(f"Number of passages encoded: {total}")
        dist_utils.barrier()
        logger.info(f"{total} passages encoded on process: {dist_utils.get_rank()}")

        if not index.is_index_trained():
            logger.info(f"Building faiss indices")

            index.train_index_bychunks()


    @torch.no_grad()
    def _retrieve(
        self,
        index,
        topk,
        query,
        query_ids_retriever,
        query_mask_retriever,
        batch_metadata=None,
        filtering_fun=None,
        iter_stats={},
        posterior = False
    ):
        
        retriever = self.post_retriever if posterior else self.retriever
        retriever.eval()
        if len(query) > 0:
            query_emb = retriever(query_ids_retriever, query_mask_retriever, is_passages=False)
        else:
            query_emb = torch.empty((0, EMBEDDINGS_DIM)).cuda()  # TODO: broken
        if self.training:
            retriever.train()

        search_start = time.time()
        if self.opt.grpc:
            passages ,passages_emb= asyncio.run(run_retrieve(query_emb=query_emb))
            scores = None
        elif self.opt.server:
            passages, scores = call_retrieve_api(query_emb, topk)
            passages_emb = None
        else: 
            if filtering_fun is not None:
                passages, scores, passages_emb = index.search_knn(query_emb, topk * self.opt.filtering_overretrieve_ratio)
                passages, scores = filtering_fun(batch_metadata, passages, scores, topk, training=self.training)
            else:
                passages, scores = index.search_knn(query_emb, topk)
                passages_emb = None
        iter_stats["runtime/search"] = (time.time() - search_start, 1)

        # print(f"passages:{passages}\nscores:{scores}\nquery_emb:{query_emb}\npassages_emb:{passages_emb}")
        # assert 1==0
        return passages, scores, query_emb, passages_emb

    @torch.no_grad()
    def retrieve_with_rerank(
        self,
        index,
        topk,
        query,
        query_ids_retriever,
        query_mask_retriever,
        batch_metadata=None,
        filtering_fun=None,
        iter_stats={},
        posterior=False
    ):
        bsz = len(query)
        to_rerank = self.opt.n_to_rerank_with_retrieve_with_rerank

        # first, do the retrieval
        passages, _, query_emb, _ = self._retrieve(
            index,
            to_rerank,
            query,
            query_ids_retriever,
            query_mask_retriever,
            batch_metadata,
            filtering_fun,
            iter_stats,
            posterior
        )

        retrieverfp16 = self._get_fp16_retriever_copy(posterior=posterior)
        retrieverfp16.eval()
        fstr = self.opt.retriever_format
        flat_passage_strings = [fstr.format(**p) for ps in passages for p in ps]
        encoder_batch_size = min(len(flat_passage_strings), self.opt.per_gpu_embedder_batch_size)
        passage_emb, output_passages, output_scores = (
            query_emb.new_zeros(len(flat_passage_strings), query_emb.shape[-1]),
            [],
            [],
        )

        for b in range(0, len(flat_passage_strings), encoder_batch_size):
            batch = flat_passage_strings[b : b + encoder_batch_size]
            batch_enc = self.retriever_tokenizer(
                batch,
                padding="longest",
                return_tensors="pt",
                max_length=min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH),
                truncation=True,
            )
            batch_emb = retrieverfp16(**_to_cuda(batch_enc), is_passages=True).to(query_emb.device)
            passage_emb[b : b + encoder_batch_size] = batch_emb

        passage_emb = passage_emb.view(bsz, to_rerank, -1) # (B, L, H)
        retriever_scores = torch.einsum("id, ijd->ij", [query_emb, passage_emb]) #(B, H), (B, L, H) --> (B, L)
        sorted_scores, sorted_ids = torch.sort(retriever_scores, dim=-1, descending=True) # (B, L)

        top_retriever_scores, top_retriever_inds = sorted_scores[:, :topk], sorted_ids[:, :topk] # (B, K)
        topk_indices_dup = top_retriever_inds.unsqueeze(2).expand(bsz, topk, passage_emb.size(-1)) # (B, K, H)
        topk_passage_embd = torch.gather(passage_emb, 1, topk_indices_dup) # (B, K, H)

        # some metrics
        MRR = 1/(top_retriever_inds.float()+1).mean(-1).mean().item()
        iter_stats["MRR"] = (MRR, len(query))
        _, rev_sorted_ids = torch.sort(sorted_ids, dim=-1)
        MRR_rev = 1/(rev_sorted_ids[:, :topk].float()+1).mean(-1).mean().item()
        iter_stats["MRR_rev"] = (MRR_rev, len(query))


        for i in range(bsz):
            output_passages.append([passages[i][j] for j in top_retriever_inds[i]])
            output_scores.append(top_retriever_scores[i].tolist())
        return output_passages, output_scores, query_emb, topk_passage_embd

    def rerank(
        self,
        query,
        passages,
        model
        ):
        pass

    @torch.no_grad()
    def retrieve(self, *args, **kwargs):
        retrieve_func = self.retrieve_with_rerank if self.opt.retrieve_with_rerank else self._retrieve
        return retrieve_func(*args, **kwargs)
        # return passages, scores

    def append_query(self, query, passages):
        return [self.opt.encoder_format.format(query=query, **p) for p in passages]

    def retriever_tokenize(self, query):
        if self.retriever_tokenizer:
            query_enc = self.retriever_tokenizer(
                query,
                max_length=min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            query_enc = _to_cuda(query_enc)
        else:
            query_enc = None
        return _to_cuda(query_enc)

    def generator_tokenize(self, query, target, target_tokens):
        if target_tokens is None:
            if self.opt.decoder_prompt_format is not None:
                modified_query = [self.opt.decoder_prompt_format.format_map({"query": q}) for q in query]
                target = [q + t for (q, t) in zip(modified_query, target)]

                query_mask = self.generator_tokenizer(
                    modified_query,
                    max_length=self.opt.target_maxlength,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["attention_mask"]

            if self.opt.decoder_format is not None:
                target = [self.opt.decoder_format.format(target=t) for t in target]
            target = [t + "</s>" if not t.endswith("</s>") else t for t in target]

            target_tokens = self.generator_tokenizer(
                target,
                max_length=self.opt.target_maxlength,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )

        decoder_input_ids = self.generator._shift_right(target_tokens["input_ids"])#"<s>0,1,2"
        labels = target_tokens["input_ids"].masked_fill(~target_tokens["attention_mask"].bool(), IGNORE_INDEX)#"0,1,2"

        # If decoder prompt is not None mask labels such that the model is not trained to predict the prompt
        if self.opt.decoder_prompt_format is not None:
            query_mask = self.generator_tokenizer(
                modified_query,
                max_length=self.opt.target_maxlength,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )["attention_mask"]

            padding = torch.zeros((query_mask.size(0), target_tokens["input_ids"].size(-1) - query_mask.size(-1)))
            query_mask = torch.cat([query_mask, padding], dim=1)
            labels = labels.masked_fill(query_mask.bool(), IGNORE_INDEX)

        return labels.cuda(), decoder_input_ids.cuda()

    def tokenize(self, query, target, target_tokens):
        if query is None and target is None:
            return None, None, None

        assert (
            target_tokens is None or self.opt.decoder_prompt_format is None
        ), "decoder_prompt_format not compatible with target tokenized in iterator"

        query_enc = self.retriever_tokenize(query)
        labels, decoder_input_ids = self.generator_tokenize(query, target, target_tokens)
        return query_enc, labels, decoder_input_ids

    def tokenize_passages(self, query, passages):
        if len(query) == 0:
            return None, None

        query_passages = [self.append_query(q, p) for q, p in zip(query, passages)]

        fstr = self.opt.retriever_format
        retriever_passages = [[fstr.format(**p) for p in example] for example in passages]
        if self.retriever_passage_tokenizer:
            retriever_tok = encode_passages(
                retriever_passages,
                self.retriever_passage_tokenizer,
                min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH),
            )
            retriever_tok = _to_cuda(retriever_tok)
        else:
            retriever_tok = None
        generator_tok = encode_passages(query_passages, self.generator_tokenizer, self.opt.text_maxlength)
        generator_tok = _to_cuda(generator_tok) # (B, K, L)
        return generator_tok, retriever_tok
    
    def get_passages_tokens(self,passages):
        fstr = self.opt.retriever_format
        passages_text = [[fstr.format(**p) for p in example] for example in passages]
        retriever_tokens = encode_passages(
            passages_text, self.retriever_passage_tokenizer, min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH))
        retriever_tokens = {k: v.reshape(-1, v.size(-1)) for k, v in retriever_tokens.items()}
        retriever_tokens = _to_cuda(retriever_tokens)
        return retriever_tokens


    def get_condition_format(self, q, p):
        if 'GPT' in self.opt.generator_model_type:
            if self.opt.concat_doc:
                condition_text = self.generator_tokenizer.bos_token + f'Give a short answer to the Question based on relevant information given in Input.\nInput:{p}\nQuestion: {q}\n'
            if self.opt.dialog:
                context = 'title: {} context: {}'.format(p['title'], p['text'])
                condition_text = self.generator_tokenizer.bos_token + f'Give an answer or response to the dialog based on relevant information given in the Input.\ndialog: {q}\nInput:{context}\n'
            else:
                context = 'title: {} context: {}'.format(p['title'], p['text'])
                condition_text = self.generator_tokenizer.bos_token + f'Give a short answer to the Question based on relevant information given in Input.\nInput:{context}\n'
        elif 'llama' in self.opt.generator_model_type or 'mistral' in self.opt.generator_model_type:
            if self.opt.concat_doc:
                condition_text = self.generator_tokenizer.bos_token + f'[INST] Give a short answer to the Question based on relevant information given in Input.\nInput:{p}'
            else:
                context = 'title: {} context: {}'.format(p['title'], p['text'])
                condition_text = self.generator_tokenizer.bos_token + f'[INST] Give a short answer to the Question based on relevant information given in Input.\nInput:{context}'
        return condition_text

    def tokenize_casual(self, query, passages, target, concat_doc=False):
        # tokenize passages for casual LM
        fstr = self.opt.retriever_format
        training_ids = []
        training_labels = []
        training_masks = []
        if self.opt.concat_doc:
            for q, ps, t in zip(query, passages, target):
                context_list = []
                for p in ps:
                    context_list.append('title: {} context: {}'.format(p['title'], p['text']))
                query_text = f"\nQuestion: {q}\n[/INST]"
                concat_context = '\n'.join(context_list)
                condition_text = self.get_condition_format(q, concat_context)
                target_text = t + self.generator_tokenizer.eos_token
                query_ids = self.generator_tokenizer(text=query_text,
                                                    max_length=self.opt.target_maxlength, 
                                                    truncation=True,
                                                    add_special_tokens=False)['input_ids']
                condition_ids = self.generator_tokenizer(text=condition_text,
                                max_length=self.opt.text_maxlength, 
                                truncation=True,
                                add_special_tokens=False)['input_ids']
                target_ids = self.generator_tokenizer(text=target_text,
                                    max_length=self.opt.target_maxlength, 
                                    truncation=True,
                                    add_special_tokens=False)['input_ids']
                max_len = self.opt.text_maxlength 
                if len(condition_ids)+len(target_ids)+len(query_ids)>self.opt.text_maxlength:
                    condition_ids = condition_ids[:self.opt.text_maxlength-(len(target_ids)+len(query_ids))]

                context_ids = condition_ids
                context_ids = condition_ids + target_ids#ids拼接
                mask_ids = len(context_ids)*[1]#长len(context_ids)
                label_ids = len(condition_ids)*[IGNORE_INDEX] + target_ids#长len(context_ids),前0后1
                # context = q + 'title: {} context: {}'.format(p['title'], p['text']) + 'answer: '+ t
                training_ids.append(context_ids)
                training_labels.append(label_ids)
                training_masks.append(mask_ids)
        elif 'GPT' in self.opt.generator_model_type:
            for q, ps, t in zip(query, passages, target):
                for p in ps:
                    context = 'title: {} context: {}'.format(p['title'], p['text'])
                    target_text = t + self.generator_tokenizer.eos_token
                    context_ids = self.generator_tokenizer(text=context, 
                                                            max_length=self.opt.text_maxlength, 
                                                            truncation=True,
                                                            add_special_tokens=False)['input_ids']
                    target_ids = self.generator_tokenizer(text=target_text,
                                                        max_length=self.opt.target_maxlength, 
                                                        truncation=True,
                                                        add_special_tokens=False)['input_ids']
                    dialog_ids = self.generator_tokenizer(text=q,
                                                        max_length=self.opt.target_maxlength, 
                                                        truncation=True,
                                                        add_special_tokens=False)['input_ids']
                    if len(context_ids)+len(target_ids)+len(dialog_ids)>self.opt.text_maxlength:
                        context_ids = context_ids[:self.opt.text_maxlength-(len(target_ids)+len(dialog_ids))] #如果大于最大输入长度，就把passage部分截断，保证target完整
                    rank = dist.get_rank()
                    # if rank ==0:
                    #     print(target_text)
                    #     print(condition_text)
                    #     print(condition_ids)
                    #     print(target_ids)
                    #     print(self.generator_tokenizer.decode(condition_ids, skip_special_tokens=False))
                    # assert 1==0
                    condition_ids = context_ids + dialog_ids
                    sent_ids = condition_ids +  target_ids #ids拼接
                    mask_ids = len(sent_ids)*[1]#长len(context_ids)
                    # print(mask_ids)
                    # assert 1==0
                    label_ids = len(condition_ids)*[IGNORE_INDEX] + target_ids #长len(context_ids),前0后1
                    # context = q + 'title: {} context: {}'.format(p['title'], p['text']) + 'answer: '+ t
                    # print(len(sent_ids))
                    # print(len(mask_ids))
                    # print(len(label_ids))
                    
                    training_ids.append(sent_ids)
                    training_labels.append(label_ids)
                    training_masks.append(mask_ids)
        else:
            for q, ps, t in zip(query, passages, target):
                for p in ps:
                    condition_text = self.get_condition_format(q, p)
                    target_text = t + self.generator_tokenizer.eos_token
                    query_text = f"\nQuestion: {q}\n[/INST]"
                    condition_ids = self.generator_tokenizer(text=condition_text, 
                                                            max_length=self.opt.text_maxlength, 
                                                            truncation=True,
                                                            add_special_tokens=False)['input_ids']
                    target_ids = self.generator_tokenizer(text=target_text,
                                                        max_length=self.opt.target_maxlength, 
                                                        truncation=True,
                                                        add_special_tokens=False)['input_ids']
                    query_ids = self.generator_tokenizer(text=query_text,
                                                        max_length=self.opt.target_maxlength, 
                                                        truncation=True,
                                                        add_special_tokens=False)['input_ids']
                    if len(condition_ids)+len(target_ids)+len(query_ids)>self.opt.text_maxlength:
                        condition_ids = condition_ids[:self.opt.text_maxlength-(len(target_ids)+len(query_ids))] #如果大于最大输入长度，就把passage部分截断，保证target完整
                    rank = dist.get_rank()
                    # if rank ==0:
                    #     print(target_text)
                    #     print(condition_text)
                    #     print(condition_ids)
                    #     print(target_ids)
                    #     print(self.generator_tokenizer.decode(condition_ids, skip_special_tokens=False))
                    # assert 1==0
                    condi_and_q_ids = condition_ids + query_ids
                    context_ids = condi_and_q_ids +  target_ids #ids拼接
                    mask_ids = len(context_ids)*[1]#长len(context_ids)
                    # print(mask_ids)
                    # assert 1==0
                    label_ids = len(condi_and_q_ids)*[IGNORE_INDEX] + target_ids #长len(context_ids),前0后1
                    # context = q + 'title: {} context: {}'.format(p['title'], p['text']) + 'answer: '+ t
                    training_ids.append(context_ids)
                    training_labels.append(label_ids)
                    training_masks.append(mask_ids)
        
        #right padding
        max_len = max([len(ids) for ids in training_ids])
        training_ids = torch.tensor([ids + (max_len-len(ids))*[self.generator_tokenizer.pad_token_id] for ids in training_ids], dtype=torch.long).cuda() # (B*K, L)
        training_labels = torch.tensor([ids + (max_len-len(ids))*[IGNORE_INDEX] for ids in training_labels], dtype=torch.long).cuda()
        training_masks = torch.tensor([ids + (max_len-len(ids))*[0] for ids in training_masks], dtype=torch.long).cuda()
        #training_masks = training_ids.ne(self.generator_tokenizer.pad_token_id).cuda()#原pad_token?
        # tokenize retrieved passages
        retriever_passages = [[fstr.format(**p) for p in example] for example in passages]
        if self.retriever_tokenizer and self.opt.gold_score_mode == 'rag':
            retriever_tok = encode_passages(
                retriever_passages,
                self.retriever_tokenizer,
                min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH),
            )
            retriever_tok = _to_cuda(retriever_tok)
        else:
            retriever_tok = None
        return training_ids, training_labels, training_masks, retriever_tok


    def tokenize_casual4gen(self, query, passages, concat_doc=False):
        # tokenize passages for casual LM generation
        fstr = self.opt.retriever_format
        training_ids = []
        training_masks = []
        if self.opt.concat_doc:
            for q, ps in zip(query, passages):
                context_list = []
                for p in ps:
                    context_list.append('title: {} context: {}'.format(p['title'], p['text']))
                query_text = f"\nQuestion: {q}\n[/INST]"
                concat_context = '\n'.join(context_list)
                condition_text = self.get_condition_format(q, concat_context)
                #target_text = t + self.generator_tokenizer.eos_token
                condition_ids = self.generator_tokenizer(text=condition_text,
                                max_length=self.opt.text_maxlength, 
                                truncation=True,
                                add_special_tokens=False)['input_ids']
                query_ids = self.generator_tokenizer(text=query_text,
                                                    max_length=self.opt.target_maxlength, 
                                                    truncation=True,
                                                    add_special_tokens=False)['input_ids']
                max_len = self.opt.text_maxlength 
                if len(condition_ids)+len(query_ids)>self.opt.text_maxlength:
                    condition_ids = condition_ids[:self.opt.text_maxlength-len(query_ids)]

                context_ids = condition_ids
                context_ids = condition_ids + query_ids#ids拼接
                mask_ids = len(context_ids)*[1]#长len(context_ids)
                #label_ids = len(condition_ids)*[IGNORE_INDEX] + target_ids#长len(context_ids),前0后1
                # context = q + 'title: {} context: {}'.format(p['title'], p['text']) + 'answer: '+ t
                training_ids.append(context_ids)
                #training_labels.append(label_ids)
                training_masks.append(mask_ids)
        elif 'GPT' in self.opt.generator_model_type:
            for q, ps in zip(query, passages):
                for p in ps:
                    context = 'title: {} context: {}'.format(p['title'], p['text'])
                    context_ids = self.generator_tokenizer(text=context, 
                                                            max_length=self.opt.text_maxlength, 
                                                            truncation=True,
                                                            add_special_tokens=False)['input_ids']
                    dialog_ids = self.generator_tokenizer(text=q,
                                                        max_length=self.opt.target_maxlength, 
                                                        truncation=True,
                                                        add_special_tokens=False)['input_ids']
                    if len(context_ids)+len(dialog_ids)>self.opt.text_maxlength:
                        context_ids = context_ids[:self.opt.text_maxlength-len(dialog_ids)] #如果大于最大输入长度，就把passage部分截断，保证target完整
                    rank = dist.get_rank()
                    # if rank ==0:
                    #     print(target_text)
                    #     print(condition_text)
                    #     print(condition_ids)
                    #     print(target_ids)
                    #     print(self.generator_tokenizer.decode(condition_ids, skip_special_tokens=False))
                    # assert 1==0
                    condition_ids = context_ids + dialog_ids
                    sent_ids = copy.deepcopy(condition_ids) #ids拼接
                    mask_ids = len(sent_ids)*[1]#长len(context_ids)

                    
                    training_ids.append(sent_ids)
                    training_masks.append(mask_ids)
        else:
            for q, ps in zip(query, passages):
                for p in ps:
                    condition_text = self.get_condition_format(q, p)
                    query_text = f"\nQuestion: {q}\n[/INST]"
                    condition_ids = self.generator_tokenizer(text=condition_text, 
                                                            max_length=self.opt.text_maxlength, 
                                                            truncation=True,
                                                            add_special_tokens=False)['input_ids']

                    query_ids = self.generator_tokenizer(text=query_text,
                                                        max_length=self.opt.target_maxlength, 
                                                        truncation=True,
                                                        add_special_tokens=False)['input_ids']
                    if len(condition_ids)+len(query_ids)>self.opt.text_maxlength:
                        condition_ids = condition_ids[:self.opt.text_maxlength-len(query_ids)] #如果大于最大输入长度，就把passage部分截断，保证target完整
                    rank = dist.get_rank()
                    # if rank ==0:
                    #     print(target_text)
                    #     print(condition_text)
                    #     print(condition_ids)
                    #     print(target_ids)
                    #     print(self.generator_tokenizer.decode(condition_ids, skip_special_tokens=False))
                    # assert 1==0
                    condi_and_q_ids = condition_ids + query_ids
                    context_ids = copy.deepcopy(condi_and_q_ids)#ids拼接
                    mask_ids = len(context_ids)*[1]#长len(context_ids)
                    # print(mask_ids)
                    # assert 1==0
                    training_ids.append(context_ids)
                    training_masks.append(mask_ids)
        
        
        max_len = max([len(ids) for ids in training_ids])
        # pad from the left
        training_ids = torch.tensor([(max_len-len(ids))*[self.generator_tokenizer.pad_token_id] + ids for ids in training_ids], dtype=torch.long).cuda() # (B*K, L)
        training_masks = torch.tensor([(max_len-len(ids))*[0] + ids  for ids in training_masks]).cuda()
        #training_masks = training_ids.ne(self.generator_tokenizer.pad_token_id).cuda()
        return training_ids, training_masks


    def SelectDoc(self, documents, indices):
        # documents: a list of B*K*(text)
        # indices: tensor (B,n)
        B, n = indices.size()
        results = []
        for i in range(B):
            result = []
            for j in range(n):
                result.append(documents[i][indices[i][j]])
            results.append(result)
        return results
    
    def process_passages(self, passages, encoder):
        bsz = len(passages)
        fstr = self.opt.retriever_format
        passages_text = [[fstr.format(**p) for p in example] for example in passages]
        tokens = encode_passages(passages_text, self.retriever_passage_tokenizer, min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH))
        tokens = _to_cuda(tokens) # (B, n, L) for each value
        tokens = {k: v.reshape(-1, v.size(-1)) for k, v in tokens.items()} # (B*n, L)
        passage_emb = encoder(**tokens, is_passages=True) # (B*n, H)
        passage_emb = passage_emb.view(bsz, -1, passage_emb.size(-1)) # (B,n,H)
        return passage_emb

    def forward(
        self,
        index,
        query,
        target,
        target_tokens=None,
        passages=None,
        batch_metadata=None,
        filtering_fun=None,
        use_cache=False,
        train_retriever=False,
        iter_stats={},
    ):
        forward_start = time.time()
        bsz = len(query)
        n_context_training = self.opt.n_context
        topk = self.opt.n_context
        cfg = self.generator.encoder.config if not self.opt.decoder_only else None
        #print(query)
        # query_mask_generator = (
        #     self.generator_tokenizer.batch_encode_plus(
        #         query,
        #         max_length=self.opt.text_maxlength,
        #         padding="longest",
        #         truncation=True,
        #         return_tensors="pt",
        #         add_special_tokens=False,
        #     )["attention_mask"]
        #     .bool()
        #     .cuda()
        # )
        if 'GPT' in self.opt.generator_model_type:
            query_to_retrieve = [remove_speakers(q)for i,q in enumerate(query)]
        else:
            query_to_retrieve = query
        query_to_retrieve = [remove_speakers(q)for i,q in enumerate(query)]
        training_info = {
            'query': query[0],
            'response': target[0]
        }
        # if self.opt.gold_score_mode=='jsa1':
        #     KL = None
        #     if self.opt.simplify_JSA:
        #         query_enc, labels, decoder_input_ids = self.tokenize(query_to_retrieve, target, target_tokens)#query_enc得到retriever的enc，其他事gen的ids
        #         passages, _, _, passage_emb = self.retrieve(
        #             index,
        #             self.opt.retriever_n_context,
        #             query_to_retrieve,
        #             query_enc["input_ids"],
        #             query_enc["attention_mask"],
        #             batch_metadata=batch_metadata,
        #             filtering_fun=filtering_fun,
        #             iter_stats=iter_stats,
        #         )
        #         passage_emb = passage_emb.to(torch.float32)
        #         training_info['Retrieved_top10_passages'] = [p['text'] for p in passages[0][:10]]
        #         query_emb = self.retriever(**query_enc, is_passages=False)
        #         logits = torch.einsum("id, ijd->ij", [query_emb, passage_emb])
        #         probs = F.softmax(logits/self.opt.temperature_jsa, dim=-1)
        #         # MIS steps
        #         total_proposed_ids = torch.multinomial(probs, self.opt.mis_step, replacement=True) # (B, n)
        #         training_info['Proposal_indices'] = ','.join([str(item) for item in total_proposed_ids[0,:].cpu().tolist()])
        #         self.generator.eval()
        #         with torch.no_grad():
        #             total_proposed_passages = self.SelectDoc(passages, total_proposed_ids) # a list of texts of [B, n]
        #             accept_rate = 0
        #             total_turns = 0# calculate lm probs on all the proposed passages
        #             cfg.n_context = 1
        #             cfg.bsz = bsz * self.opt.mis_step
        #             generator_tokens, _ = self.tokenize_passages(query, total_proposed_passages) # (B, n, L)
        #             generator_ids = generator_tokens["input_ids"].view(cfg.bsz, -1) # (B*n, L_in)
        #             generator_mask = generator_tokens["attention_mask"].bool().view(cfg.bsz, -1) # (B*n, L_in)
        #             repeated_labels = torch.repeat_interleave(labels, self.opt.mis_step, dim=0) # (B*n, L_out)
        #             generator_output = self.generator(
        #                 input_ids=generator_ids.cuda(),
        #                 attention_mask=generator_mask.cuda(),
        #                 labels=repeated_labels,
        #                 use_cache=False
        #             )
        #             log_lm_probs = F.log_softmax(generator_output.logits, dim=-1) # (B*n, L_out, V)
        #             labels4gather = repeated_labels.masked_fill(repeated_labels==-100, 0)
        #             log_lm_probs = torch.gather(log_lm_probs, -1, labels4gather.unsqueeze(-1)).squeeze(-1) # (B*n, L_out)
        #             log_lm_probs = (log_lm_probs*(repeated_labels!=-100)).sum(-1).view(bsz, -1) # (B, n)
        #             cfg.n_context = n_context_training
        #             cfg.bsz = bsz
        #             total_sampled_ids = []
        #             for i in range(self.opt.mis_step):
        #                 proposed_idx = total_proposed_ids[:, i].unsqueeze(1) # (B, 1)
        #                 log_lm_prob = log_lm_probs[:, i]
        #                 if i==0:
        #                     sampled_idx = proposed_idx.clone()
        #                 else:
        #                     sampled_idx = pv_sampled_idx.clone()
        #                     accept_probs = torch.exp((log_lm_prob-pv_log_lm_prob)/self.opt.temperature_lm)
        #                     for j in range(bsz):
        #                         rand_num = random.random()
        #                         total_turns += 1
        #                         if rand_num<=accept_probs[j]:
        #                             accept_rate += 1
        #                             sampled_idx[j,:] = proposed_idx[j,:]
        #                         else:
        #                             log_lm_prob[j] = pv_log_lm_prob[j]
        #                 total_sampled_ids.append(sampled_idx)
        #                 pv_log_lm_prob = log_lm_prob
        #                 pv_sampled_idx = sampled_idx
        #         self.generator.train()
        #         # calculate probs with gradient
        #         total_sampled_ids = torch.cat(total_sampled_ids, dim=1) # (B, n)
        #         sampled_ids = total_sampled_ids[:, -self.opt.training_sample_num:]
        #         # sampled_ids = torch.cat(total_sampled_ids[-self.opt.training_sample_num:], dim=1) # (B, n)
        #         training_info['Accept_rate'] = accept_rate
        #         training_info['MIS_sampled_indices'] = ','.join([str(item) for item in sampled_ids[0,:].cpu().tolist()])
        #         sampled_passages = self.SelectDoc(passages, sampled_ids) # a list of texts of [B, n]
        #         training_info['MIS_sampled_passages'] = [p['text'] for p in sampled_passages[0]]
        #         if self.opt.use_gradient_checkpoint_retriever and "dpr" not in self.opt.retriever_model_path :
        #             self.retriever.gradient_checkpointing_enable()
        #         if not self.opt.query_side_retriever_training and self.opt.reduce_norm:
        #             # positive scores
        #             sampled_prior_embed = self.process_passages(sampled_passages, self.retriever)
        #             sampled_prior_scores = torch.einsum("id, ijd->ij", [query_emb.detach(), sampled_prior_embed])
        #             # negative scores
        #             neg_prior_ids = torch.multinomial(probs, self.opt.training_sample_num, replacement=True)
        #             neg_prior_passages = self.SelectDoc(passages, neg_prior_ids)
        #             neg_prior_embed = self.process_passages(neg_prior_passages, self.retriever)
        #             neg_prior_scores = torch.einsum("id, ijd->ij", [query_emb.detach(), neg_prior_embed])
        #             training_info['Prior_sampled_ids'] = neg_prior_ids[0,:].cpu().tolist()
        #             training_info['Prior_sampled_passages'] = neg_prior_passages[0]
        #             # calculate loss
        #             prior_obj = (sampled_prior_scores.mean(-1) - neg_prior_scores.mean(-1)) # (B, )
        #             # add log prob loss
        #             log_prior_prob = torch.log(torch.gather(probs, 1, sampled_ids) + self.eps).mean(-1) # (B,)
        #             log_prior_prob += prior_obj
        #         else:
        #             log_prior_prob = torch.log(torch.gather(probs, 1, sampled_ids) + self.eps).mean(-1) # (B,)
        #         if self.opt.use_gradient_checkpoint_retriever and "dpr" not in self.opt.retriever_model_path :
        #             self.retriever.gradient_checkpointing_disable()
        #         generator_tokens, _ = self.tokenize_passages(query, sampled_passages) # (B, n, L)
        #         if self.opt.use_gradient_checkpoint_generator:
        #             self.generator.gradient_checkpointing_enable()
        #         if self.opt.fid_training:
        #             cfg.bsz = bsz
        #             cfg.n_context = self.opt.training_sample_num
        #             generator_ids = generator_tokens["input_ids"].view(cfg.bsz, -1) # (B, n*L_in)
        #             generator_mask = generator_tokens["attention_mask"].bool().view(cfg.bsz, -1) # (B, n*L_in)
        #             generator_output = self.generator(
        #                 input_ids=generator_ids.cuda(),
        #                 attention_mask=generator_mask.cuda(),
        #                 decoder_input_ids=decoder_input_ids,
        #                 labels=labels,
        #                 use_cache=False,
        #             )
        #             generator_loss = generator_output[0]
        #         else:
        #             cfg.n_context = 1
        #             cfg.bsz = bsz * self.opt.training_sample_num
        #             generator_ids = generator_tokens["input_ids"].view(cfg.bsz, -1) # (B*n, L_in)
        #             generator_mask = generator_tokens["attention_mask"].bool().view(cfg.bsz, -1) # (B*n, L_in)
        #             repeated_labels = torch.repeat_interleave(labels, self.opt.training_sample_num, dim=0) # (B*n, L_in)
        #             repeated_decoder_input_ids = torch.repeat_interleave(decoder_input_ids, self.opt.training_sample_num, dim=0)
        #             generator_output = self.generator(
        #                 input_ids=generator_ids.cuda(),
        #                 attention_mask=generator_mask.cuda(),
        #                 decoder_input_ids=repeated_decoder_input_ids,
        #                 labels=repeated_labels,
        #                 use_cache=False
        #             )
        #             cfg.n_context = n_context_training
        #             cfg.bsz = bsz
        #             log_lm_prob = F.log_softmax(generator_output.logits, dim=-1) # (B*n, L_out, V)
        #             labels4gather = repeated_labels.masked_fill(repeated_labels==-100, 0) # (B*n, L_out)
        #             log_lm_prob = torch.gather(log_lm_prob, -1, labels4gather.unsqueeze(-1)).squeeze(-1) # (B*n, L_out)
        #             generator_loss = -(log_lm_prob*(repeated_labels!=-100)).sum(-1) # (B*n,)
        #             generator_loss = generator_loss.view(bsz, -1).mean(-1).mean()
        #         retriever_loss = -log_prior_prob.mean()
        #         if self.opt.use_gradient_checkpoint_generator:
        #             self.generator.gradient_checkpointing_disable()
        #     else:
        #         post_query = [q+' [SEP] '+t for q, t in zip(query_to_retrieve, target)]
        #         post_query_enc = self.retriever_tokenize(post_query)
        #         prior_query_enc = self.retriever_tokenize(query_to_retrieve)
        #         #prior_query_enc, labels, decoder_input_ids = self.tokenize(query_to_retrieve, target, target_tokens)
        #         #decoder_input_ids, labels, decoder_mask, retriever_tokens = self.tokenize_casual(query, passages, target)
        #         if not self.opt.use_file_passages:
        #             passages, scores, _, passages_emb  = self.retrieve(
        #                 index,
        #                 self.opt.retriever_n_context,
        #                 post_query,
        #                 post_query_enc["input_ids"],
        #                 post_query_enc["attention_mask"],
        #                 batch_metadata=batch_metadata,
        #                 filtering_fun=filtering_fun,
        #                 iter_stats=iter_stats,
        #                 posterior=True
        #             )
        #             training_info['Retrieved_top20_passages'] = [p['text'] for p in passages[0][:20]]
        #             #passages_emb = passages_emb.to(torch.float32)
        #         else:
        #             passages = [p[:self.opt.retriever_n_context] for p in passages]

        #         fstr = self.opt.retriever_format
        #         passages_text = [[fstr.format(**p) for p in example] for example in passages]
        #         retriever_tokens = encode_passages(
        #             passages_text, self.retriever_passage_tokenizer, min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH))
        #         retriever_tokens = {k: v.reshape(-1, v.size(-1)) for k, v in retriever_tokens.items()}
        #         retriever_tokens = _to_cuda(retriever_tokens) # (B, 100, L) for each value
        #         # print(retriever_tokens["input_ids"].shape)
        #         passage_emb = self.retriever(**retriever_tokens, is_passages=True,grad_no_pass=True)
        #         passage_emb = passage_emb.view(bsz, -1, passage_emb.size(-1))


        #         #print(passage_emb,"\n",post_passage_emb_wg_topk)
        #         # calculate posterior probs over topk
        #         if self.opt.use_gradient_checkpoint_retriever and "dpr" not in self.opt.retriever_model_path :
        #             self.post_retriever.gradient_checkpointing_enable()
        #             self.retriever.gradient_checkpointing_enable()

        #         post_query_emb = self.post_retriever(**post_query_enc, is_passages=False)
        #         if self.opt.query_side_retriever_training or self.opt.decouple_encoder:
        #             post_passage_emb = passage_emb
        #         else:
        #             if self.opt.passages_with_grad==-1:
        #                 # directly feed all the retrieved passages into the passage encoder
        #                 retriever_tokens = {k: v.reshape(-1, v.size(-1)) for k, v in retriever_tokens.items()} # (B*K, L)
        #                 post_passage_emb = self.post_retriever(**retriever_tokens, is_passages=True).to(post_query_emb) # (B*K, H)
        #                 post_passage_emb = post_passage_emb.view(bsz, -1, post_passage_emb.size(-1)) # (B,K,H)
        #             else:
        #                 # randomly select some passages and feed them into the passage encoder 
        #                 # weights = torch.ones(bsz, self.opt.retriever_n_context).cuda() # (B, K)
        #                 # passage_ids = torch.multinomial(weights, self.opt.passages_with_grad) # (B, g)
        #                 # select top passages and feed them into the passage encoder
        #                 passage_ids = torch.arange(self.opt.passages_with_grad).expand(bsz, self.opt.passages_with_grad).cuda()
        #                 passage_ids_dup = passage_ids.unsqueeze(-1).expand(passage_ids.shape + (BERT_MAX_SEQ_LENGTH, )) # (B, g) --> (B, g, L)
        #                 seleted_tokens = {k: torch.gather(v, 1, passage_ids_dup).reshape(-1, v.size(-1)) for k, v in retriever_tokens.items()} # (B*g, L)
        #                 seleted_passage_emb = self.post_retriever(**seleted_tokens, is_passages=True).to(post_query_emb) # (B*g, H)
        #                 seleted_passage_emb = seleted_passage_emb.view(bsz, -1, seleted_passage_emb.size(-1)) # (B, g, H)
        #                 post_passage_emb = passages_emb.clone() # (B,K,H)
        #                 for b in range(bsz):
        #                     post_passage_emb[b, passage_ids[b, :], :] = seleted_passage_emb[b, :, :]
        #         # with OptionalNoGrad(condition=self.opt.contrastive_learning):

        #         #POST PART#
        #         post_passage_emb = post_passage_emb.to(post_query_emb.device)
        #         post_logits = torch.einsum("id, ijd->ij", [post_query_emb, post_passage_emb]) # (B,100)
        #         _, post_top_retrieved_doc_ids = post_logits.topk(topk)#(B,topk)
        #         #POST PART#
        #         rank = dist.get_rank()
        #         # if rank == 0:
        #         #     post_passages = [[passages[i][j] for j in post_top_retrieved_doc_ids[i]] for i in range(bsz)]
        #         #     print(post_passages)
        #         #     print(post_query)
        #         # assert 1==0
        #         # calculate prior probs over topk
        #         prior_query_emb = self.retriever(**prior_query_enc, is_passages=False)
        #         prior_passage_emb = passage_emb

        #         #PRIOR PART
        #         prior_passage_emb = prior_passage_emb.to(prior_query_emb)
        #         prior_logits = torch.einsum("id, ijd->ij", [prior_query_emb, prior_passage_emb])
        #         #prior_topk_documents_embeddings = torch.gather(prior_passage_emb, 1, prior_top_retrieved_doc_ids.unsqueeze(-1).expand(-1, -1, passage_emb.size(-1)))
        #         _, proir_top_retrieved_doc_ids = prior_logits.topk(topk)#(B,K)
        #         if self.opt.unil_postandprior:
        #             all_doc_ids = torch.cat((post_top_retrieved_doc_ids,proir_top_retrieved_doc_ids), dim=1)
        #             all_t = []
        #             for i in range(all_doc_ids.size(0)):
        #                 unique_elements = []
        #                 seen = set()
        #                 #print(f"NOW: {i}")
        #                 for j in range(all_doc_ids.size(1)):
        #                     if all_doc_ids[i, j].item() not in seen or unique_elements.count(all_doc_ids[i, j]) == 0:
        #                         #print(tensor[i, j].item())
        #                         seen.add(all_doc_ids[i, j].item())
        #                         unique_elements.append(int(all_doc_ids[i, j].item()))
        #                         #print(f"unique_elements: {unique_elements}")
        #                 all_t.append(unique_elements)
    
        #             unique_retrieved_doc_ids = torch.tensor(all_t)
        #             unique_retrieved_doc_ids = unique_retrieved_doc_ids.to(prior_logits.device).to(torch.int64)
        #             union_topk_documents_text = [[passages[i][j] for j in unique_retrieved_doc_ids[i]] for i in range(bsz)]
        #             prior_topk_score = torch.gather(prior_logits, 1, unique_retrieved_doc_ids)# (B, K)
        #             prior_probs = F.softmax(prior_topk_score/self.opt.temperature_jsa, dim=-1) # (B, K),p(h|x),
        #             #print(f"prior_probs:{prior_probs}")
        #             #assert 1==0
        #             # posterior_topk_documents_text = [[passages[i][j] for j in post_top_retrieved_doc_ids[i]] for i in range(bsz)]
        #             # posterior_topk_documents_embeddings = torch.gather(post_passage_emb, 1, post_top_retrieved_doc_ids.unsqueeze(-1).expand(-1, -1, passage_emb.size(-1)))
        #             #post_topk_score = torch.gather(post_logits, 1, post_top_retrieved_doc_ids)
        #             post_topk_score = torch.gather(post_logits, 1, unique_retrieved_doc_ids)
        #             post_probs = F.softmax(post_topk_score/self.opt.temperature_jsa, dim=-1) # (B,topk),q(h|x,y)
        #             #caculate grad
        #             if not self.opt.query_side_retriever_training:
        #                 union_passage_token_topk = self.get_passages_tokens(union_topk_documents_text)
        #                 union_passage_emb_wg_topk = self.retriever(**union_passage_token_topk, is_passages=True)
        #                 union_passage_emb_wg_topk = union_passage_emb_wg_topk.view(bsz, -1, union_passage_emb_wg_topk.size(-1))
        #                 #print(priorerior_topk_documents_embeddings,"\n",prior_passage_emb_wg_topk)
        #                 prior_topk_score1 = torch.einsum("id, ijd->ij", [prior_query_emb, union_passage_emb_wg_topk])
        #                 prior_probs4train = F.softmax(prior_topk_score1/self.opt.temperature_jsa, dim=-1) # (B, K),p(h|x),
        #                 # post_passage_token_topk = self.get_passages_tokens(union_topk_documents_text)
        #                 # post_passage_emb_wg_topk = self.post_retriever(**post_passage_token_topk, is_passages=True)
        #                 # post_passage_emb_wg_topk = post_passage_emb_wg_topk.view(bsz, -1, post_passage_emb_wg_topk.size(-1))
        #                 post_topk_score1 = torch.einsum("id, ijd->ij", [post_query_emb, union_passage_emb_wg_topk])
        #                 post_probs4train = F.softmax(post_topk_score1/self.opt.temperature_jsa, dim=-1)

        #             else:
        #                 post_probs4train = post_probs
        #                 prior_probs4train = prior_probs

        #         else:
        #             posterior_topk_documents_text = [[passages[i][j] for j in post_top_retrieved_doc_ids[i]] for i in range(bsz)]
        #             posterior_topk_documents_embeddings = torch.gather(post_passage_emb, 1, post_top_retrieved_doc_ids.unsqueeze(-1).expand(-1, -1, passage_emb.size(-1)))
        #             post_topk_score = torch.gather(post_logits, 1, post_top_retrieved_doc_ids)
        #             #post_topk_score = torch.gather(post_logits, 1, unique_retrieved_doc_ids)
        #             post_probs = F.softmax(post_topk_score/self.opt.temperature_jsa, dim=-1) # (B,topk),q(h|x,y)
        #             prior_logits = torch.einsum("id, ijd->ij", [prior_query_emb, posterior_topk_documents_embeddings])
        #             prior_probs = F.softmax(prior_logits/self.opt.temperature_jsa, dim=-1) # (B, K),p(h|x),
        #             #caculate grad
        #             if not self.opt.query_side_retriever_training:
        #                 post_passage_token_topk = self.get_passages_tokens(posterior_topk_documents_text)
        #                 post_passage_emb_wg_topk = self.post_retriever(**post_passage_token_topk, is_passages=True)
        #                 post_passage_emb_wg_topk = post_passage_emb_wg_topk.view(bsz, -1, post_passage_emb_wg_topk.size(-1))

        #                 post_topk_score1 = torch.einsum("id, ijd->ij", [post_query_emb, post_passage_emb_wg_topk])
        #                 post_probs4train = F.softmax(post_topk_score1/self.opt.temperature_jsa, dim=-1)

        #                 prior_topk_score1 = torch.einsum("id, ijd->ij", [prior_query_emb, post_passage_emb_wg_topk])
        #                 prior_probs4train = F.softmax(prior_topk_score1/self.opt.temperature_jsa, dim=-1) # (B, K),p(h|x),

        #             else:
        #                 post_probs4train = post_probs
        #                 prior_probs4train = prior_probs
                    
        #             # print(f"post_top_retrieved_doc_ids:{post_top_retrieved_doc_ids}")
        #             # print(f"proir_top_retrieved_doc_ids:{proir_top_retrieved_doc_ids}")
        #             # #print(f"unique_retrieved_doc_ids:{unique_retrieved_doc_ids}")
        #             # print(f"prior_probs:{prior_probs}")
        #             # print(f"post_probs:{post_probs}")
        #             # print(f"post_probs4train:{post_probs4train}")
        #             # print(f"prior_probs4train:{prior_probs4train}")
        #             # assert 1==0
        #         #PRIOR PART

        #         training_info['Prior_probs'] = ','.join(['{:.3f}'.format(p) for p in prior_probs[0, :].cpu().tolist()])
        #         training_info['Post_probs'] = ','.join(['{:.3f}'.format(p) for p in post_probs[0, :].cpu().tolist()])
        #         # MIS sampling
        #         accept_rate = 0
        #         total_turns = 0
        #         total_proposed_ids = torch.multinomial(post_probs.float(), self.opt.mis_step, replacement=True) # (B, n), ids从0开始
        #         training_info['Proposal_indices'] = ','.join(str(item) for item in total_proposed_ids[0,:].cpu().tolist())
        #         #print(f"total_proposed_ids:{total_proposed_ids}")
        #         self.generator.eval()
        #         with torch.no_grad():
        #             # calculate lm probs on the proposed passages
        #             if self.opt.unil_postandprior:
        #                 total_proposed_passages = self.SelectDoc(union_topk_documents_text, total_proposed_ids)
        #             else:
        #                 total_proposed_passages = self.SelectDoc(posterior_topk_documents_text, total_proposed_ids) # a list of texts of [B, n]

        #             if self.opt.decoder_only:
        #                 generator_ids, repeated_labels, generator_mask, _ = self.tokenize_casual(query, total_proposed_passages, target) # (B*n, L)


                    
        #             generator_output = self.generator(
        #                 input_ids=generator_ids.cuda(),
        #                 attention_mask=generator_mask.cuda(),
        #                 labels=repeated_labels,
        #                 use_cache=False
        #             )
        #             seq_logits = generator_output.logits
        #             # log_lm_probs = F.log_softmax(generator_output.logits, dim=-1) # (B*n, L_out, V)
        #             # labels4gather = repeated_labels.masked_fill(repeated_labels==-100, 0)
        #             # log_lm_probs = torch.gather(log_lm_probs, -1, labels4gather.unsqueeze(-1)).squeeze(-1) # (B*n, L_out)
        #             # log_lm_probs = (log_lm_probs*(repeated_labels!=-100)).sum(-1).view(bsz, -1) # (B, n)
        #             log_lm_probs = self.get_llm_score(seq_logits, repeated_labels, self.opt.mis_step)
        #             training_info['Proposed_log_lm_probs'] = ','.join(['{:.3f}'.format(p) for p in log_lm_probs[0,:].cpu().tolist()])

        #             # cfg.n_context = n_context_training
        #             # cfg.bsz = bsz
        #             total_sampled_ids = []
        #             random_numbers = []
        #             for i in range(self.opt.mis_step):
        #                 proposed_idx = total_proposed_ids[:, i].unsqueeze(1) # (B, 1)
        #                 post_prob = torch.gather(post_probs, 1, proposed_idx).squeeze(1) #(B,)
        #                 prior_prob = torch.gather(prior_probs, 1, proposed_idx).squeeze(1)
        #                 log_lm_prob = log_lm_probs[:, i]

        #                 if i==0:
        #                     sampled_idx = proposed_idx.clone()
        #                 else:
        #                     sampled_idx = pv_sampled_idx.clone()
        #                     lm_prob_ratio = torch.exp((log_lm_prob-pv_log_lm_prob)/self.opt.temperature_lm)
        #                     accept_probs = lm_prob_ratio*prior_prob*pv_post_prob/(pv_prior_prob*post_prob+self.eps)
        #                     for j in range(bsz):
        #                         rand_num = random.random()
        #                         if j==0:
        #                             random_numbers.append(rand_num)
        #                         total_turns += 1
        #                         if rand_num<=accept_probs[j]:
        #                             accept_rate += 1
        #                             sampled_idx[j,:] = proposed_idx[j,:]#(1,1)
        #                         else:
        #                             post_prob[j] = pv_post_prob[j]
        #                             prior_prob[j] = pv_prior_prob[j]
        #                             log_lm_prob[j] = pv_log_lm_prob[j]
        #                 total_sampled_ids.append(sampled_idx)
        #                 pv_post_prob = post_prob # previous posterior prob
        #                 pv_prior_prob = prior_prob
        #                 pv_log_lm_prob = log_lm_prob
        #                 pv_sampled_idx = sampled_idx
        #             training_info['Random_numbers'] = ','.join(['{:.3f}'.format(p) for p in random_numbers])
        #         # calculate probs with gradient
        #         self.generator.train()
        #         total_sampled_ids = torch.cat(total_sampled_ids, dim=1) # (B, n)
        #         #print(f"total_sampled_ids:{total_sampled_ids}")
        #         # id_counts = []
        #         # for i in range(bsz):
        #         #     si = total_sampled_ids[i, :] # (n,)
        #         #     id_count = torch.bincount(si, minlength=self.opt.retriever_n_context) # (K, )
        #         #     id_counts.append(id_count)
        #         # id_counts = torch.stack(id_counts, dim=0) # (B,K)
        #         # training_info['id_count'] = ','.join([str(item) for item in id_counts[0,:].cpu().tolist()])
        #         # id_weights = id_counts/self.opt.mis_step # (B,K)
        #         if self.opt.use_all_mis:
        #             #unique_ids = torch.unique(total_sampled_ids)
        #             counts = []
        #             unique_ids = []
        #             probabilities = []
        #             for i in range(bsz):
        #                 si = total_sampled_ids[i, :] # (n,)
        #                 id_count = torch.bincount(si) # (K, )
        #                 unique_id = torch.unique(si)
        #                 total_count = si.numel()
        #                 probabilitie = id_count.float() / total_count
        #                 uniid_count = id_count[unique_id]
        #                 probabilitie = probabilitie[unique_id]
        #                 if self.opt.mis_topk != 0 and probabilitie.size(0)>self.opt.mis_topk:
        #                     probabilitie, topk_indices = torch.topk(probabilitie, k=self.opt.mis_topk, dim=0)
        #                     unique_id = unique_id.gather(0, topk_indices)
        #                 # print(si)
        #                 # print(probabilitie)
        #                 # print(id_count)
        #                 # print(unique_id)
        #                 # assert 1==0
        #                 probabilities.append(probabilitie)
        #                 counts.append(uniid_count)
        #                 unique_ids.append(unique_id)
        #                 #topk_probs, topk_indices = torch.topk(probabilities, k=self.opt.mis_topk, dim=0)
        #             #counts = torch.stack(counts, dim=0)
        #             probabilities = torch.stack(probabilities, dim=0)#(B,N)
        #             sampled_ids = torch.stack(unique_ids, dim=0)
        #             unicounts = torch.stack(counts, dim=0)
        #             # print(counts)
        #             # print(probabilities)
        #             # print(sampled_ids)
        #             # assert 1==0
        #         else:
        #             sampled_ids = total_sampled_ids[:, -topk:]#选择后几个
        #         # sampled_ids = torch.cat(total_sampled_ids[-self.opt.training_sample_num:], dim=1) # (B, n)
        #         # print(f"post_top_retrieved_doc_ids:{post_top_retrieved_doc_ids}")
                
        #         #print(f"sampled_ids:{sampled_ids}")
        #         # assert 1==0
        #         training_info['Accept_rate'] = accept_rate
        #         iter_stats["accept_rate"] = (accept_rate/total_turns, len(query))
        #         #i = iter_stats["accept_rate"]
        #         #print(f"iter_stats[acceptrate]:{i}")

        #         if self.opt.unil_postandprior:
        #             sampled_passages = self.SelectDoc(union_topk_documents_text, sampled_ids)
        #         else:
        #             sampled_passages = self.SelectDoc(posterior_topk_documents_text, sampled_ids) # a list of texts of [B, n]
        #         if self.opt.mis_step<100:
        #             training_info['MIS_sampled_indices'] = sampled_ids[0,:].cpu().tolist()
        #             training_info['MIS_sampled_passages'] = [p['text'] for p in sampled_passages[0]]
        #         if self.opt.contrastive_learning:
        #             sampled_passages_text = [[fstr.format(**p) for p in example] for example in sampled_passages]
        #             sampled_retriever_tokens = encode_passages(
        #                 sampled_passages_text, self.retriever_passage_tokenizer, min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH))
        #             sampled_retriever_tokens = _to_cuda(sampled_retriever_tokens) # (B, n, L)
        #             # retrieve negative samples
        #             neg_ids = torch.randint(0, self.opt.retriever_n_context, (bsz, self.opt.training_sample_num))
        #             neg_passages = self.SelectDoc(passages, neg_ids)
        #             neg_passages_text = [[fstr.format(**p) for p in example] for example in neg_passages]
        #             neg_retriever_tokens = encode_passages(
        #                 neg_passages_text, self.retriever_passage_tokenizer, min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH))
        #             neg_retriever_tokens = _to_cuda(neg_retriever_tokens) # (B, n, L)
        #             for k, v in sampled_retriever_tokens.items():
        #                 sampled_retriever_tokens[k] = torch.cat([v, neg_retriever_tokens[k]], dim=1) # (B, 2n, L)
        #             sampled_retriever_tokens = {k: v.reshape(-1, v.size(-1)) for k, v in sampled_retriever_tokens.items()} # (B*2n, L)
        #             # calculate posterior prob
        #             post_passage_emb = self.post_retriever(**sampled_retriever_tokens, is_passages=True).to(post_query_emb.device) # (B*2n, H)
        #             post_passage_emb = post_passage_emb.view(bsz, -1, post_passage_emb.size(-1)) # (B,2n,H)
        #             log_post_prob = F.log_softmax(torch.einsum("id, ijd->ij", [post_query_emb, post_passage_emb]), dim=-1) # (B, 2n)
        #             log_post_prob = log_post_prob[:, :self.opt.training_sample_num].mean(-1) # (B, n) --> (B,)
        #             # calculate prior prob
        #             prior_passage_emb = self.retriever(**sampled_retriever_tokens, is_passages=True).to(prior_query_emb.device) # (B*n, H)
        #             prior_passage_emb = prior_passage_emb.view(bsz, -1, prior_passage_emb.size(-1)) # (B,n,H)
        #             log_prior_prob = F.log_softmax(torch.einsum("id, ijd->ij", [prior_query_emb, prior_passage_emb]), dim=-1) # (B,n)
        #             log_prior_prob = log_prior_prob[:, :self.opt.training_sample_num].mean(-1)
        #         elif self.opt.reduce_norm:
        #             # positive scores
        #             sampled_post_embed = self.process_passages(sampled_passages, self.post_retriever)
        #             sampled_post_scores = torch.einsum("id, ijd->ij", [post_query_emb.detach(), sampled_post_embed]) # (B, n)
        #             sampled_prior_embed = self.process_passages(sampled_passages, self.retriever)
        #             sampled_prior_scores = torch.einsum("id, ijd->ij", [prior_query_emb.detach(), sampled_prior_embed])
        #             # negative scores
        #             neg_post_ids = torch.multinomial(post_probs, self.opt.training_sample_num, replacement=True) # (B,n)
        #             neg_post_passages = self.SelectDoc(passages, neg_post_ids)
        #             training_info['Post_sampled_ids'] = neg_post_ids[0,:].cpu().tolist()
        #             training_info['Post_sampled_passages'] = neg_post_passages[0]
        #             neg_post_embed = self.process_passages(neg_post_passages, self.post_retriever)
        #             neg_post_scores = torch.einsum("id, ijd->ij", [post_query_emb.detach(), neg_post_embed])
        #             neg_prior_ids = torch.multinomial(prior_probs, self.opt.training_sample_num, replacement=True)
        #             neg_prior_passages = self.SelectDoc(passages, neg_prior_ids)
        #             training_info['Prior_sampled_ids'] = neg_prior_ids[0,:].cpu().tolist()
        #             training_info['Prior_sampled_passages'] = neg_prior_passages[0]
        #             neg_prior_embed = self.process_passages(neg_prior_passages, self.retriever)
        #             neg_prior_scores = torch.einsum("id, ijd->ij", [prior_query_emb.detach(), neg_prior_embed])
        #             # calculate loss
        #             prior_obj = (sampled_prior_scores.mean(-1) - neg_prior_scores.mean(-1)) # (B, )
        #             post_obj = (sampled_post_scores.mean(-1) - neg_post_scores.mean(-1)) # (B,)
        #             # add log prob loss
        #             log_prior_prob = torch.log(torch.gather(prior_probs, 1, sampled_ids) + self.eps).mean(-1) # (B,)
        #             log_post_prob = torch.log(torch.gather(post_probs, 1, sampled_ids) + self.eps).mean(-1)
        #             log_prior_prob += prior_obj
        #             log_post_prob += post_obj
        #         else:
        #             # if self.opt.mis_step>=100:
        #             #     # weight the probs
        #             #     log_prior_prob = (torch.log(prior_probs+self.eps)*id_weights)
        #             #     log_post_prob = (torch.log(post_probs+self.eps)*id_weights)
        #             # else:
        #             log_prior_prob = torch.log(torch.gather(prior_probs4train, 1, sampled_ids) + self.eps) # (B,K)
        #             log_post_prob = torch.log(torch.gather(post_probs4train, 1, sampled_ids) + self.eps)
        #             # print(log_post_prob.requires_grad)
        #             # assert 1==0
        #         if self.opt.use_gradient_checkpoint_retriever and "dpr" not in self.opt.retriever_model_path :
        #             self.retriever.gradient_checkpointing_disable()
        #             self.post_retriever.gradient_checkpointing_disable()
        #         if self.opt.decoder_only:
        #             passages_to_encode =  sampled_passages
        #             # for i in passages_to_encode:
        #             #     print(i)
        #             # assert 1==0
        #             generator_ids, generator_labels, generator_mask, _ = self.tokenize_casual(query, passages_to_encode, target) # (B*K, L)
        #         else:
        #             generator_tokens, _ = self.tokenize_passages(query, sampled_passages) # (B, n, L)
        #         if self.opt.use_gradient_checkpoint_generator:
        #             self.generator.gradient_checkpointing_enable()
        #         if self.opt.decoder_only:
        #             generator_output = self.generator(
        #                 input_ids=generator_ids,
        #                 attention_mask=generator_mask,
        #                 labels=generator_labels,
        #                 use_cache=False
        #             )
        #             seq_logits = generator_output.logits # (B*K, L, V)
        #             if self.opt.use_all_mis:
        #                 _,topk = probabilities.shape
        #             batch_size_topk, sequence_length, _ = seq_logits.shape
        #             seq_logits = seq_logits / self.opt.temperature_gold
        #             shift_logits = seq_logits[..., :-1, :].contiguous()       
        #             shift_labels = generator_labels[..., 1:].contiguous()
        #             loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        #             loss1 = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        #             loss1 = loss1.reshape(batch_size_topk, -1) 
        #             #shift_labels = shift_labels.reshape(batch_size_topk, -1)
        #             #print(shift_labels.shape)
        #             loss1 = loss1.sum(dim=-1) / (shift_labels != -100).sum(dim=-1)
        #             loss1 = loss1.reshape(-1, topk)
        #             if self.opt.use_all_mis:
        #                 probabilities = F.softmax(torch.log(probabilities)/self.opt.temperature_score, dim=1) + self.eps
        #                 decoder_loss = (probabilities*loss1).sum(dim=-1)
        #             else:
        #                 decoder_loss = loss1.sum(dim=-1)/loss1.size(1)#(B,)
        #             #print(loss1)
        #             #decoder_loss = loss1.sum(dim=-1)#/loss1.size(1)#(B,)
        #             #print(decoder_loss)
        #             #assert 1==0
        #             loss = decoder_loss
        #             loss = loss.mean()
        #             # print(decoder_loss,loss,loss1)
        #             # assert 1==0
        #             # log_prior_prob = log_prior_prob.mean(-1)
        #             # log_post_prob = log_post_prob.mean(-1)

        #         else:
        #             if self.opt.fid_training:
        #                 cfg.bsz = bsz
        #                 cfg.n_context = self.opt.training_sample_num if self.opt.mis_step<100 else self.opt.retriever_n_context
        #                 generator_ids = generator_tokens["input_ids"].view(cfg.bsz, -1) # (B, n*L_in)
        #                 generator_mask = generator_tokens["attention_mask"].bool().view(cfg.bsz, -1) # (B, n*L_in)
        #                 generator_output = self.generator(
        #                     input_ids=generator_ids.cuda(),
        #                     attention_mask=generator_mask.cuda(),
        #                     decoder_input_ids=decoder_input_ids,
        #                     labels=labels,
        #                     use_cache=False,
        #                 )
        #                 generator_loss = generator_output[0]
        #             else:
        #                 cfg.n_context = 1
        #                 cfg.bsz = bsz * self.opt.training_sample_num
        #                 generator_ids = generator_tokens["input_ids"].view(cfg.bsz, -1) # (B*n, L_in)
        #                 generator_mask = generator_tokens["attention_mask"].bool().view(cfg.bsz, -1) # (B*n, L_in)
        #                 repeated_labels = torch.repeat_interleave(labels, self.opt.training_sample_num, dim=0) # (B*n, L_in)
        #                 generator_output = self.generator(
        #                     input_ids=generator_ids.cuda(),
        #                     attention_mask=generator_mask.cuda(),
        #                     labels=repeated_labels,
        #                     use_cache=False
        #                 )
        #                 cfg.n_context = n_context_training
        #                 cfg.bsz = bsz
        #                 log_lm_prob = F.log_softmax(generator_output.logits, dim=-1) # (B*n, L_out, V)
        #                 labels4gather = repeated_labels.masked_fill(repeated_labels==-100, 0) # (B*n, L_out)
        #                 log_lm_prob = torch.gather(log_lm_prob, -1, labels4gather.unsqueeze(-1)).squeeze(-1) # (B*n, L_out)
        #                 generator_loss = -(log_lm_prob*(repeated_labels!=-100)).sum(-1) # (B*n,)
        #                 generator_loss = generator_loss.view(bsz, -1).mean(-1).mean()
        #         type = 1
        #         if type ==1:
        #             generator_loss = (probabilities*(loss1-log_prior_prob - log_post_prob)).sum(dim=-1)#(B,)
        #         elif type ==2:
        #             generator_loss = ((probabilities*loss1).sum(dim=-1)-log_prior_prob.mean(-1) - log_post_prob.mean(-1))
        #         else:
        #         # generator_loss = generator_loss.mean()
        #             ll = (unicounts*(-loss1+log_prior_prob + log_post_prob))#(B,)
        #             ll = ll.logsumexp(dim = -1)
        #             nll_loss = - ll
        #             generator_loss = nll_loss.mean()
        #         retriever_loss = None
        #         if self.opt.use_gradient_checkpoint_generator:
        #             self.generator.gradient_checkpointing_disable()
        if self.opt.gen_method == "concat":
            KL = None
            retriever_loss = None
            assert self.opt.decoder_only, "This mode is only for decoder model"
            #assert self.opt.train_retriever, "The retriever must be trained in RAG method"
            prior_query_enc = self.retriever_tokenize(query_to_retrieve)
            #prior_query_enc, labels, decoder_input_ids = self.tokenize(query_to_retrieve, target, target_tokens)
            #decoder_input_ids, labels, decoder_mask, retriever_tokens = self.tokenize_casual(query, passages, target)
            if not self.opt.use_file_passages:
                #print(post_passages[0])
                passages, scores, _, passages_emb  = self.retrieve(
                    index,
                    topk,
                    query_to_retrieve,
                    prior_query_enc["input_ids"],
                    prior_query_enc["attention_mask"],
                    batch_metadata=batch_metadata,
                    filtering_fun=filtering_fun,
                    iter_stats=iter_stats,
                )
                #prior_question_embeddings = self.retriever(**prior_query_enc, is_passages=False)
            # if self.opt.use_passage_refresh:
            #     passages = [[passages[i][j] for j in top_retrieved_doc_ids[i]] for i in range(bsz)]
            # else:
            #     passgaes = passages1
            # if rank ==0: 
                
            #     print(f'origin-passages:{passages1}')
            #     print(f'index-passgaes:{passages}')
            #passages = [p[:self.opt.retriever_n_context] for p in passages]

            assert 'extra_id' not in self.opt.qa_prompt_format, "The sentinel token should not appear in the QA prompt of decoder only model"
            generator_ids, generator_labels, generator_mask, retriever_tokens = self.tokenize_casual(query, passages, target) # (B*K, L)
            cfg = None
            retriever_loss = None


            #doc_score计算
            # query_emb = self.retriever(**query_enc, is_passages=False)
            # retriever_tokens = {k: v.reshape(-1, v.size(-1)) for k, v in retriever_tokens.items()}
            # passage_emb = self.retriever(**retriever_tokens, is_passages=True).to(query_emb)
            # passage_emb = passage_emb.view(bsz, -1, passage_emb.size(-1))
            # retriever_score = torch.einsum("id, ijd->ij", [query_emb, passage_emb]) # (B,K)
            # p_z_given_x = F.softmax(retriever_score, dim=-1) # (B, K)
            if self.training:
                self.generator.train()
                
            # TODO: add rag training
            # calculate the generator loss

            if self.opt.use_gradient_checkpoint_generator:
                self.generator.gradient_checkpointing_enable()
            generator_output = self.generator(
                input_ids=generator_ids,
                attention_mask=generator_mask,
                labels=generator_labels,
                use_cache=False
            )
            if self.opt.use_gradient_checkpoint_generator:
                self.generator.gradient_checkpointing_disable()
            #rag_sequence的loss计算

            seq_logits = generator_output.logits # (B, L, V)

            seq_logits = seq_logits / self.opt.temperature_gold
            shift_logits = seq_logits[..., :-1, :].contiguous()        
            shift_labels = generator_labels[..., 1:].contiguous()
            # if rank == 0:
            #     for a, b in zip(shift_logits, shift_labels):
            #         print(f"!!!TEXT!!!:\n{a}\n")
            #         print(f"!!!LABEL!!!:\n{b}\n")
            #         print("-" * 40)
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.reshape(bsz, -1) # batch_size_topk x seq_length
            shift_labels = shift_labels.reshape(bsz, -1)
            loss = loss.sum(dim=-1) / (shift_labels != -100).sum(dim=-1)
            loss = loss.mean()
            #loss = loss.sum(dim=-1) 
            generator_loss = loss.mean()    
        elif self.opt.gold_score_mode=="rag":
            assert self.opt.decoder_only, "This mode is only for decoder model"
            rank = dist.get_rank()
            # (B*K, L)
            cfg = None
            retriever_loss = None
            KL = None
            #assert self.opt.train_retriever, "The retriever must be trained in RAG method"
            query_enc, decoder_input_ids = self.retriever_tokenize(query), None
            if not self.opt.use_file_passages:
                retrieve_start = time.time()
                passages, _, _, _ = self.retrieve(
                    index,
                    topk,
                    query_to_retrieve,
                    query_enc["input_ids"],
                    query_enc["attention_mask"],
                    batch_metadata=batch_metadata,
                    filtering_fun=filtering_fun,
                    iter_stats=iter_stats,
                )#self.opt.retriever_n_context
                iter_stats["runtime/retrieve"] = (time.time() - retrieve_start, 1)
                training_info['Retrieved_top20_passages'] = [p['text'] for p in passages[0][:20]]
                prior_question_embeddings = self.retriever(**query_enc, is_passages=False)
                prior_topk_documents_token = self.get_passages_tokens(passages)
                prior_topk_documents_embeddings = self.retriever(**prior_topk_documents_token, is_passages=True,grad_no_pass=True)
                prior_topk_documents_embeddings = prior_topk_documents_embeddings.view(bsz, -1, prior_topk_documents_embeddings.size(-1))
                prior_topk_score = torch.einsum("id, ijd->ij", [prior_question_embeddings, prior_topk_documents_embeddings])
                prior_probs = F.softmax(prior_topk_score, dim=-1)
                if not self.opt.query_side_retriever_training:
                    prior_passage_emb_wg_topk = self.retriever(**prior_topk_documents_token, is_passages=True)
                    prior_passage_emb_wg_topk = prior_passage_emb_wg_topk.view(bsz, -1, prior_passage_emb_wg_topk.size(-1))
                    #print(posterior_topk_documents_embeddings,"\n",post_passage_emb_wg_topk)
                    #post_topk_score = torch.einsum("id, ijd->ij", [posterior_question_embeddings, prior_topk_documents_embeddings])
                    prior_topk_score = torch.einsum("id, ijd->ij", [prior_question_embeddings, prior_passage_emb_wg_topk])
                    prior_probs4train = F.softmax(prior_topk_score, dim=-1)
                    #post_probs4train = F.softmax(post_topk_score/self.opt.temperature_jsa, dim=-1)
                else:    
                    prior_probs4train = prior_probs
                prior_topk_documents_text = passages
            else:
                passages = [p[:self.opt.retriever_n_context] for p in passages]
                fstr = self.opt.retriever_format
                ret_passages = [[fstr.format(**p) for p in example] for example in passages]
                retriever_doc = encode_passages(
                    ret_passages,
                    self.retriever_passage_tokenizer,
                    min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH),
                )
                retriever_doc = _to_cuda(retriever_doc)
                retriever_docs = {k: v.reshape(-1, v.size(-1)) for k, v in retriever_doc.items()}
                with torch.no_grad(): 
                    passage_emb = self.retriever(**retriever_docs, is_passages=True,grad_no_pass=True)
                passage_emb = passage_emb.view(bsz, -1, passage_emb.size(-1))
                    # if self.opt.use_passage_refresh:
                    #     passages = [[passages[i][j] for j in top_retrieved_doc_ids[i]] for i in range(bsz)]
                    # else:
                    #     passgaes = passages1
                    # if rank ==0: 
                        
                    #     print(f'origin-passages:{passages1}')
                    #     print(f'index-passgaes:{passages}')
                    #passages = [p[:self.opt.retriever_n_context] for p in passages]


                if self.opt.use_gradient_checkpoint_retriever and "dpr" not in self.opt.retriever_model_path :
                    self.retriever.gradient_checkpointing_enable()

                #doc_score计算
                query_emb = self.retriever(**query_enc, is_passages=False)
                passage_emb = passage_emb.to(query_emb)
                retriever_score = torch.einsum("id, ijd->ij", [query_emb, passage_emb]) # (B,K)
                _, top_retrieved_doc_ids = retriever_score.topk(topk)
                prior_topk_documents_text = [[passages[i][j] for j in top_retrieved_doc_ids[i]] for i in range(bsz)]
                prior_topk_documents_embeddings = torch.gather(passage_emb, 1, top_retrieved_doc_ids.unsqueeze(-1).expand(-1, -1, passage_emb.size(-1)))
                prior_topk_score = torch.gather(retriever_score, 1, top_retrieved_doc_ids)
                p_z_given_x = F.softmax(retriever_score, dim=-1) # (B, K)

                if not self.opt.query_side_retriever_training:
                    passage_token_topk = self.get_passages_tokens(prior_topk_documents_text)
                    passage_emb_wg_topk = self.retriever(**passage_token_topk, is_passages=True)
                    passage_emb_wg_topk = passage_emb_wg_topk.view(bsz, -1, passage_emb_wg_topk.size(-1))
                    #print(posterior_topk_documents_embeddings,"\n",post_passage_emb_wg_topk)
                    prior_probs4train = torch.einsum("id, ijd->ij", [query_emb, passage_emb_wg_topk])
                #assert 1==0
                else:
                    prior_probs4train = prior_topk_score

            if self.training:
                self.generator.train()
            generator_ids, generator_labels, generator_mask, _ = self.tokenize_casual(query, prior_topk_documents_text, target)
            # TODO: add rag training
            # calculate the generator loss

            if self.opt.use_gradient_checkpoint_generator:
                self.generator.gradient_checkpointing_enable()
            generator_output = self.generator(
                input_ids=generator_ids,
                attention_mask=generator_mask,
                labels=generator_labels,
                use_cache=False
            )
            if rank ==0 and self.test:
                with torch.no_grad():
                    rank = dist.get_rank()
                    generate_test_inputs, attention_test_mask = self.tokenize_casual4gen(query, passages)
                    
                    print(f"before_gen_inputids:{self.generator_tokenizer.decode(generate_test_inputs[0], skip_special_tokens=True)}")
                    input_len = generate_test_inputs.size(1)
                    outputs = self.generator.generate(
                        input_ids = generate_test_inputs,
                        attention_mask = attention_test_mask,
                        max_new_tokens=50,
                        return_dict_in_generate=True,
                        output_scores=True,
                        forced_bos_token_id=None,
                        length_penalty=self.opt.generation_length_penalty,
                        do_sample=False,
                        pad_token_id=self.generator_tokenizer.eos_token_id
                    )
                    generation = outputs.sequences
                    generation1 = generation[:, input_len:]


                    print(f"all:{self.generator_tokenizer.decode(generation[0], skip_special_tokens=True)}")
                    print(f"gen_answer:{self.generator_tokenizer.decode(generation1[0], skip_special_tokens=True)}")

            if self.opt.use_gradient_checkpoint_generator:
                self.generator.gradient_checkpointing_disable()
            #rag_sequence的loss计算
            doc_scores = prior_probs4train


            decoded_texts = [self.generator_tokenizer.decode(ids, skip_special_tokens=False) for ids in generator_ids]
            test_label = torch.where(generator_labels != -100,generator_labels,self.generator_tokenizer.pad_token_id)
            decoded_labels = [self.generator_tokenizer.decode(lab, skip_special_tokens=False) for lab in test_label]
            if rank == 0 and self.test:
                for text_ids, label_ids in zip(generator_ids, generator_labels):
                    print(f"!!!TEXT!!!:\n{text_ids}\n")
                    print(f"!!!LABEL!!!:\n{label_ids}\n")
                    print("-" * 40)
                assert 1==0
            if rank == 0 and self.test:
                for text, label in zip(decoded_texts, decoded_labels):
                    print(f"!!!TEXT!!!:\n{text}\n")
                    print(f"!!!LABEL!!!:\n{label}\n")
                    print("-" * 40)


            seq_logits = generator_output.logits # (B*K, L, V)
            batch_size_topk, sequence_length, _ = seq_logits.shape

            seq_logits = seq_logits / self.opt.temperature_gold
            shift_logits = seq_logits[..., :-1, :].contiguous()        
            shift_labels = generator_labels[..., 1:].contiguous()
            # if rank == 0:
            #     for a, b in zip(shift_logits, shift_labels):
            #         print(f"!!!TEXT!!!:\n{a}\n")
            #         print(f"!!!LABEL!!!:\n{b}\n")
            #         print("-" * 40)
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.reshape(batch_size_topk, -1) # batch_size_topk x seq_length
            if rank==0 and self.test:
                print(loss[0])
            # shift_labels = shift_labels.reshape(batch_size_topk, -1)
            shift_labels = shift_labels.reshape(batch_size_topk, -1)
            loss = loss.sum(dim=-1) / (shift_labels != -100).sum(dim=-1) 
            loss = loss.reshape(-1, topk) # batch_size x topk
            if 1==1:
                p_y_given_zx = torch.exp(-loss) 
                p_z_given_x = F.softmax(doc_scores, dim=-1) 
                p_y_given_x = (p_z_given_x * p_y_given_zx).sum(dim=-1) + self.eps # (B, )
                loss = -torch.log(p_y_given_x).mean()
                generator_loss = -torch.log(p_y_given_x).mean()

            else:
                seg_logprobs = -loss
                doc_logprobs = nn.functional.log_softmax(doc_scores / self.opt.temperature_score, dim=1) # batch_size * topk
                ll = seg_logprobs + doc_logprobs
                if rank==0 and self.test:
                    print(ll)
                ll = ll.logsumexp(dim = 1)  # logsumexp over docs
                nll_loss = -ll
                loss = nll_loss.mean()
                generator_loss = nll_loss.mean()
            #retriever_loss = 0 # retriever loss is contained in generator_loss




            # generator_logits = generator_output.logits # (B*K, L, V)
            # generator_probs = F.softmax(generator_logits, dim=-1)
            # labels4gather = generator_labels.masked_fill(generator_labels==-100, 0)
            # generator_probs = torch.gather(generator_logits, dim=-1, index=labels4gather[..., None]).squeeze(-1) # (B*K, L)
            # generator_probs = (generator_probs*(generator_labels!=-100)).sum(-1) # (B*K),sequence层面的
            # p_y_given_zx = generator_probs.view(bsz, -1) # (B, K)
            # p_y_given_x = (p_z_given_x * p_y_given_zx).sum(dim=-1) + self.eps#(B,)
            # generator_loss = -torch.log(p_y_given_x).mean()
            # retriever_loss = 0 # retriever loss is contained in generator_loss       
        elif self.opt.gold_score_mode=="vrag": 
            #print("VRAG-STRAT")
            rank = dist.get_rank()
            prior_query_enc = self.retriever_tokenize(query_to_retrieve)
            post_query = [q+' [SEP] '+t for q, t in zip(query_to_retrieve, target)]
            post_query_enc = self.retriever_tokenize(post_query)
            if not self.opt.use_file_passages:
                post_passages, post_scores, _, post_passages_emb  = self.retrieve(
                    index,
                    topk,
                    post_query,
                    post_query_enc["input_ids"],
                    post_query_enc["attention_mask"],
                    batch_metadata=batch_metadata,
                    filtering_fun=filtering_fun,
                    iter_stats=iter_stats,
                    posterior=True
                )
                prior_passages, prior_scores, _, prior_passages_emb  = self.retrieve(
                    index,
                    topk,
                    query_to_retrieve,
                    prior_query_enc["input_ids"],
                    prior_query_enc["attention_mask"],
                    batch_metadata=batch_metadata,
                    filtering_fun=filtering_fun,
                    iter_stats=iter_stats,
                )

                priorids = [[d["id"] for d in batch] for batch in prior_passages]
                prior_top_retrieved_doc_ids =  torch.tensor(priorids, dtype=torch.int64)
                postids = [[d["id"] for d in batch] for batch in post_passages]
                post_top_retrieved_doc_ids =  torch.tensor(postids, dtype=torch.int64)
                #print(post_top_retrieved_doc_ids,prior_top_retrieved_doc_ids)
                posterior_question_embeddings = self.post_retriever(**post_query_enc, is_passages=False)
                prior_question_embeddings = self.retriever(**prior_query_enc, is_passages=False)
                if not self.opt.query_side_retriever_training:
                    post_passage_token_topk = self.get_passages_tokens(post_passages)
                    post_passage_emb_wg_topk = self.post_retriever(**post_passage_token_topk, is_passages=True)
                    posterior_topk_documents_embeddings = post_passage_emb_wg_topk.view(bsz, -1, post_passage_emb_wg_topk.size(-1))
                    #print(posterior_topk_documents_embeddings,"\n",post_passage_emb_wg_topk)
                    post_topk_score = torch.einsum("id, ijd->ij", [posterior_question_embeddings, posterior_topk_documents_embeddings])

                    prior_passage_token_topk = self.get_passages_tokens(prior_passages)
                    prior_passage_emb_wg_topk = self.retriever(**prior_passage_token_topk, is_passages=True)
                    prior_topk_documents_embeddings = prior_passage_emb_wg_topk.view(bsz, -1, prior_passage_emb_wg_topk.size(-1))
                    #print(posterior_topk_documents_embeddings,"\n",post_passage_emb_wg_topk)
                    prior_topk_score = torch.einsum("id, ijd->ij", [prior_question_embeddings, prior_topk_documents_embeddings])
                else:    
                    post_passage_token_topk = self.get_passages_tokens(post_passages)
                    post_passage_emb_wg_topk = self.post_retriever(**post_passage_token_topk, is_passages=True,grad_no_pass=True)
                    posterior_topk_documents_embeddings = post_passage_emb_wg_topk.view(bsz, -1, post_passage_emb_wg_topk.size(-1))
                    posterior_topk_documents_embeddings = posterior_topk_documents_embeddings.to(posterior_question_embeddings)
                    prior_passage_token_topk = self.get_passages_tokens(prior_passages)
                    prior_passage_emb_wg_topk = self.retriever(**prior_passage_token_topk, is_passages=True,grad_no_pass=True)
                    prior_topk_documents_embeddings = prior_passage_emb_wg_topk.view(bsz, -1, prior_passage_emb_wg_topk.size(-1))
                    prior_topk_documents_embeddings = prior_topk_documents_embeddings.to(prior_question_embeddings)
                    post_topk_score = torch.einsum("id, ijd->ij", [posterior_question_embeddings, posterior_topk_documents_embeddings])
                    prior_topk_score = torch.einsum("id, ijd->ij", [prior_question_embeddings, prior_topk_documents_embeddings])
                posterior_topk_documents_text = post_passages
                prior_topk_documents_text = prior_passages
                #passages_emb = passages_emb.to(torch.float32)
            else:
                passages = [p[:self.opt.retriever_n_context] for p in passages]
                if self.opt.use_gradient_checkpoint_retriever and "dpr" not in self.opt.retriever_model_path :
                    self.retriever.gradient_checkpointing_enable()
                    self.post_retriever.gradient_checkpointing_enable()
                fstr = self.opt.retriever_format
                ret_passages = [[fstr.format(**p) for p in example] for example in passages]
                retriever_doc = encode_passages(
                    ret_passages,
                    self.retriever_passage_tokenizer,
                    min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH),
                )
                retriever_doc = _to_cuda(retriever_doc)
                retriever_docs = {k: v.reshape(-1, v.size(-1)) for k, v in retriever_doc.items()}
                with torch.no_grad(): 
                    passage_emb = self.post_retriever(**retriever_docs, is_passages=True,grad_no_pass=True)
                passage_emb = passage_emb.view(bsz, -1, passage_emb.size(-1))
                posterior_question_embeddings = self.post_retriever(**post_query_enc, is_passages=False)
                #passage_emb = passage_emb.view(bsz, -1, passage_emb.size(-1))#(B,100,V)V是emb维度
                passage_emb = passage_emb.to(posterior_question_embeddings)
                post_retriever_score = torch.einsum("id, ijd->ij", [posterior_question_embeddings, passage_emb]) # (B,100)
                #print(post_retriever_score.shape)
                _, post_top_retrieved_doc_ids = post_retriever_score.topk(topk)#(B,K)
                posterior_topk_documents_text = [[passages[i][j] for j in post_top_retrieved_doc_ids[i]] for i in range(bsz)]
                posterior_topk_documents_embeddings = torch.gather(passage_emb, 1, post_top_retrieved_doc_ids.unsqueeze(-1).expand(-1, -1, passage_emb.size(-1)))
            
                if not self.opt.query_side_retriever_training:
                    post_passage_token_topk = self.get_passages_tokens(posterior_topk_documents_text)
                    post_passage_emb_wg_topk = self.post_retriever(**post_passage_token_topk, is_passages=True)
                    post_passage_emb_wg_topk = post_passage_emb_wg_topk.view(bsz, -1, post_passage_emb_wg_topk.size(-1))
                    #print(posterior_topk_documents_embeddings,"\n",post_passage_emb_wg_topk)
                    post_topk_score = torch.einsum("id, ijd->ij", [posterior_question_embeddings, post_passage_emb_wg_topk])
                #assert 1==0
                else:
                    post_topk_score = torch.gather(post_retriever_score, 1, post_top_retrieved_doc_ids)
                #post_probs = F.softmax(post_topk_score/self.opt.temperature_jsa, dim=-1)
                #torch.cuda.empty_cache()
                #prior_emb embedding part

                prior_question_embeddings = self.retriever(**prior_query_enc, is_passages=False)
                if self.opt.union_kl:
                    passage_emb = passage_emb.to(prior_question_embeddings)
                    prior_retriever_score = torch.einsum("id, ijd->ij", [prior_question_embeddings, passage_emb]) # (B,100)
                    _, prior_top_retrieved_doc_ids = prior_retriever_score.topk(topk)#(B,K)
                    prior_topk_documents_text = [[passages[i][j] for j in prior_top_retrieved_doc_ids[i]] for i in range(bsz)]
                    prior_topk_documents_embeddings = torch.gather(passage_emb, 1, prior_top_retrieved_doc_ids.unsqueeze(-1).expand(-1, -1, passage_emb.size(-1)))
                if not self.opt.query_side_retriever_training:
                    prior_passage_token_topk = self.get_passages_tokens(prior_topk_documents_text)
                    prior_passage_emb_wg_topk = self.retriever(**prior_passage_token_topk, is_passages=True)
                    prior_passage_emb_wg_topk = prior_passage_emb_wg_topk.view(bsz, -1, prior_passage_emb_wg_topk.size(-1))
                    #print(posterior_topk_documents_embeddings,"\n",post_passage_emb_wg_topk)
                    prior_topk_score = torch.einsum("id, ijd->ij", [prior_question_embeddings, prior_passage_emb_wg_topk])
                else:
                    prior_topk_score = torch.gather(prior_retriever_score, 1, prior_top_retrieved_doc_ids)
            if self.opt.use_gradient_checkpoint_retriever and "dpr" not in self.opt.retriever_model_path :
                self.retriever.gradient_checkpointing_disable()
                self.post_retriever.gradient_checkpointing_disable()

            if self.training:
                self.generator.train()
                

            generator_ids, generator_labels, generator_mask, _= self.tokenize_casual(query_to_retrieve, posterior_topk_documents_text, target)
            # if rank ==0:
            #     label_ids = generator_labels[0].masked_fill_(generator_labels[0] == IGNORE_INDEX, self.generator_tokenizer.eos_token_id)
            #     print(f"before_gen_inputids:{generator_ids[0]}")
            #     print(f"before_generator_labelids:{self.generator_tokenizer.decode(label_ids, skip_special_tokens=False)}")
            #     print(f"before_gen_input:{self.generator_tokenizer.decode(generator_ids[0], skip_special_tokens=False)}")
            #     print(f"before_gen_label:{generator_labels[0]}")
            #     print(f"before_gen_generator_mask:{generator_mask[0]}")

            if self.opt.use_gradient_checkpoint_generator:
                self.generator.gradient_checkpointing_enable()
            generator_output = self.generator(
                input_ids=generator_ids,
                attention_mask=generator_mask,
                labels=generator_labels,
                use_cache=False
            )
            # x = self.decoder_model._prepare_inputs(
            #     decoder_input_ids, decoder_response_ids, posterior_topk_documents_text)

            # decoder_input_ids_, decoder_response_ids_, _ = x

            # decoder_loss, _ = self.decoder_model(
            #     [decoder_input_ids_.cuda(), decoder_response_ids_.cuda()]) # (B, K)
            
            posterior_dist = F.softmax(post_topk_score/self.opt.temperature_score, dim=1) + self.eps # (B, K)
            # print(posterior_dist)
            # assert 1==0
            seq_logits = generator_output.logits # (B*K, L, V)
            # if rank ==0:
            #     print(seq_logits.shape)
            batch_size_topk, sequence_length, _ = seq_logits.shape
            seq_logits = seq_logits / self.opt.temperature_gold
            shift_logits = seq_logits[..., :-1, :].contiguous()       
            shift_labels = generator_labels[..., 1:].contiguous()
            # if rank == 0:
            #     for a, b in zip(shift_logits, shift_labels):
            #         print(f"!!!TEXT!!!:\n{a}\n")
            #         print(f"!!!LABEL!!!:\n{b}\n")
            #         print("-" * 40)
            # assert 1==0

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            loss1 = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss1 = loss1.reshape(batch_size_topk, -1)
            shift_labels = shift_labels.reshape(batch_size_topk, -1)
            loss1 = loss1.sum(dim=-1) / (shift_labels != -100).sum(dim=-1)
            decoder_loss = loss1.reshape(-1, topk)
            #loss = decoder_loss.sum(dim=-1)/decoder_loss.size(1) # (B,)
            if self.opt.standard_mc:
                loss = decoder_loss.sum(dim=-1)/decoder_loss.size(1) # (B,)
            else:
                loss = (posterior_dist * decoder_loss).sum(dim=-1)
            # print(f"posterior_dist:{posterior_dist}")
            # print(f"decoder_loss:{decoder_loss}")
            # assert 1==0

            loss = loss.mean()

            if self.opt.use_gradient_checkpoint_generator:
                self.generator.gradient_checkpointing_disable()
            #log_info['loss_lm'] = loss.item()
            if self.opt.union_kl:
                prior_model_outputs = {
                    "topk_documents_ids": prior_top_retrieved_doc_ids,
                    "question_embeddings": prior_question_embeddings,
                    "topk_documents_embeddings": prior_topk_documents_embeddings
                }
                posterior_model_outputs = {
                    "topk_documents_ids": post_top_retrieved_doc_ids,
                    "question_embeddings": posterior_question_embeddings,
                    "topk_documents_embeddings": posterior_topk_documents_embeddings
                }

                KL = GetUnionKL(prior_model_outputs, posterior_model_outputs)
                #print(KL)
            else:
                # posterior_topk_documents_embeddings = posterior_topk_documents_embeddings.float()
                # prior_question_embeddings = prior_question_embeddings.float()
                # print(f"这个prior_question_embeddings张量{prior_question_embeddings.requires_grad}计算梯度。")
                # print(f"这个posterior_topk_documents_embeddings张量{posterior_topk_documents_embeddings.requires_grad}计算梯度。")
                # assert 1==0
                if not self.opt.query_side_retriever_training:
                    #post_passage_emb_wg_topk
                    log_prior_prob_on_topk = F.log_softmax(
                        torch.einsum("id, ijd->ij", [prior_question_embeddings, post_passage_emb_wg_topk]),dim = 1).float()# (B,K)
                else:
                    log_prior_prob_on_topk = F.log_softmax(
                        torch.einsum("id, ijd->ij", [prior_question_embeddings, posterior_topk_documents_embeddings]),dim = 1).float()# (B,K)
                posterior_dist = posterior_dist.float()
                KL = F.kl_div(log_prior_prob_on_topk, posterior_dist, reduction='batchmean')

            generator_loss = loss + self.kl_beta * KL
            # print(generator_loss)
            # assert 1==0
            retriever_loss = None
            #log_info['loss_kl'] = KL.item()
        elif self.opt.gold_score_mode=="jsa":
            KL = None
            if self.opt.simplify_JSA:
                query_enc, labels, decoder_input_ids = self.tokenize(query_to_retrieve, target, target_tokens)#query_enc得到retriever的enc，其他事gen的ids
                
            else:
                post_query = [q+' [SEP] '+t for q, t in zip(query_to_retrieve, target)]
                post_query_enc = self.retriever_tokenize(post_query)
                prior_query_enc = self.retriever_tokenize(query_to_retrieve)
                #prior_query_enc, labels, decoder_input_ids = self.tokenize(query_to_retrieve, target, target_tokens)
                #decoder_input_ids, labels, decoder_mask, retriever_tokens = self.tokenize_casual(query, passages, target)
                if self.opt.use_gradient_checkpoint_retriever and "dpr" not in self.opt.retriever_model_path :
                    self.post_retriever.gradient_checkpointing_enable()
                    self.retriever.gradient_checkpointing_enable()
                if not self.opt.use_file_passages:
                    post_passages, post_scores, _, post_passages_emb  = self.retrieve(
                        index,
                        topk,
                        post_query,
                        post_query_enc["input_ids"],
                        post_query_enc["attention_mask"],
                        batch_metadata=batch_metadata,
                        filtering_fun=filtering_fun,
                        iter_stats=iter_stats,
                        posterior=True
                    )
                    #print(post_passages[0])
                    prior_passages, prior_scores, _, prior_passages_emb  = self.retrieve(
                        index,
                        topk,
                        query_to_retrieve,
                        prior_query_enc["input_ids"],
                        prior_query_enc["attention_mask"],
                        batch_metadata=batch_metadata,
                        filtering_fun=filtering_fun,
                        iter_stats=iter_stats,
                    )

                    priorids = [[d["id"] for d in batch] for batch in prior_passages]
                    prior_top_retrieved_doc_ids =  torch.tensor(priorids, dtype=torch.int64)
                    postids = [[d["id"] for d in batch] for batch in post_passages]
                    post_top_retrieved_doc_ids =  torch.tensor(postids, dtype=torch.int64)
                    #print(post_top_retrieved_doc_ids,prior_top_retrieved_doc_ids)
                    posterior_question_embeddings = self.post_retriever(**post_query_enc, is_passages=False)
                    prior_question_embeddings = self.retriever(**prior_query_enc, is_passages=False)
                    if self.opt.unil_postandprior:
                        all_doc_ids = torch.cat((post_top_retrieved_doc_ids,prior_top_retrieved_doc_ids), dim=1)
                        #print(all_doc_ids)
                        all_t = []
                        for i in range(all_doc_ids.size(0)):
                            unique_elements = []
                            seen = set()
                            #print(f"NOW: {i}")
                            for j in range(all_doc_ids.size(1)):
                                if all_doc_ids[i, j].item() not in seen or unique_elements.count(all_doc_ids[i, j]) == 0:
                                    #print(tensor[i, j].item())
                                    seen.add(all_doc_ids[i, j].item())
                                    unique_elements.append(int(all_doc_ids[i, j].item()))
                                    #print(f"unique_elements: {unique_elements}")
                            all_t.append(unique_elements)
                        #print(all_t)
                        unique_retrieved_doc_ids = torch.tensor(all_t)
                        #print(unique_retrieved_doc_ids)
                        unique_retrieved_doc_ids = unique_retrieved_doc_ids.to(prior_question_embeddings.device)#.to(torch.int64)
                        #print(unique_retrieved_doc_ids)
                        union_topk_documents_text = union_of_passages(prior_passages,post_passages,unique_retrieved_doc_ids)
                        #print(union_topk_documents_text[0])
                        union_passage_token_topk = self.get_passages_tokens(union_topk_documents_text)
                        union_passage_emb_wg_topk = self.post_retriever(**union_passage_token_topk, is_passages=True,grad_no_pass=True)
                        union_topk_documents_embeddings = union_passage_emb_wg_topk.view(bsz, -1, union_passage_emb_wg_topk.size(-1))
                        #print(posterior_topk_documents_embeddings,"\n",post_passage_emb_wg_topk)
                        post_topk_score = torch.einsum("id, ijd->ij", [posterior_question_embeddings, union_topk_documents_embeddings])
                        prior_topk_score = torch.einsum("id, ijd->ij", [prior_question_embeddings, union_topk_documents_embeddings])
                        prior_probs = F.softmax(prior_topk_score/self.opt.temperature_jsa, dim=-1)
                        post_probs = F.softmax(post_topk_score/self.opt.temperature_jsa, dim=-1)
                    if not self.opt.query_side_retriever_training:
                        union_passage_token_topk = self.get_passages_tokens(union_topk_documents_text)
                        union_passage_emb_wg_topk = self.post_retriever(**union_passage_token_topk, is_passages=True)
                        union_topk_documents_embeddings = union_passage_emb_wg_topk.view(bsz, -1, union_passage_emb_wg_topk.size(-1))
                        #print(posterior_topk_documents_embeddings,"\n",post_passage_emb_wg_topk)
                        post_topk_score = torch.einsum("id, ijd->ij", [posterior_question_embeddings, union_topk_documents_embeddings])
                        prior_topk_score = torch.einsum("id, ijd->ij", [prior_question_embeddings, union_topk_documents_embeddings])
                        prior_probs4train = F.softmax(prior_topk_score/self.opt.temperature_jsa, dim=-1)
                        post_probs4train = F.softmax(post_topk_score/self.opt.temperature_jsa, dim=-1)
                    else:    
                        prior_probs4train = prior_probs
                        post_probs4train = post_probs
                else:
                    passages = [p[:self.opt.retriever_n_context] for p in passages]




                training_info['Prior_probs'] = ','.join(['{:.3f}'.format(p) for p in prior_probs[0, :].cpu().tolist()])
                training_info['Post_probs'] = ','.join(['{:.3f}'.format(p) for p in post_probs[0, :].cpu().tolist()])
                # MIS sampling
                accept_rate = 0
                total_turns = 0
                total_proposed_ids = torch.multinomial(post_probs.float(), self.opt.mis_step, replacement=True) # (B, n), ids从0开始
                training_info['Proposal_indices'] = ','.join(str(item) for item in total_proposed_ids[0,:].cpu().tolist())
                #print(f"total_proposed_ids:{total_proposed_ids}")
                self.generator.eval()
                with torch.no_grad():
                    # calculate lm probs on the proposed passages
                    if self.opt.unil_postandprior:
                        total_proposed_passages = self.SelectDoc(union_topk_documents_text, total_proposed_ids)
                    else:
                        total_proposed_passages = self.SelectDoc(posterior_topk_documents_text, total_proposed_ids) # a list of texts of [B, n]

                    if self.opt.decoder_only:
                        generator_ids, repeated_labels, generator_mask, _ = self.tokenize_casual(query, total_proposed_passages, target) # (B*n, L)


                    
                    generator_output = self.generator(
                        input_ids=generator_ids.cuda(),
                        attention_mask=generator_mask.cuda(),
                        labels=repeated_labels,
                        use_cache=False
                    )
                    #original_dtype = generator_output.logits.dtype
                    seq_logits = generator_output.logits #/ self.opt.temperature_gold
                    # log_lm_probs = F.log_softmax(generator_output.logits, dim=-1) # (B*n, L_out, V)
                    # labels4gather = repeated_labels.masked_fill(repeated_labels==-100, 0)
                    # log_lm_probs = torch.gather(log_lm_probs, -1, labels4gather.unsqueeze(-1)).squeeze(-1) # (B*n, L_out)
                    # log_lm_probs = (log_lm_probs*(repeated_labels!=-100)).sum(-1).view(bsz, -1) # (B, n)
                    log_lm_probs = self.get_llm_score(seq_logits, repeated_labels, self.opt.mis_step)
                    training_info['Proposed_log_lm_probs'] = ','.join(['{:.3f}'.format(p) for p in log_lm_probs[0,:].cpu().tolist()])

                    # cfg.n_context = n_context_training
                    # cfg.bsz = bsz
                    total_sampled_ids = []
                    random_numbers = []
                    for i in range(self.opt.mis_step):
                        proposed_idx = total_proposed_ids[:, i].unsqueeze(1) # (B, 1)
                        post_prob = torch.gather(post_probs, 1, proposed_idx).squeeze(1) #(B,)
                        prior_prob = torch.gather(prior_probs, 1, proposed_idx).squeeze(1)
                        log_lm_prob = log_lm_probs[:, i]

                        if i==0:
                            sampled_idx = proposed_idx.clone()
                        else:
                            sampled_idx = pv_sampled_idx.clone()
                            lm_prob_ratio = torch.exp((log_lm_prob-pv_log_lm_prob)/self.opt.temperature_lm)
                            accept_probs = lm_prob_ratio*prior_prob*pv_post_prob/(pv_prior_prob*post_prob+self.eps)
                            for j in range(bsz):
                                rand_num = random.random()
                                if j==0:
                                    random_numbers.append(rand_num)
                                total_turns += 1
                                if rand_num<=accept_probs[j]:
                                    accept_rate += 1
                                    sampled_idx[j,:] = proposed_idx[j,:]#(1,1)
                                else:
                                    post_prob[j] = pv_post_prob[j]
                                    prior_prob[j] = pv_prior_prob[j]
                                    log_lm_prob[j] = pv_log_lm_prob[j]
                        total_sampled_ids.append(sampled_idx)
                        pv_post_prob = post_prob # previous posterior prob
                        pv_prior_prob = prior_prob
                        pv_log_lm_prob = log_lm_prob
                        pv_sampled_idx = sampled_idx
                    training_info['Random_numbers'] = ','.join(['{:.3f}'.format(p) for p in random_numbers])
                # calculate probs with gradient
                self.generator.train()
                total_sampled_ids = torch.cat(total_sampled_ids, dim=1) # (B, n)
                #print(f"total_sampled_ids:{total_sampled_ids}")
                # id_counts = []
                # for i in range(bsz):
                #     si = total_sampled_ids[i, :] # (n,)
                #     id_count = torch.bincount(si, minlength=self.opt.retriever_n_context) # (K, )
                #     id_counts.append(id_count)
                # id_counts = torch.stack(id_counts, dim=0) # (B,K)
                # training_info['id_count'] = ','.join([str(item) for item in id_counts[0,:].cpu().tolist()])
                # id_weights = id_counts/self.opt.mis_step # (B,K)
                if self.opt.use_all_mis:
                    #unique_ids = torch.unique(total_sampled_ids)
                    counts = []
                    unique_ids = []
                    probabilities = []
                    for i in range(bsz):
                        si = total_sampled_ids[i, :] # (n,)
                        id_count = torch.bincount(si) # (K, )
                        unique_id = torch.unique(si)
                        total_count = si.numel()
                        probabilitie = id_count.float() / total_count
                        uniid_count = id_count[unique_id]
                        probabilitie = probabilitie[unique_id]
                        if self.opt.mis_topk != 0 and probabilitie.size(0)>self.opt.mis_topk:
                            probabilitie, topk_indices = torch.topk(probabilitie, k=self.opt.mis_topk, dim=0)
                            unique_id = unique_id.gather(0, topk_indices)
                        # print(si)
                        # print(probabilitie)
                        # print(id_count)
                        # print(unique_id)
                        # assert 1==0
                        probabilities.append(probabilitie)
                        counts.append(uniid_count)
                        unique_ids.append(unique_id)
                        #topk_probs, topk_indices = torch.topk(probabilities, k=self.opt.mis_topk, dim=0)
                    #counts = torch.stack(counts, dim=0)
                    probabilities = torch.stack(probabilities, dim=0)#(B,N)
                    sampled_ids = torch.stack(unique_ids, dim=0)
                    unicounts = torch.stack(counts, dim=0)
                    # print(counts)
                    # print(probabilities)
                    # print(sampled_ids)
                    # assert 1==0
                else:
                    sampled_ids = total_sampled_ids[:, -topk:]#选择后几个
                # sampled_ids = torch.cat(total_sampled_ids[-self.opt.training_sample_num:], dim=1) # (B, n)
                # print(f"post_top_retrieved_doc_ids:{post_top_retrieved_doc_ids}")
                
                #print(f"sampled_ids:{sampled_ids}")
                # assert 1==0
                training_info['Accept_rate'] = accept_rate
                iter_stats["accept_rate"] = (accept_rate/total_turns, len(query))
                #i = iter_stats["accept_rate"]
                #print(f"iter_stats[acceptrate]:{i}")

                if self.opt.unil_postandprior:
                    sampled_passages = self.SelectDoc(union_topk_documents_text, sampled_ids)
                else:
                    sampled_passages = self.SelectDoc(posterior_topk_documents_text, sampled_ids) # a list of texts of [B, n]
                if self.opt.mis_step<100:
                    training_info['MIS_sampled_indices'] = sampled_ids[0,:].cpu().tolist()
                    training_info['MIS_sampled_passages'] = [p['text'] for p in sampled_passages[0]]
                if self.opt.contrastive_learning:
                    sampled_passages_text = [[fstr.format(**p) for p in example] for example in sampled_passages]
                    sampled_retriever_tokens = encode_passages(
                        sampled_passages_text, self.retriever_passage_tokenizer, min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH))
                    sampled_retriever_tokens = _to_cuda(sampled_retriever_tokens) # (B, n, L)
                    # retrieve negative samples
                    neg_ids = torch.randint(0, self.opt.retriever_n_context, (bsz, self.opt.training_sample_num))
                    neg_passages = self.SelectDoc(passages, neg_ids)
                    neg_passages_text = [[fstr.format(**p) for p in example] for example in neg_passages]
                    neg_retriever_tokens = encode_passages(
                        neg_passages_text, self.retriever_passage_tokenizer, min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH))
                    neg_retriever_tokens = _to_cuda(neg_retriever_tokens) # (B, n, L)
                    for k, v in sampled_retriever_tokens.items():
                        sampled_retriever_tokens[k] = torch.cat([v, neg_retriever_tokens[k]], dim=1) # (B, 2n, L)
                    sampled_retriever_tokens = {k: v.reshape(-1, v.size(-1)) for k, v in sampled_retriever_tokens.items()} # (B*2n, L)
                    # calculate posterior prob
                    post_passage_emb = self.post_retriever(**sampled_retriever_tokens, is_passages=True).to(post_query_emb) # (B*2n, H)
                    post_passage_emb = post_passage_emb.view(bsz, -1, post_passage_emb.size(-1)) # (B,2n,H)
                    log_post_prob = F.log_softmax(torch.einsum("id, ijd->ij", [post_query_emb, post_passage_emb]), dim=-1) # (B, 2n)
                    log_post_prob = log_post_prob[:, :self.opt.training_sample_num].mean(-1) # (B, n) --> (B,)
                    # calculate prior prob
                    prior_passage_emb = self.retriever(**sampled_retriever_tokens, is_passages=True).to(prior_query_emb) # (B*n, H)
                    prior_passage_emb = prior_passage_emb.view(bsz, -1, prior_passage_emb.size(-1)) # (B,n,H)
                    log_prior_prob = F.log_softmax(torch.einsum("id, ijd->ij", [prior_query_emb, prior_passage_emb]), dim=-1) # (B,n)
                    log_prior_prob = log_prior_prob[:, :self.opt.training_sample_num].mean(-1)
                elif self.opt.reduce_norm:
                    # positive scores
                    sampled_post_embed = self.process_passages(sampled_passages, self.post_retriever)
                    sampled_post_scores = torch.einsum("id, ijd->ij", [post_query_emb.detach(), sampled_post_embed]) # (B, n)
                    sampled_prior_embed = self.process_passages(sampled_passages, self.retriever)
                    sampled_prior_scores = torch.einsum("id, ijd->ij", [prior_query_emb.detach(), sampled_prior_embed])
                    # negative scores
                    neg_post_ids = torch.multinomial(post_probs, self.opt.training_sample_num, replacement=True) # (B,n)
                    neg_post_passages = self.SelectDoc(passages, neg_post_ids)
                    training_info['Post_sampled_ids'] = neg_post_ids[0,:].cpu().tolist()
                    training_info['Post_sampled_passages'] = neg_post_passages[0]
                    neg_post_embed = self.process_passages(neg_post_passages, self.post_retriever)
                    neg_post_scores = torch.einsum("id, ijd->ij", [post_query_emb.detach(), neg_post_embed])
                    neg_prior_ids = torch.multinomial(prior_probs, self.opt.training_sample_num, replacement=True)
                    neg_prior_passages = self.SelectDoc(passages, neg_prior_ids)
                    training_info['Prior_sampled_ids'] = neg_prior_ids[0,:].cpu().tolist()
                    training_info['Prior_sampled_passages'] = neg_prior_passages[0]
                    neg_prior_embed = self.process_passages(neg_prior_passages, self.retriever)
                    neg_prior_scores = torch.einsum("id, ijd->ij", [prior_query_emb.detach(), neg_prior_embed])
                    # calculate loss
                    prior_obj = (sampled_prior_scores.mean(-1) - neg_prior_scores.mean(-1)) # (B, )
                    post_obj = (sampled_post_scores.mean(-1) - neg_post_scores.mean(-1)) # (B,)
                    # add log prob loss
                    log_prior_prob = torch.log(torch.gather(prior_probs, 1, sampled_ids) + self.eps).mean(-1) # (B,)
                    log_post_prob = torch.log(torch.gather(post_probs, 1, sampled_ids) + self.eps).mean(-1)
                    log_prior_prob += prior_obj
                    log_post_prob += post_obj
                else:
                    # if self.opt.mis_step>=100:
                    #     # weight the probs
                    #     log_prior_prob = (torch.log(prior_probs+self.eps)*id_weights)
                    #     log_post_prob = (torch.log(post_probs+self.eps)*id_weights)
                    # else:
                    log_prior_prob = torch.log(torch.gather(prior_probs4train, 1, sampled_ids) + self.eps) # (B,K)
                    log_post_prob = torch.log(torch.gather(post_probs4train, 1, sampled_ids) + self.eps)
                    # print(log_post_prob.requires_grad)
                    # assert 1==0

                if self.opt.decoder_only:
                    passages_to_encode =  sampled_passages
                    # for i in passages_to_encode:
                    #     print(i)
                    # assert 1==0
                    generator_ids, generator_labels, generator_mask, _ = self.tokenize_casual(query, passages_to_encode, target) # (B*K, L)
                else:
                    generator_tokens, _ = self.tokenize_passages(query, sampled_passages) # (B, n, L)
                if self.opt.use_gradient_checkpoint_generator:
                    self.generator.gradient_checkpointing_enable()
                if self.opt.decoder_only:
                    generator_output = self.generator(
                        input_ids=generator_ids,
                        attention_mask=generator_mask,
                        labels=generator_labels,
                        use_cache=False
                    )
                    seq_logits = generator_output.logits # (B*K, L, V)
                    if self.opt.use_all_mis:
                        _,topk = probabilities.shape
                    batch_size_topk, sequence_length, _ = seq_logits.shape
                    seq_logits = seq_logits / self.opt.temperature_gold
                    shift_logits = seq_logits[..., :-1, :].contiguous()       
                    shift_labels = generator_labels[..., 1:].contiguous()
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
                    loss1 = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss1 = loss1.reshape(batch_size_topk, -1) 
                    #shift_labels = shift_labels.reshape(batch_size_topk, -1)
                    #print(shift_labels.shape)
                    loss1 = loss1.sum(dim=-1) / (shift_labels != -100).sum(dim=-1)
                    loss1 = loss1.reshape(-1, topk)
                    if self.opt.use_all_mis:
                        # probabilities = F.softmax(torch.log(probabilities)/self.opt.temperature_score, dim=1) + self.eps
                        decoder_loss = (probabilities*loss1).sum(dim=-1)
                    else:
                        decoder_loss = loss1.sum(dim=-1)/loss1.size(1)#(B,)
                    #print(loss1)
                    #decoder_loss = loss1.sum(dim=-1)#/loss1.size(1)#(B,)
                    #print(decoder_loss)
                    #assert 1==0
                    loss = decoder_loss
                    loss = loss.mean()
                    # print(decoder_loss,loss,loss1)
                    # assert 1==0
                    # log_prior_prob = log_prior_prob.mean(-1)
                    # log_post_prob = log_post_prob.mean(-1)

                else:
                    if self.opt.fid_training:
                        cfg.bsz = bsz
                        cfg.n_context = self.opt.training_sample_num if self.opt.mis_step<100 else self.opt.retriever_n_context
                        generator_ids = generator_tokens["input_ids"].view(cfg.bsz, -1) # (B, n*L_in)
                        generator_mask = generator_tokens["attention_mask"].bool().view(cfg.bsz, -1) # (B, n*L_in)
                        generator_output = self.generator(
                            input_ids=generator_ids.cuda(),
                            attention_mask=generator_mask.cuda(),
                            decoder_input_ids=decoder_input_ids,
                            labels=labels,
                            use_cache=False,
                        )
                        generator_loss = generator_output[0]
                    else:
                        cfg.n_context = 1
                        cfg.bsz = bsz * self.opt.training_sample_num
                        generator_ids = generator_tokens["input_ids"].view(cfg.bsz, -1) # (B*n, L_in)
                        generator_mask = generator_tokens["attention_mask"].bool().view(cfg.bsz, -1) # (B*n, L_in)
                        repeated_labels = torch.repeat_interleave(labels, self.opt.training_sample_num, dim=0) # (B*n, L_in)
                        generator_output = self.generator(
                            input_ids=generator_ids.cuda(),
                            attention_mask=generator_mask.cuda(),
                            labels=repeated_labels,
                            use_cache=False
                        )
                        cfg.n_context = n_context_training
                        cfg.bsz = bsz
                        log_lm_prob = F.log_softmax(generator_output.logits, dim=-1) # (B*n, L_out, V)
                        labels4gather = repeated_labels.masked_fill(repeated_labels==-100, 0) # (B*n, L_out)
                        log_lm_prob = torch.gather(log_lm_prob, -1, labels4gather.unsqueeze(-1)).squeeze(-1) # (B*n, L_out)
                        generator_loss = -(log_lm_prob*(repeated_labels!=-100)).sum(-1) # (B*n,)
                        generator_loss = generator_loss.view(bsz, -1).mean(-1).mean()
                type = 1
                if type ==1:
                    generator_loss = (probabilities*(loss1-log_prior_prob - log_post_prob)).sum(dim=-1)#(B,)
                elif type ==2:
                    generator_loss = ((probabilities*loss1).sum(dim=-1)-log_prior_prob.mean(-1) - log_post_prob.mean(-1))
                else:
                # generator_loss = generator_loss.mean()
                    ll = (unicounts*(-loss1+log_prior_prob + log_post_prob))#(B,)
                    ll = ll.logsumexp(dim = -1)
                    nll_loss = - ll
                    generator_loss = nll_loss.mean()
                retriever_loss = None
                # if self.opt.use_gradient_checkpoint_generator:
                #     self.generator.gradient_checkpointing_disable()
        if not self.opt.decoder_only:
            self.generator.reset_score_storage()
        iter_stats["loss/generator_loss"] = (loss.item(), len(query))
        if retriever_loss is not None:
            iter_stats["loss/retriever_loss"] = (retriever_loss.item(), len(query))
        if KL is not None:
            iter_stats["KL"] = (KL.item(), len(query))
        iter_stats["runtime/forward"] = (time.time() - forward_start, 1)
        #print(f"IN:{iter_stats}")
        return generator_loss, retriever_loss, training_info,iter_stats

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score / self.opt.temperature_gold, dim=-1)
        score = torch.nn.functional.log_softmax(score / self.opt.temperature_score, dim=-1)
        return torch.nn.KLDivLoss()(score, gold_score)

    def logprob(self, score, gold_score, labels, training_info):
        with torch.no_grad():
            mask_labels = labels >= 0
            if self.opt.decoder_only:
                repeated_labels = labels
            else:
                repeated_labels = torch.repeat_interleave(labels, self.opt.retriever_n_context, dim=0)
            repeated_labels[repeated_labels == IGNORE_INDEX] = 0

            

            gold_log_prob = torch.nn.functional.log_softmax(gold_score / self.opt.temperature_gold, dim=-1) # (B*K, L, V)
            gold_log_probs = torch.gather(gold_log_prob, dim=-1, index=repeated_labels[..., None]).view(
                gold_log_prob.size(0), -1
            ) # (B*K, L)
            gold_sent_probs = (gold_log_probs*(repeated_labels!=IGNORE_INDEX)).sum(-1).view(score.size(0), -1) # (B, K)
            training_info['Gold_log_probs'] = ','.join(['{:.3f}'.format(p) for p in gold_sent_probs[0, :].cpu().tolist()])
            training_info['Retriever_score'] = ','.join(['{:.3f}'.format(p) for p in score[0, :].cpu().tolist()])
            gold_log_probs = gold_log_probs.view(score.size(0), score.size(1), -1) # (B, K, T)

        log_score = torch.nn.functional.log_softmax(score / self.opt.temperature_score, dim=-1) # (B, K)
        log_prob = gold_log_probs + log_score[..., None]
        logsumprobs = torch.logsumexp(log_prob, dim=1)
        loss = -1 * torch.sum(logsumprobs * mask_labels) / torch.sum(mask_labels)


        return loss

    @torch.no_grad()
    def compute_generator_loss_and_logits(self, tokens, decoder_input_ids, labels):
        cfg = self.generator.encoder.config
        cfg.bsz = tokens["input_ids"].size(0)
        cfg.n_context = min(self.opt.n_context, tokens["input_ids"].size(1))

        generator_loss = self.generator(
            input_ids=tokens["input_ids"].cuda().view(tokens["input_ids"].size(0), -1),
            attention_mask=tokens["attention_mask"].cuda().view(tokens["attention_mask"].size(0), -1),
            decoder_input_ids=decoder_input_ids.cuda(),
            labels=labels.cuda(),
            use_cache=False,
        )
        return generator_loss[0].cpu().item(), generator_loss[1]

    @torch.no_grad()
    def generate(self, tokens, query, choices=None):
        # tokens: (B,K,L)
        cfg = self.generator.encoder.config
        cfg.bsz = tokens["input_ids"].size(0)
        cfg.n_context = min(self.opt.n_context, tokens["input_ids"].size(1))

        tokens = {k: v.view(v.size(0), -1) for k, v in tokens.items()} # (B, K*L)

        bos_token_id = None

        prefix_allowed_tokens_fn = None
        if self.opt.decoder_prompt_format is not None:
            prefix_str = [self.opt.decoder_prompt_format.format_map({"query": q}) for q in query]
            prefix_allowed_tokens_fn = self.get_prefix_allowed_tokens_fn(prefix_str)

        outputs = self.generator.generate(
            input_ids=tokens["input_ids"].cuda(),
            attention_mask=tokens["attention_mask"].cuda(),
            num_return_sequences=1,
            max_length=self.opt.generation_max_length,
            min_length=self.opt.generation_min_length,
            num_beams=self.opt.generation_num_beams,
            length_penalty=self.opt.generation_length_penalty,
            forced_bos_token_id=bos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            return_dict_in_generate=True,
            output_scores=True
        )

        return outputs

    def get_prefix_allowed_tokens_fn(self, prefix_str: Optional[str] = None):
        if prefix_str:
            prefix_tokens_ids = self.generator_tokenizer.batch_encode_plus(prefix_str, add_special_tokens=False)[
                "input_ids"
            ]

            def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
                if input_ids.shape[-1] > len(prefix_tokens_ids[batch_id]):
                    return self.generator_ALL_TOKENS

                return prefix_tokens_ids[batch_id][input_ids.shape[-1] - 1]

        else:
            prefix_allowed_tokens_fn = None

        return prefix_allowed_tokens_fn
    
    @torch.no_grad()
    def method_generate(self,generation,scores,bsz,topk,ret_scores):
        rank = dist.get_rank()
        if self.opt.gen_method =="fast_deocde1":
            # use fast-decoding (rag-sequnce) to select one generation among topk
            logits = torch.stack(scores, dim=1) # (B*K, L_out, V)
            #L_out = logits.size(1)
            # greedy decoding, directly use the max prob
            target_length = torch.sum(generation != self.generator_tokenizer.pad_token_id, dim = 1)
            log_probs = F.log_softmax(logits, dim=-1).max(-1).values # (B*K, L_out)
            sent_probs = torch.exp(log_probs.sum(-1)/target_length) # (B*K, )，加入target_length的tensor
            sent_probs = sent_probs.view(bsz, topk) # (B,K)
            if not self.opt.use_file_passages:
                ret_scores = torch.tensor(ret_scores, device=sent_probs.device) # (B, K)
            ret_probs = F.softmax(ret_scores / self.opt.gen_doc_scores, dim=-1)
            total_probs = sent_probs*ret_probs # (B, K)
            max_indices = total_probs.max(-1).indices # (B,)
            generation = generation.view(bsz, topk, -1) # (B*K, L) --> (B, K, L)
            orin_gen = generation
            max_indices = max_indices.view(bsz, 1, 1).expand(bsz, 1, generation.size(-1)) # (B, 1, L)
            generation = torch.gather(generation, 1, max_indices).squeeze(1) # (B,L)
            
        if self.opt.gen_method =="fast_deocde2":
            logits = torch.stack(scores, dim=1)
            if rank ==0:
                print(f"logits.shape:{logits.shape}") # (B*K, L_out, V)
            target_length = torch.sum(generation != self.generator_tokenizer.pad_token_id, dim = 1)
            #L_out = logits.size(1)
            # greedy decoding, directly use the max prob
            log_probs = F.log_softmax(logits, dim=-1).max(-1).values # (B*K, L_out)
            log_probs = log_probs.sum(-1)/target_length # (B*K, )
            log_probs = log_probs.view(bsz, topk) # (B,K)
            # if not self.opt.use_file_passages:
            #     ret_scores = torch.tensor(ret_scores, device=sent_probs.device) # (B, K)
            log_ret_probs = ret_scores / self.opt.gen_doc_scores
            total_scores = log_probs + log_ret_probs # (B, K)
            max_indices = total_scores.max(-1).indices # (B,)
            generation = generation.view(bsz, topk, -1) # (B*K, L) --> (B, K, L)
            orin_gen = generation
            max_indices = max_indices.view(bsz, 1, 1).expand(bsz, 1, generation.size(-1)) # (B, 1, L)
            generation = torch.gather(generation, 1, max_indices).squeeze(1) # (B,L)          

        if self.opt.gen_method =="concat":
            generation = generation
            orin_gen = generation
        return generation,orin_gen    

    def get_llm_score(self, seq_logits, labels, topk = None):
        topk = topk if topk is not None else self.topk

        seq_logits = seq_logits
        batch_size_topk, sequence_length, _ = seq_logits.shape
        batch_size = batch_size_topk // topk

        shift_logits = seq_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.reshape(batch_size_topk, -1) # batch_size_topk x seq_length
        shift_labels = shift_labels.reshape(batch_size_topk, -1)
        loss = loss.sum(dim=-1) / (shift_labels != -100).sum(dim=-1)
        loss = loss.reshape(-1, topk) # batch_size x n_docs

        seg_logprobs = -loss
        return seg_logprobs
    def GetParameters(self):
        prior_params = list(self.retriever.parameters())
        posterior_params = list(self.post_retriever.parameters())
        decoder_params = list(self.generator.parameters())

        if (self.fix_DPR or (self.fix_prior and self.fix_posterior)):
            return decoder_params
        elif (self.fix_posterior and self.fix_decoder):
            return prior_params
        elif (not self.fix_prior and not self.fix_posterior and not self.fix_decoder):
            return prior_params + posterior_params + decoder_params
def select_crossattention_scores(scores, mode):
    if "eval" in mode:
        return scores[mode[len("eval") :]]
    elif "std" in mode:
        return scores[mode[len("std") :]]


def _to_cuda(tok_dict):
    return {k: v.cuda() for k, v in tok_dict.items()}

def union_of_passages(prior_passages, post_passages, unique_ids):
    # 获取唯一的 id 集合
    union_passages = []
    # 存储已经添加过的元素的 id
    added_ids = set()
    # 遍历 prior_passages
    # print(f"prior_passages:{prior_passages}")
    # print(f"post_passages:{post_passages}")
    for batch in prior_passages:
        for passage in batch:
            if passage["id"] in unique_ids and passage["id"] not in added_ids:
                union_passages.append(passage)
                added_ids.add(passage["id"])
    # 遍历 post_passages
    for batch in post_passages:
        for passage in batch:
            if passage["id"] in unique_ids and passage["id"] not in added_ids:
                union_passages.append(passage)
                added_ids.add(passage["id"])

    return [union_passages]
def GetUnionKL(prior_model_outputs, posterior_model_outputs):
    prior_topk_documents_ids = prior_model_outputs["topk_documents_ids"]
    posterior_topk_documents_ids = posterior_model_outputs["topk_documents_ids"]
    prior_question_embeddings = prior_model_outputs["question_embeddings"]
    posterior_question_embeddings = posterior_model_outputs["question_embeddings"]
    prior_topk_documents_embeddings = prior_model_outputs["topk_documents_embeddings"]
    posterior_topk_documents_embeddings = posterior_model_outputs["topk_documents_embeddings"]

    batch_size = len(prior_topk_documents_ids)
    topk = len(prior_topk_documents_ids[0])

    KL = 0
    for i in range(batch_size):
        all_docs_embeds = []
        s = set()
        for j in range(topk):
            id1, id2 = prior_topk_documents_ids[i][j], posterior_topk_documents_ids[i][j]
            if id1 not in s:
                s.add(id1)
                all_docs_embeds.append(prior_topk_documents_embeddings[i][j])
            if id2 not in s:
                s.add(id2)
                all_docs_embeds.append(posterior_topk_documents_embeddings[i][j])


        all_docs_embeds = torch.stack(all_docs_embeds).T.cuda() # (H,N)
        #print(all_docs_embeds.shape)
        
        prior_logits_full = prior_question_embeddings[i].unsqueeze(
            0) @ all_docs_embeds # (1, H)*(H,N): (1, N)
        posterior_logits_full = posterior_question_embeddings[i].unsqueeze(
            0) @ all_docs_embeds

        prior_log_dist_full = F.log_softmax(
            prior_logits_full, dim=-1).squeeze().float() # (N)
        posterior_dist_full = F.softmax(
            posterior_logits_full, dim=-1).squeeze().float()

        KL += F.kl_div(prior_log_dist_full, posterior_dist_full, reduction='sum')
    KL /= batch_size
    # print(KL)
    # assert 1==0
    return KL
def remove_speakers(text):
    import re
    # 使用正则表达式替换 <speaker1> 和 <speaker2> 及其后面的空格
    cleaned_text = re.sub(r'<speaker1>\s*', '', text)
    cleaned_text = re.sub(r'<speaker2>\s*', '', cleaned_text)
    return cleaned_text
# def call_retrieve_api(query_embs=None,topk=10):
#     # 定义请求数据
    
#     #query_embs = torch.rand(2,1024)
#     bsz = query_embs.size(0)
#     query_embs = query_embs.to(torch.float32)
#     query_emb_list = query_embs.cpu().numpy().flatten().tolist()

#     data = {
#         "query_embs": query_emb_list,
#         "bsz": bsz,
#         "topk": topk
#     }

#     # 发送请求
#     response = requests.post("http://paraai-n32-h-01-agent-2:29501/retrieve", json=data)

#     # 处理响应
#     if response.status_code == 200:
#         results = response.json()
#         #print(results[0])
        
#         return results[0],results[1]
        
#     else:
#         print(f"请求失败，状态码: {response.status_code}")