# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import errno
import logging
import os
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
import transformers

import src.fid
from src import dist_utils
from src.rag import RAG
from src.retrievers import Embedding_con, DualEncoderRetriever, UntiedDualEncoderRetriever, Embedding_Ret
from src.util import cast_to_precision, set_dropout, set_optim
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer,GPT2LMHeadModel

Number = Union[float, int]

logger = logging.getLogger(__name__)


def get_checkpoint_path(opt):
    checkpoint_path = Path(opt.checkpoint_dir) / opt.name#mdefalut:experiment
    return checkpoint_path

def get_class(model_name):
    if 'dpr' in model_name and 'question' in model_name:
        return transformers.DPRQuestionEncoder, transformers.DPRQuestionEncoderTokenizer
    elif 'dpr' in model_name and 'ctx' in model_name:
        return transformers.DPRContextEncoder, transformers.DPRContextEncoderTokenizer
    elif 'bge' in model_name or 'retriever' in model_name or 'nomic' in model_name or 'gte' in model_name:
        return transformers.AutoModel, transformers.AutoTokenizer
    else:
        print('Unseen class')

def create_checkpoint_directories(opt):
    checkpoint_path = get_checkpoint_path(opt)
    os.makedirs(checkpoint_path, exist_ok=True)
    if opt.save_index_path:
        os.makedirs(opt.save_index_path, exist_ok=True)
    dist_utils.barrier()
    return checkpoint_path, opt.save_index_path


def load_retriever(opt, opt_checkpoint=None):
    # if opt.use_file_passages:
    #     return None, None

    encoder = Embedding_Ret(opt.retriever_model_path)

    passage_encoder = Embedding_Ret(opt.passage_model_path) if opt.passage_model_path is not None else None

    # contriever_encoder = Contriever.from_pretrained(opt.retriever_model_path)
    if 'dpr' in opt.retriever_model_path:
        retriever_tokenizer = transformers.DPRQuestionEncoderTokenizer.from_pretrained(opt.retriever_model_path)
        retriever_passage_tokenizer = transformers.DPRContextEncoderTokenizer.from_pretrained(opt.passage_model_path)
    else:
        retriever_tokenizer = transformers.AutoTokenizer.from_pretrained(opt.retriever_model_path)
        retriever_passage_tokenizer = None

    if opt.dialog:
        #print("RET_SPECIAL_TOKEN")
        SPECIAL_TOKENS = {
        "additional_special_tokens": ["<speaker1>", "<speaker2>"]
        }
        retriever_tokenizer.add_special_tokens(SPECIAL_TOKENS)
        encoder.encoder.resize_token_embeddings(len(retriever_tokenizer))
        if passage_encoder is not None:
            passage_encoder.encoder.resize_token_embeddings(len(retriever_tokenizer))
        #print("SPECIALL_TOKEN",retriever_tokenizer.special_tokens_map["additional_special_tokens"])
        special_token_ids = retriever_tokenizer.all_special_ids
        #print("Special token IDs:", special_token_ids)

    # once you have done query side training you cannot go back to a parameter-tied retriever

    params_equal = True

    retriever = UntiedDualEncoderRetriever(opt, encoder, passage_encoder)

    for param in retriever.passage_retriever.parameters():
        print(param.requires_grad)
        break
    for param in retriever.query_retriever.parameters():
        print(param.requires_grad)
        break


    return retriever, retriever_tokenizer,retriever_passage_tokenizer


def _convert_state_dict_from_dual_encoder_retriever(state_dict):
    """handles when we want to load an UntiedDualEncoderRetriever from a DualEncoderRetriever state dict"""
    new_state_dict = {}
    for k, tensor in state_dict.items():
        if k.startswith("retriever"):
            new_state_dict[k.replace("retriever.retriever", "retriever.passage_retriever")] = tensor
            new_state_dict[k.replace("retriever.retriever", "retriever.query_retriever")] = tensor
        else:
            new_state_dict[k] = tensor
    return new_state_dict


def load_generator(opt):
    generator = None
    if not opt.retrieve_only:
        if 't5' in opt.generator_model_type:
            assert not opt.decoder_only, "The T5 model is not compatible with the decoder_only setting"
            generator = src.fid.FiD.from_pretrained(opt.generator_model_type)

            if opt.compute_crossattention_stats or "eval" in opt.gold_score_mode or "std" in opt.gold_score_mode:
                generator.overwrite_forward_crossattention()
                generator.create_crossattention_storage()
            generator_tokenizer = transformers.AutoTokenizer.from_pretrained(opt.generator_model_type)
        else:
            # decoder-only generator
            if "GPT" in opt.generator_model_type:
                config = AutoConfig.from_pretrained(opt.generator_model_type)
                generator = GPT2LMHeadModel.from_pretrained(
                opt.generator_model_type, config=config)
                #generator = AutoModelForCausalLM.from_pretrained(opt.generator_model_type, torch_dtype=torch.float16, low_cpu_mem_usage=True)
            else:
                generator = AutoModelForCausalLM.from_pretrained(opt.generator_model_type, torch_dtype=torch.float16, low_cpu_mem_usage=True)
            generator_tokenizer = transformers.AutoTokenizer.from_pretrained(opt.generator_model_type)
            #print(f"SPECIALL_TOKEN!!:generator_tokenizer.pad_token:{generator_tokenizer.pad_token}/n,generator_tokenizer.pad_token:{generator_tokenizer.pad_token_id}")
            if not generator_tokenizer.pad_token:
                generator_tokenizer.pad_token = generator_tokenizer.eos_token
                generator_tokenizer.pad_token_id = generator_tokenizer.eos_token_id
            if "GPT" in opt.generator_model_type:
                SPECIAL_TOKENS = {
                    "bos_token": "<bos>",
                    "eos_token": "<eos>",
                    "pad_token": "<pad>",
                }
            else:
                SPECIAL_TOKENS = {}
            if opt.dialog:
                SPECIAL_TOKENS.update({
                "additional_special_tokens": ["<speaker1>", "<speaker2>"]
                })
            generator_tokenizer.add_special_tokens(SPECIAL_TOKENS)
            generator.resize_token_embeddings(len(generator_tokenizer))
            #print("SPECIALL_TOKEN",generator_tokenizer.special_tokens_map["additional_special_tokens"])
            special_token_ids = generator_tokenizer.all_special_ids

            # print(f"PAD_TOKEN!!:generator_tokenizer.pad_token:{generator_tokenizer.pad_token}/n,generator_tokenizer.pad_token:{generator_tokenizer.pad_token_id}")
            # print(f"EOS_TOKEN!!:generator_tokenizer.eos_token:{generator_tokenizer.eos_token}/n,generator_tokenizer.eos_token:{generator_tokenizer.eos_token_id}")
            # print(f"BOS_TOKEN!!:generator_tokenizer.bos_token:{generator_tokenizer.bos_token}/n,generator_tokenizer.bos_token:{generator_tokenizer.bos_token_id}")

    if opt.use_lora:
        print("start load lora model!!")
        from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
        peft_config = LoraConfig(
            task_type = TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16.0, 
            lora_dropout=0.0,
            bias="none",
            target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
        )
        generator.enable_input_require_grads()
        generator = get_peft_model(generator, peft_config)
        print("successfully load lora model!!")
        generator.print_trainable_parameters()
    return generator, generator_tokenizer


def _set_generator_encoder_cfg(model, opt):
    if model.generator is not None and hasattr(model.generator, 'encoder'):
        cfg = model.generator.encoder.config
        cfg.n_context = opt.n_context
        cfg.bsz = opt.per_gpu_batch_size


def _cast_atlas_to_precision(atlas_model, precision):
    if atlas_model.generator is not None:
        atlas_model.generator = cast_to_precision(atlas_model.generator, precision)
    if atlas_model.retriever is not None and precision == "bf16":
        atlas_model.retriever = cast_to_precision(atlas_model.retriever, precision)
        if atlas_model.post_retriever is not None:
            atlas_model.post_retriever = cast_to_precision(atlas_model.post_retriever, precision)


def _cast_and_set_attrs_and_send_to_device(model, opt):
    _set_generator_encoder_cfg(model, opt)
    set_dropout(model, opt.dropout)
    _cast_atlas_to_precision(model, opt.precision)
    model = model.to(opt.device)
    return model


def _load_atlas_model_state(opt, opt_checkpoint, model, model_dict):
    model_dict = {
            k.replace("retriever.module", "retriever").replace("generator.module", "generator"): v for k, v in model_dict.items()
        }
    # if opt.decouple_encoder or (opt.query_side_retriever_training and not opt_checkpoint.query_side_retriever_training):
    #     model_dict = _convert_state_dict_from_dual_encoder_retriever(model_dict)
    if 'mistral' or "GPT" in opt.generator_model_type:
        if opt.retrieve_only:  # dont load generator if in retrieve only mode
            model_dict = {k: v for k, v in model_dict.items() if not k.startswith("generator")}

        # if opt.use_file_passages:  # dont load retriever if in use_file_passages mode
        #     model_dict = {k: v for k, v in model_dict.items() if not k.startswith("retriever")}
        if opt.gold_score_mode in ['vrag', 'jsa'] and not opt.simplify_JSA:
            # add posterior retriever state dict
            retriever_dict = {k: v for k, v in model_dict.items() if k.startswith("retriever")}
            post_retriever_exist = False
            for k,v in model_dict.items():
                if k.startswith('post_retriever'):
                    post_retriever_exist = True
                    break
            if not post_retriever_exist:
                for k, v in retriever_dict.items():
                    new_key = k.replace('retriever', 'post_retriever')
                    model_dict[new_key] = v
        model.load_state_dict(model_dict)
    else:
        # only load retriever
        model_dict = {k.replace('retriever.', ''): v for k, v in model_dict.items() if k.startswith("retriever")}
        model.retriever.load_state_dict(model_dict)
    #model.load_state_dict(model_dict)
    model = _cast_and_set_attrs_and_send_to_device(model, opt)
    return model

def para_num(para,name):
    total_params = 0
    for param in para:
        total_params += param.numel()  # 计算总参数量
    print(f"Total number of parameters in the {name}: {total_params}")

def load_atlas_model(dir_path, opt, reset_params=False, eval_only=False):
    epoch_path = os.path.realpath(dir_path)
    save_path = os.path.join(epoch_path, "model.pth.tar")
    logger.info(f"Loading {epoch_path}")
    logger.info(f"loading checkpoint {save_path}")
    checkpoint = torch.load(save_path, map_location="cpu")
    opt_checkpoint = checkpoint["opt"]
    step = checkpoint["step"]
    model_dict = checkpoint["model"]
    #model = torch.load(save_path)
    # print(*model_dict.keys())
    # assert 1==0
    generator, generator_tokenizer = load_generator(opt)
    #assert 1==0
    retriever, retriever_tokenizer,retriever_passage_tokenizer = load_retriever(opt, opt_checkpoint)

    model = RAG(opt, generator, retriever, generator_tokenizer, retriever_tokenizer,retriever_passage_tokenizer)

    if opt.load_pretrained_weights:
        model = _load_atlas_model_state(opt, opt_checkpoint, model, model_dict)
    else:
        model = _cast_and_set_attrs_and_send_to_device(model, opt)
    if eval_only:
        return model, None, None, None, None, opt_checkpoint, step

    if not reset_params:
        optimizer, scheduler, retr_optimizer, retr_scheduler = set_optim(opt_checkpoint, model)
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if opt.train_retriever:
            retr_scheduler.load_state_dict(checkpoint["retr_scheduler"])
            retr_optimizer.load_state_dict(checkpoint["retr_optimizer"])
    else:
        optimizer, scheduler, retr_optimizer, retr_scheduler = set_optim(opt, model)

    return model, optimizer, scheduler, retr_optimizer, retr_scheduler, opt_checkpoint, step


def init_atlas_model(opt, eval_only):
    print("START INIT")
    generator, generator_tokenizer = load_generator(opt)
    retriever, retriever_tokenizer,retriever_passage_tokenizer = load_retriever(opt)

    model = RAG(opt, generator, retriever, generator_tokenizer, retriever_tokenizer,retriever_passage_tokenizer)
    # para_num(model.post_retriever.query_retriever.parameters(),"post_retriever.query_retriever")
    # para_num(model.post_retriever.passage_retriever.parameters(),"post_retriever.passage_retriever")
    # para_num(model.retriever.query_retriever.parameters(),"retriever.query_retriever")
    # para_num(model.retriever.passage_retriever.parameters(),"retriever.passage_retriever")
    # para_num(model.parameters(),"retriever")
    # params_equal = True
    # for param1, param2 in zip(model.post_retriever.query_retriever.parameters(), model.post_retriever.passage_retriever.parameters()):
    #     if not torch.equal(param1, param2):
    #         params_equal = False
    #         break
    # if params_equal:
    #     print("模型参数相同")
    # else:
    #     print("模型参数不相同")
    model = _cast_and_set_attrs_and_send_to_device(model, opt)

    if eval_only:
        return model, None, None, None, None, opt, 0

    optimizer, scheduler, retr_optimizer, retr_scheduler = set_optim(opt, model)
    return model, optimizer, scheduler, retr_optimizer, retr_scheduler, opt, 0


def load_or_initialize_atlas_model(opt, eval_only=False):
    """
    Either initializes a Atlas from t5 and retriever or loads one from disk.

    if opt.model_path is "none" and {opt.checkpoint_dir/opt.name} doesn't exist, it will init a Atlas

    or, if opt.model_path is "none" and {opt.checkpoint_dir/opt.name} does exist, it will load the Atlas at opt.checkpoint_dir/opt.name/latest

    or, if opt.model_path is not "none" it will load the saved Atlas in opt.model_path
    """
    checkpoint_path = get_checkpoint_path(opt)
    latest_checkpoint_path = os.path.join(checkpoint_path, "checkpoint", "latest")
    #assert 1==0
    if opt.model_path == "none":
        #print("check1")

        print("check2") # Fresh run:
        return init_atlas_model(opt, eval_only)

    else:  # fresh finetune run, initialized from old model
        load_path, reset_params = opt.model_path, True
    #print("check3")
    model, optimizer, scheduler, retr_optimizer, retr_scheduler, opt_checkpoint, loaded_step = load_atlas_model(
        load_path, opt, reset_params=reset_params, eval_only=eval_only
    )
    logger.info(f"Model loaded from {load_path}")
    step = 0 if opt.model_path != "none" else loaded_step

    return model, optimizer, scheduler, retr_optimizer, retr_scheduler, opt, step

def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e
    

def save_atlas_model(model, optimizer, scheduler, retr_optimizer, retr_scheduler, step, opt, dir_path, name):

    if opt.save_optimizer and opt.shard_optim:
        optimizer.consolidate_state_dict()
        if retr_optimizer:
            retr_optimizer.consolidate_state_dict()

    if not opt.is_main:
        return 0

    model_to_save = model.module if hasattr(model, "module") else model
    path = os.path.join(dir_path, "checkpoint")
    epoch_path = os.path.join(path, name)  # "step-%s" % step)
    os.makedirs(epoch_path, exist_ok=True)
    cp = os.path.join(path, "latest")
    fp = os.path.join(epoch_path, "model.pth.tar")

    optim_state = optimizer.state_dict() if opt.save_optimizer else None
    if retr_optimizer and opt.save_optimizer:
        retr_optim_state = retr_optimizer.state_dict()
    else:
        retr_optim_state = None
    checkpoint = {
        "step": step,
        "model": model_to_save.state_dict(),
        "optimizer": optim_state,
        "retr_optimizer": retr_optim_state,
        "scheduler": scheduler.state_dict(),
        "retr_scheduler": retr_scheduler.state_dict() if retr_scheduler else None,
        "opt": opt,
    }
    torch.save(checkpoint, fp)
    symlink_force(epoch_path, cp)
    if opt.save_optimizer and opt.shard_optim:
        optimizer._all_states = []
