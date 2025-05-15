# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import defaultdict
import asyncio
import random
import numpy as np
import subprocess
import torch
import shutil
import multiprocessing
import json
import torch.cuda
import logging
from evaluate import evaluate
from src import dist_utils, slurm, util
from src.index_io import load_or_initialize_index, save_embeddings_and_index
from src.model_io import create_checkpoint_directories, load_or_initialize_atlas_model, save_atlas_model, symlink_force
from src.options import get_options
from src.tasks import get_task
import torch.distributed as dist
#from async
import sys
import matplotlib.pyplot as plt
import threading
import torch.multiprocessing as mp
sys.path.append(r"/mnt/workspace/caohy/tasi-test/")
from rebuildgrpc.async_init_build_client import run_init
from rebuildgrpc.async_init_build_client import run_build
#FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)


os.environ["TOKENIZERS_PARALLELISM"] = "true"
GRAD_SCALE_UPPER_BOUND_MEAN: int = 1000
GRAD_SCALE_LOWER_BOUND_MEAN: float = 0.01
THRESHOLD_GRAD_STATS: int = 100
INIT_EMB_MODEL_PATH = "/mnt/workspace/caohy/tasi-test/model/embedding/bge"
LASTEST = None
logger = logging.getLogger(__name__)
async def remain(load_dir):
    task = asyncio.create_task(run_build(load_dir=load_dir))
    response = await task
def run_build_in_thread(load_dir=LASTEST):
        # try:
        asyncio.run(remain(load_dir))

        # except Exception as e:
        #     print(f"发生异常: {e}")
        # finally:
        #     print("进程即将退出")
# def run_build_in_thread(load_dir):
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     try:
#         loop.run_until_complete(remain(load_dir))
#     except Exception as e:
#         print(f"发生异常: {e}")
#     finally:
#         print("进程即将退出")
#         loop.close()
def plot(all_batch_indices, all_loss_values, label, save_path):
    """
    绘制训练损失曲线和平均损失曲线的函数，并保存图像。

    参数:
    all_batch_indices (list): 所有步的索引列表。
    all_loss_values (list): 所有步的损失值列表。
    label (str): 图表的标签。
    save_path (str): 保存图像的路径，默认为当前目录下的'loss_plot.png'。
    """
    # 计算平均损失值
    average_loss_values = [sum(all_loss_values[:i+1]) / (i+1) for i in range(len(all_loss_values))]

    # 绘制训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(all_batch_indices, all_loss_values, label=label, color='blue')

    # 绘制平均损失曲线
    plt.plot(all_batch_indices, average_loss_values, label=f'Average {label}', color='red', linestyle='--')

    plt.xlabel('Step')
    plt.ylabel(label)
    plt.legend()
    plt.grid(True)
    step = len(all_batch_indices)  # 获取步数的最大值
    plt.xlim(0, step)
    if "loss" not in label:
        plt.ylim(0, 1)
    else:
        plt.ylim(0, 20)

    # 保存图像
    save_filename = os.path.join(save_path, f"{label}.png")
    plt.savefig(save_filename)
    plt.close()
def train(
    model,
    index,
    passages,
    optimizer,
    scheduler,
    retr_optimizer,
    retr_scheduler,
    step,
    opt,
    checkpoint_path,
):
    tb_logger = util.init_tb_logger(os.path.join(opt.checkpoint_dir, opt.name), is_main=opt.is_main)
    run_stats = util.WeightedAvgStats()
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    rank = dist.get_rank()
    if not opt.use_file_passages and opt.load_index_path is None:
        # rebuild index
        indexing_start = time.time()
        if opt.grpc :
            if rank ==0:
                build_response_str = asyncio.run(run_build())
        elif opt.server:
            print("----------PASS-BUILDING--------------------------------------------------------")
        else:
            unwrapped_model.build_index(index, passages, opt.per_gpu_embedder_batch_size, logger)

        logger.info("Initial indexing time:{:.3f} min".format((time.time() - indexing_start)/60))
        if rank ==0:
            print("Initial indexing time:{:.3f} min".format((time.time() - indexing_start)/60))
    # different seed for different sampling depending on global_rank
        # if opt.save_index_path is not None:
        #     save_embeddings_and_index(index, opt)
    # assert 1==0
    torch.manual_seed(opt.global_rank + opt.seed)

    scale = 2.0
    grad_stats = defaultdict(lambda: [])
    task = get_task(opt, unwrapped_model.generator_tokenizer)
    index_refresh_scheduler = util.IndexRefreshScheduler(
        opt.refresh_index, opt.freeze_retriever_steps, opt.train_retriever
    )
    # for data_path in opt.eval_data:
    #     logger.info('Initial Evaluation')
    #     dataset_name = os.path.basename(data_path)
    #     # metrics = run_retrieval_only(model, index, opt, data_path, step)
    #     metrics = evaluate(model, index, opt, data_path, step)
    #     log_message = f"Dataset: {dataset_name}"
    #     for k, v in metrics.items():
    #         log_message += f" | {v:.3f} {k}"
    #         if tb_logger:
    #             tb_logger.add_scalar(f"{dataset_name}/{k}", v, step)
    #     logger.info(log_message)
    epoch = 0
    check1 = 1
    rebuild_time = 0
    NEW_DIR_EXIST = False
    EVAL_AFTER_REBUILD = False
    while step < opt.total_steps:
        data_iterator = task.data_iterator(
            opt.train_data, opt.global_rank, opt.world_size, repeat_if_less_than_world_size=True, opt=opt
        )
        data_iterator = filter(None, map(task.process, data_iterator))
        data_iterator = task.batch_iterator(data_iterator, opt.per_gpu_batch_size, drop_last=True, shuffle=opt.shuffle)
        epoch += 1
        all_loss = []
        all_generator_loss = []
        all_step = []
        all_accept_rate = []
        for i, batch in enumerate(data_iterator):
            # print(batch)
            # assert 1==0
            iter_stats = {}
            model.train()

            #index的更新和refresh
            if opt.rebuild:
            #if not opt.query_side_retriever_training and not opt.use_file_passages and index_refresh_scheduler.is_time_to_refresh(step):
                if  (not opt.query_side_retriever_training and not opt.use_file_passages and index_refresh_scheduler.is_time_to_refresh(step))or (opt.grpc and NEW_DIR_EXIST == True):
                    if step != 0 or opt.load_index_path is not None:  # Dont refresh index if just loaded it
                        indexing_start = time.time()
                        if opt.grpc :
                            if rank ==0:    
                                print(f"rank{rank} start rebuild")    
                                p = mp.Process(target=run_build_in_thread,args=(LASTEST,))
                                p.start()
                                NEW_DIR_EXIST = False
                        else:
                            unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
                            unwrapped_model.build_index(index, passages, opt.per_gpu_embedder_batch_size, logger)
                            EVAL_AFTER_REBUILD = True
                        iter_stats["runtime/indexing"] = (time.time() - indexing_start, 1)
                        # if opt.save_index_path is not None and not opt.grpc:
                        #     save_embeddings_and_index(index, opt)

            step += 1
            train_step_start = time.time()
            rank = dist.get_rank()
            #model.require_backward_grad_sync = (step % opt.accumulation_steps == 0)

            # model.retriever.gradient_checkpointing_enable()
            # model.post_retriever.gradient_checkpointing_enable()
            #model.generator.gradient_checkpointing_enable()
            generator_loss, retriever_loss, training_info,iter_stats = model(
                index=index,
                query=batch["query"],
                target=batch["target"],
                target_tokens=batch.get("target_tokens"),
                passages=batch["passages"] if opt.use_file_passages else None,
                batch_metadata=batch.get("metadata"),
                filtering_fun=task.filter,
                train_retriever=opt.train_retriever,
                iter_stats=iter_stats,
            )
            #print(print(f"OUT:{iter_stats}"))
            if step < opt.log_detail_num and dist.get_rank()==0:
                log_detail_path = os.path.join(checkpoint_path, f'training_info_step{step}.json')
                json.dump(training_info, open(log_detail_path, 'w'), indent=2)
            #print(f"step{step}:gen_loss:{generator_loss.item()}")
            if retriever_loss is not None and opt.train_retriever:
                train_loss = generator_loss.float() + retriever_loss
            else:
                train_loss = generator_loss.float()

            iter_stats["loss/train_loss"] = (train_loss.item(), len(batch["query"]))

            backward_start = time.time()
            #train_loss = scale * train_loss
            train_loss.backward()
            iter_stats["runtime/backward"] = (time.time() - backward_start, 1)

            model_update_start = time.time()

            #画图


            if step % opt.accumulation_steps == 0 : #and not stats["skip_example"]:
                # if opt.is_distributed and opt.shard_optim:
                #     optimizer.clip_grad_norm(1)
                #     # if opt.train_retriever:
                #     #     retr_optimizer.clip_grad_norm(scale * opt.clip)
                # else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                if not opt.separate_learning_rates:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    scheduler.step()
                    if opt.train_retriever:
                        retr_optimizer.step()
                        retr_scheduler.step()
                    optimizer.zero_grad()
                    retr_optimizer.zero_grad()

            
            iter_stats["runtime/model_update"] = (time.time() - model_update_start, 1)
            iter_stats["runtime/train_step"] = (time.time() - train_step_start, 1)
            run_stats.update(iter_stats)
            genloss = iter_stats["loss/generator_loss"][0]
            #print(run_stats.average_stats)
            if opt.gold_score_mode in ['jsa']  and opt.gen_method != "concat":
                acp = iter_stats["accept_rate"][0]
            else :
                acp = None
            if step % opt.log_freq == 0:
                log = f"EPOCH:{epoch} | {step}/{opt.total_steps}"
                log += f" | gen_loss:{genloss}"
                log += f" | train_loss:{train_loss.item()}"
                log += f" | accept_rate:{acp}"
                # for k, v in sorted(run_stats.average_stats.items()):
                #     log += f" | {k}: {v:.3g}"
                #     if tb_logger:
                #         tb_logger.add_scalar(k, v, step)
                log += f" | lr: {scheduler.get_last_lr()[0]:0.2g}"
                log += f" | Memory: {torch.cuda.max_memory_allocated()//1e9} GiB"
                # if tb_logger:
                #     tb_logger.add_scalar("lr", scheduler.get_last_lr()[0], step)
                #     tb_logger.add_scalar("ret_lr", retr_scheduler.get_last_lr()[0], step)

                logger.info(log)
                #run_stats.reset()

            all_loss.append(train_loss.item())
            all_generator_loss.append(iter_stats["loss/generator_loss"][0])
            all_step.append(step)
            if opt.gold_score_mode in ['jsa'] and opt.gen_method != "concat":
                all_accept_rate.append(iter_stats["accept_rate"][0])
                plot(all_step,all_accept_rate,"accept_rate", checkpoint_path)
            
            plot(all_step,all_generator_loss,"generator_loss", checkpoint_path)
            plot(all_step,all_loss,"all_loss", checkpoint_path)
            #画图

            #print(f"step:{step},still_training2")
            if (step % opt.eval_freq == 0 or EVAL_AFTER_REBUILD == True ) and step >=0 :
                for data_path in opt.eval_data:
                    dataset_name = os.path.basename(data_path)
                    # metrics = run_retrieval_only(model, index, opt, data_path, step)
                    metrics = evaluate(model, index, opt, data_path,logger, step = step)
                    log_message = f"Dataset: {dataset_name}"
                    for k, v in metrics.items():
                        log_message += f" | {v:.3f} {k}"
                        if tb_logger:
                            tb_logger.add_scalar(f"{dataset_name}/{k}", v, step)
                    logger.info(log_message)
                EVAL_AFTER_REBUILD = False
            #print(f"step:{step},still_training3")
            if step % opt.save_freq == 0 :
                save_atlas_model(
                    unwrapped_model,
                    optimizer,
                    scheduler,
                    retr_optimizer,
                    retr_scheduler,
                    step,
                    opt,
                    checkpoint_path,
                    f"step-{step}",
                )
            #print(f"step:{step},still_training4")
            if step % opt.save_build_retriever_step == 0 and rank == 0:
                # if opt.passage_model_path !=INIT_EMB_MODEL_PATH and step !=opt.save_build_retriever_step:
                #     target_path = output_dir_passage1
                #     if os.path.exists(target_path):
                #         shutil.rmtree(target_path)
                #         print(f"目录 {target_path} 及其所有内容已被删除。")
                #     else:
                #         print(f"目录 {target_path} 不存在。")
                #print(f"rank{rank}进入")
                # if opt.rebuild :
                output_dir_passage = os.path.join(checkpoint_path, "bge_passage_Embedding_Ret")
                output_dir_passage1 = os.path.join(output_dir_passage ,f"step-{step}")
                lastest1 = os.path.join(output_dir_passage, "lastest")
                os.makedirs(output_dir_passage1, exist_ok=True)
                unwrapped_model.retriever.passage_retriever.save_models(output_dir_passage1)
                unwrapped_model.retriever_tokenizer.save_pretrained(output_dir_passage1)
                symlink_force(output_dir_passage1,lastest1)
                # NEW_DIR_EXIST = True
                # LASTEST = output_dir_passage1

                output_dir0 = os.path.join(checkpoint_path, "bge_query_Embedding_Ret")
                output_dir = os.path.join(output_dir0 ,f"step-{step}")
                lastest = os.path.join(output_dir0, "lastest")
                os.makedirs(output_dir, exist_ok=True)
                unwrapped_model.retriever.query_retriever.save_models(output_dir)
                unwrapped_model.retriever_tokenizer.save_pretrained(output_dir)                    
                symlink_force(output_dir,lastest)

                # output_dir_passage = os.path.join(checkpoint_path, "bge_passage_Embedding_Ret")
                # output_dir_passage1 = os.path.join(output_dir_passage ,f"step-{step}")
                # lastest1 = os.path.join(output_dir_passage, "lastest")
                # os.makedirs(output_dir_passage1, exist_ok=True)
                # unwrapped_model.retriever.passage_retriever.save_models(output_dir_passage1)
                # unwrapped_model.retriever_tokenizer.save_pretrained(output_dir_passage1)
                # symlink_force(output_dir_passage1,lastest1)
                # NEW_DIR_EXIST = True
                # LASTEST = lastest1
                # print("save successfully",rank,step)
            #print(f"step:{step},still_training5")
            if step > opt.total_steps:
                exit()
            #print(f"step:{step},still_training6")    
            dist.barrier()
            #print(f"step:{step},still_training7")
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

    logger = util.init_logger(opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, f"run{opt.gen_method}.log"))
    if opt.is_main:
        options.print_options(opt)

    logger.info(f"world size: {dist_utils.get_world_size()}")
    rank = dist.get_rank()
    model, optimizer, scheduler, retr_optimizer, retr_scheduler, opt, step = load_or_initialize_atlas_model(opt)
    if opt.grpc :
        if rank ==0:
            passages_dir = opt.passages[1]
            print(passages_dir)
            asyncio.run(run_init(passages_dir))
            print("init over")
        index, passages = None,None
    elif opt.server:
        index, passages = None,None
    else:    
        index, passages = load_or_initialize_index(opt)
    #assert 1==0
    # if not opt.use_file_passages and not opt.grpc:
    #     logger.info('Total passages:{}, embeddings size:{}'.format(len(passages), index.embeddings.shape))
    dist_utils.barrier()
    if opt.is_distributed:
        if opt.shard_grads:
            import fairscale.nn.data_parallel
    # # 初始化 FSDP 模型
    # auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=100)
    # mixed_precision = MixedPrecision(
    #     param_dtype=torch.float16,  # 参数使用 FP16
    #     reduce_dtype=torch.float32,  # 梯度规约使用 FP32
    #     buffer_dtype=torch.float32,  # 缓冲区使用 FP32
    # )
    # model = FSDP(
    #     model,
    #     sharding_strategy=ShardingStrategy.FULL_SHARD,  # 完全分片
    #     auto_wrap_policy=auto_wrap_policy,
    #     mixed_precision=mixed_precision,
    #     device_id=torch.cuda.current_device(),
    # )
            model = fairscale.nn.data_parallel.ShardedDataParallel(
                model, optimizer, auto_refresh_trainable=False
            )

        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                find_unused_parameters=True,
            )
            model._set_static_graph()

    
    # unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    # test_input = unwrapped_model.generator_tokenizer(['My favourite condiment is'], return_tensors="pt").to("cuda")
    # print(test_input)
    # generated_ids = unwrapped_model.generator.generate(
    #                 input_ids=test_input["input_ids"],
    #                 attention_mask=test_input['attention_mask'],
    #                 max_length=unwrapped_model.opt.generation_max_length,
    #                 min_length=unwrapped_model.opt.generation_min_length,
    #                 output_scores=True,
    #                 length_penalty=opt.generation_length_penalty
    #             )
    # print(unwrapped_model.generator_tokenizer.batch_decode(generated_ids)[0])
    # assert 1==0
    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}")
    #     print(f"Parameter dtype: {param.dtype}")
    #     print(f"Parameter shape: {param.shape}\n")
    # assert 1==0

    logger.info("Start training")
    dist_utils.barrier()
    train(
        model,
        index,
        passages,
        optimizer,
        scheduler,
        retr_optimizer,
        retr_scheduler,
        step,
        opt,
        checkpoint_path,
    )