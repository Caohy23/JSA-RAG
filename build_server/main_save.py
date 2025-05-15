import os
import argparse
import json
import time
import logging
import random
import numpy as np
import torch
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from retriever import RetModel
from index import load_passages, build_index, load_existing_index
from metrics import evaluate_retriever

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def main_worker(local_rank, args):
    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.ngpus, rank=local_rank)
    print(local_rank)
    torch.cuda.set_device(local_rank)

    # Temporarily, we only use retriever to evaluate on NQ, so we dont use DDP
    if local_rank==0:
        print("Initilize models")
    retriever = RetModel(args.retriever_model).to(local_rank)
    if local_rank==0:
        print("Building or loading indices")

    passages = load_passages(args.passage_path)
    #passages = passages[:len(passages)//100] # for debugging
    build_index(args, passages, retriever, dim=retriever.p_config.hidden_size)


        


    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--dist_url", type=str,
                        help="communication url for ddp training", default="tcp://localhost:13457")

    parser.add_argument("--generation_model", type=str, help="the path of generation model")
    parser.add_argument(
        "--retriever_model",
        nargs="+",
        default=[],
        help="the path of retriever model. If there are more than one model, the first will be used to encode queries and the seconde will be used to encode passages")

    parser.add_argument("--passage_path", nargs="+", default=[], help="the path of the total passages, must be a list of jsonl files")
    parser.add_argument("--query_path", nargs="+", default=[], help="the path of queries, must be json file")
    parser.add_argument("--name", type=str, help="method and whether to load")
    parser.add_argument("--save_index", action="store_true", help="whether to store the indices")
    parser.add_argument("--post", action="store_true", help="use post to retrieve")
    parser.add_argument("--save_index_path", type=str)
    parser.add_argument("--load_index_path", type=str)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--local_rank", type=int,default=-1)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--text_maxlength", type=int, default=512)
    parser.add_argument("--gpu_embedder_batch_size",type=int, default=128)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--index_n_shards", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="", help="save top100 to every data doc if use load")
    parser.add_argument("--output_file", type=str, default="", help="Predictions will be written to this file.")

    # reranker args
    parser.add_argument("--train_reranker", action="store_true")
    parser.add_argument("--load", action="store_true")

    args = parser.parse_args()
    # for query_path in args.query_path:
    #     print(query_path)
    # assert 1==0
    mp.spawn(main_worker, nprocs=args.ngpus, args=(args,))



if (__name__ == "__main__"):
    main()
