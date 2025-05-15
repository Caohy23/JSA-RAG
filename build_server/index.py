import os
import math
import json
import copy
import torch
from tqdm import tqdm
import torch.distributed as dist
import numpy as np
import pickle
import faiss
import faiss.contrib.torch_utils
from retriever import get_embeddings
from utils import varsize_all_gather, get_varsize, varsize_gather


def load_passages(filenames, maxload=-1):
    def process_jsonl(
        fname,
        counter,
        passages,
        world_size,
        global_rank,
        maxload,
    ):
        def load_item(line):
            if line.strip() != "":
                item = json.loads(line)
                assert "id" in item
                if "title" in item and "section" in item and len(item["section"]) > 0:
                    item["title"] = f"{item['title']}: {item['section']}"
                return item
            else:
                print("empty line")
        iters = tqdm(open(fname)) if global_rank==0 else open(fname)
        for line in open(fname):
            if maxload > -1 and counter >= maxload:
                break

            ex = None
            if (counter % world_size) == global_rank:
                ex = load_item(line)
                passages.append(ex)
            counter += 1
        return passages, counter

    counter = 0
    passages = []
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    for filename in filenames:

        passages, counter = process_jsonl(
            filename,
            counter,
            passages,
            world_size,
            global_rank,
            maxload,
        )

    return passages

def _to_cuda(tok_dict):
    return {k: v.cuda() for k, v in tok_dict.items()}

@torch.no_grad()
def build_index(args, passages, retriever_model, dim):
    rank = dist.get_rank()
    index = DistributedIndex()
    index.init_embeddings(passages, dim)
    target_folder = "embedding/msmarco"

    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    embedding_file_path = os.path.join(target_folder, f"embeddings_{rank}.pkl")
    # calculate embeddings
    retriever_model.eval()
    encoder_fp16 = retriever_model.p_encoder.module if hasattr(retriever_model.p_encoder, 'module') else retriever_model.p_encoder
    encoder_fp16 = copy.deepcopy(encoder_fp16).half().eval()
    n_batch = math.ceil(len(passages) / args.gpu_embedder_batch_size)
    all_embeddings = []
    total = 0
    iters = tqdm(range(n_batch)) if rank==0 else range(n_batch)
    # get document format
    if 'nomic' in retriever_model.p_encoder_path:
        format_str = "search_document: {title} {text}"
    else:
        format_str = "{title} {text}"
    #with open(embedding_file_path, "ab") as f:
    with open(embedding_file_path, "ab") as f:
        for i in iters:
            batch = passages[i * args.gpu_embedder_batch_size : (i + 1) * args.gpu_embedder_batch_size]
            batch_to_embedding = [format_str.format(**example) for example in batch]
            batch_enc = retriever_model.p_tokenizer(
                batch_to_embedding,
                padding="longest",
                return_tensors="pt",
                max_length=args.text_maxlength,
                truncation=True,
            )

            outputs = encoder_fp16(**_to_cuda(batch_enc))
            embeddings = get_embeddings(outputs, retriever_model.p_encoder_path, model_input=batch_enc).cpu().numpy()
            #index.embeddings[:, total : total + len(embeddings)] = embeddings.T
            total += len(embeddings)
            # 分批将当前批次的嵌入向量写入文件
            saved_data = [{"emb": embeddings[i], "passage": batch[i]} for i in range(len(embeddings))]
            if i==0:
                print(saved_data)
            pickle.dump(saved_data, f)
            if i % 500 == 0 and i > 0 and rank == 0:
                print(f"Number of passages encoded: {total}")
    dist.barrier()
    print(f"{total} passages encoded on process: {dist.get_rank()}")


def load_existing_index(args):
    index = DistributedIndex()
    index.load_index(args.load_index_path, args.index_n_shards)
    return index



def serialize_listdocs(ids):
    ids = pickle.dumps(ids)
    ids = torch.tensor(list(ids), dtype=torch.uint8).cuda()
    return ids


def deserialize_listdocs(ids):
    return [pickle.loads(x.cpu().numpy().tobytes()) for x in ids]


class DistributedIndex(object):
    def __init__(self):
        self.embeddings = None
        self.doc_map = dict()

    def init_embeddings(self, passages, dim = 768):
        self.doc_map = {i: doc for i, doc in enumerate(passages)}
        self.embeddings = torch.zeros(dim, (len(passages)), dtype=torch.float16)
        self.embeddings = self.embeddings.cuda()

    def _get_saved_embedding_path(self, save_dir: str, shard: int) -> str:
        return os.path.join(save_dir, f"embeddings.{shard}.pt")

    def _get_saved_passages_path(self, save_dir: str, shard: int) -> str:
        return os.path.join(save_dir, f"passages.{shard}.pt")

    def save_index(self, path: str, total_saved_shards: int, overwrite_saved_passages: bool = False) -> None:
        """
        Saves index state to disk, which can later be loaded by the load_index method.
        Specifically, it saves the embeddings and passages into total_saved_shards separate file shards.
        This option enables loading the index in another session with a different number of workers, as long as the number of workers is divisible by total_saved_shards.
        Note that the embeddings will always be saved to disk (it will overwrite any embeddings previously saved there).
        The passages will only be saved to disk if they have not already been written to the save directory before, unless the option --overwrite_saved_passages is passed.
        """
        assert self.embeddings is not None
        rank = dist.get_rank()
        ws = dist.get_world_size()
        assert total_saved_shards % ws == 0, f"N workers must be a multiple of shards to save"
        shards_per_worker = total_saved_shards // ws
        n_embeddings = self.embeddings.shape[1]
        embeddings_per_shard = math.ceil(n_embeddings / shards_per_worker)
        assert n_embeddings == len(self.doc_map), len(self.doc_map)
        for shard_ind, (shard_start) in enumerate(range(0, n_embeddings, embeddings_per_shard)):
            shard_end = min(shard_start + embeddings_per_shard, n_embeddings)
            shard_id = shard_ind + rank * shards_per_worker  # get global shard number
            passage_shard_path = self._get_saved_passages_path(path, shard_id)
            if not os.path.exists(passage_shard_path) or overwrite_saved_passages:
                passage_shard = [self.doc_map[i] for i in range(shard_start, shard_end)]
                with open(passage_shard_path, "wb") as fobj:
                    pickle.dump(passage_shard, fobj, protocol=pickle.HIGHEST_PROTOCOL)
            embeddings_shard = self.embeddings[:, shard_start:shard_end]
            embedding_shard_path = self._get_saved_embedding_path(path, shard_id)
            torch.save(embeddings_shard, embedding_shard_path)

    def load_index(self, path: str, total_saved_shards: int):
        """
        Loads sharded embeddings and passages files (no index is loaded).
        """
        rank = dist.get_rank()
        ws = dist.get_world_size()
        assert total_saved_shards % ws == 0, f"N workers must be a multiple of shards to save"
        shards_per_worker = total_saved_shards // ws
        passages = []
        embeddings = None
        for shard_id in range(rank * shards_per_worker, (rank + 1) * shards_per_worker):
            passage_shard_path = self._get_saved_passages_path(path, shard_id)
            with open(passage_shard_path, "rb") as fobj:
                passages.append(pickle.load(fobj))
            embeddings_shard_path = self._get_saved_embedding_path(path, shard_id)
            embeddings_shard = torch.load(embeddings_shard_path, map_location="cpu").cuda()
            if embeddings is None:
                embeddings = embeddings_shard
            else:
                embeddings = torch.cat([embeddings, embeddings_shard], dim=1)
        self.doc_map = {}
        n_passages = 0
        for chunk in passages:
            for p in chunk:
                self.doc_map[n_passages] = p
                n_passages += 1
        self.embeddings = embeddings

    def _compute_scores_and_indices(self, allqueries: torch.tensor, topk: int):
        """
        Computes the distance matrix for the query embeddings and embeddings chunk and returns the k-nearest neighbours and corresponding scores.
        """
        scores = torch.matmul(allqueries.half(), self.embeddings)
        scores, indices = torch.topk(scores, topk, dim=1)

        return scores, indices


    @torch.no_grad()
    def search_knn(self, queries, topk):
        """
        Conducts exhaustive search of the k-nearest neighbours using the inner product metric.
        """
        allqueries = varsize_all_gather(queries) # gather all query
        allsizes = get_varsize(queries)
        allsizes = np.cumsum([0] + allsizes.cpu().tolist())
        # compute scores for the part of the index located on each process
        # In the following, B is the total batch size on all GPUs
        scores, indices = self._compute_scores_and_indices(allqueries, topk) # (B,K)
        embeddings = self.embeddings[:, indices.view(-1)] # (H, B*K)
        embeddings = embeddings.transpose(0,1).contiguous().view(indices.size(0), indices.size(1), -1) # (B, K, H)
        indices = indices.tolist() 
        docs = [[self.doc_map[x] for x in sample_indices] for sample_indices in indices] # a list of text of shape (B,K)
        if torch.distributed.is_initialized():
            docs = [docs[allsizes[k] : allsizes[k + 1]] for k in range(len(allsizes) - 1)] # grouped by GPU, [[gpu1 batch], [gpu2 batch], ...]
            docs = [serialize_listdocs(x) for x in docs] # convert into byte series
            scores = [scores[allsizes[k] : allsizes[k + 1]] for k in range(len(allsizes) - 1)]
            embeddings = [embeddings[allsizes[k] : allsizes[k + 1]] for k in range(len(allsizes) - 1)]
            gather_docs = [varsize_gather(docs[k], dst=k, dim=0) for k in range(dist.get_world_size())]
            gather_scores = [
                varsize_gather(scores[k], dst=k, dim=1) for k in range(dist.get_world_size())
            ]
            gather_embeddings = [varsize_gather(embeddings[k], dst=k, dim=0) for k in range(dist.get_world_size())]
            rank_scores = gather_scores[dist.get_rank()]
            rank_docs = gather_docs[dist.get_rank()]
            rank_embeddings = gather_embeddings[dist.get_rank()]
            scores = torch.cat(rank_scores, dim=1) # (b, ngpu*topk)
            embeddings = torch.cat(rank_embeddings, dim=1)
            rank_docs = deserialize_listdocs(rank_docs)
            merge_docs = [[] for _ in range(queries.size(0))] # (b, ngpu*topk)
            for docs in rank_docs:
                for k, x in enumerate(docs):
                    merge_docs[k].extend(x)
            docs = merge_docs
        _, subindices = torch.topk(scores, topk, dim=1)
        embeddings = torch.gather(
            embeddings, 1, subindices.unsqueeze(-1).expand(subindices.size(0), subindices.size(1), embeddings.size(-1))) # (b, k, H)
        scores = scores.tolist()
        subindices = subindices.tolist()
        # Extract topk scores and associated ids
        scores = [[scores[k][j] for j in idx] for k, idx in enumerate(subindices)]
        docs = [[docs[k][j] for j in idx] for k, idx in enumerate(subindices)]
        return docs, scores, embeddings

    def is_index_trained(self) -> bool:
        return True
