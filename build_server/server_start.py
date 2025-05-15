import os
import json
from tqdm import tqdm
import faiss
import argparse
import torch
import torch.distributed as dist
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from retriever import RetModel
from multiprocessing import Process, Manager
import pickle
from pydantic import BaseModel


# 定义输入请求模型
class RetrieveRequest(BaseModel):
    query_embs: list
    bsz: int = 1
    topk: int = 10

class rebuildRequest(BaseModel):
    checkpoint_path: str
    response_url: str

app = FastAPI()


class DistributedFaissIndex(object):
    def __init__(self, embedding_file_pathes, checkpoint_pathes, nlist=20, metric=faiss.METRIC_L2, gpu_ids=[0,1,2,3], gpu_batch_size=512, max_memory_number=10000):

        print("build --------------------------------------------------")
        self.doc_map = {}
        dimension = 1024  # 假设embedding的维度是1024

        # 检查GPU数量
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")

        # 初始化GPU资源
        res_list = [faiss.StandardGpuResources() for _ in gpu_ids]

        # 创建分片索引（IndexShards）
        self.index_shards = faiss.IndexShards(dimension, True, True)  
        gpu_indexes = []

        # 创建每个 GPU 索引
        for i, gpu_id in enumerate(gpu_ids):
            if gpu_id >= num_gpus:
                raise ValueError(f"GPU {gpu_id} not available (Total GPUs: {num_gpus})")

            # 创建 GPU 索引（使用不需要训练的 IndexFlatIP）
            cpu_index = faiss.IndexFlatIP(dimension)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True  # 启用 FP16 节省显存
            gpu_index = faiss.index_cpu_to_gpu(res_list[i], gpu_id, cpu_index, co)
            gpu_indexes.append(gpu_index)

        # 分批次处理数据
        last_id = 0  # 文档ID计数器
        # 第一阶段：统计总数据量（用于进度条）
        total_embeddings = 0
        for path in embedding_file_pathes:
            with open(path, "rb") as f:
                while True:
                    try:
                        total_embeddings += len(pickle.load(f))
                    except EOFError:
                        break

        # 第二阶段：实际加载数据
        batch_embeddings = []
        progress_bar = tqdm(total=total_embeddings, desc="Loading embeddings")
        for gpu_idx in range(len(embedding_file_pathes)):
            batch_embeddings = []
            embedding_file_path = embedding_file_pathes[gpu_idx]
            with open(embedding_file_path, "rb") as f:
                while True:
                    try:
                        chunk = pickle.load(f)
                        for data in chunk:
                            # 构建文档映射
                            self.doc_map[last_id] = data["passage"]
                            # 收集embeddings
                            batch_embeddings.append(data["emb"])
                            last_id += 1
                            
                            progress_bar.update(1)
                            
                            # 分批次添加

                                
                    except EOFError:
                        break

            current_gpu_index = gpu_indexes[gpu_idx]


            if isinstance(batch_embeddings, list):
                batch_embeddings = np.array(batch_embeddings, dtype=np.float32)
            elif isinstance(batch_embeddings, torch.Tensor):
                batch_embeddings = self._cast_to_numpy(batch_embeddings)
            current_gpu_index.add(batch_embeddings)
            print(f"embedding add to index{gpu_idx}")
        for gpu_index in gpu_indexes:
            self.index_shards.add_shard(gpu_index)
        progress_bar.close()
        
        #self.index_shards.add_shard(gpu_index)
        print("index——load-over*****************************")

    @torch.no_grad()
    def _cast_to_torch32(self, embeddings: torch.tensor) -> torch.tensor:
        """
        Converts a torch tensor to a contiguous float 32 torch tensor.
        """
        return embeddings.type(torch.float32).contiguous()

    @torch.no_grad()
    def _cast_to_numpy(self, embeddings: torch.tensor) -> np.ndarray:
        """
        Converts a torch tensor to a contiguous numpy float 32 ndarray.
        """
        return embeddings.cpu().to(dtype=torch.float16).numpy().astype("float32")#.copy(order="C")
    def _add_batch(self, batch_embeddings):
        """处理单批次数据添加"""
        # 转换为numpy数组
        if isinstance(batch_embeddings, list):
            batch_embeddings = np.array(batch_embeddings, dtype=np.float32)
        elif isinstance(batch_embeddings, torch.Tensor):
            batch_embeddings = self._cast_to_numpy(batch_embeddings)
        
        # 确保数据类型正确
        assert batch_embeddings.dtype == np.float32, "必须使用float32格式"
        faiss.normalize_L2(batch_embeddings)
        # 直接添加数据（IndexFlatL2不需要训练）
        self.index_shards.add(batch_embeddings)
    def search_knn(self, query_embs, topk):

        query_embs = query_embs.cpu().numpy()
        faiss.normalize_L2(query_embs)
        R = self.index_shards.search(query_embs, topk)
        #print(f"R {R}")
        D, I = R

        all_docs = []
        all_scores = []

        # 遍历每个查询向量的搜索结果
        for query_idx in range(len(I)):
            #print(I[query_idx])
            docs = [self.doc_map[i] for i in I[query_idx]]
            scores = [float(x) for x in D[query_idx]]

            # 打印每个查询向量的结果
            # for i, doc, score in zip(I[query_idx], docs, scores):
            #     #print(f"i {i} : doc {doc}, score {score}")

            all_docs.append(docs)
            all_scores.append(scores)

        return all_docs, all_scores
def get_pkl_files_in_directory(directory):
    pkl_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))
    return pkl_files
directory = '/home/bingxing2/home/scx7124/nlp_workspace/caohy/RAG/embedding/wiki'
passage_path = get_pkl_files_in_directory(directory)
#passage_path = ["/home/bingxing2/home/scx7124/nlp_workspace/caohy/RAG/embedding/wiki/embeddings_0.pkl","/home/bingxing2/home/scx7124/nlp_workspace/caohy/RAG/embedding/wiki/embeddings_1.pkl"]
checkpoint_pathes = ["/home/bingxing2/home/scx7124/nlp_workspace/caohy/tasi_rag_platform-master/model/embedding/bge-large-en"]
global_index = DistributedFaissIndex(passage_path, checkpoint_pathes)

query_emb = torch.rand(2,1024)
docs, scores = global_index.search_knn(query_emb,5)
print(f"docs {docs}, scores {scores}")

@app.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    global global_index
    if global_index is None:
        raise HTTPException(status_code=500, detail="Index is not ready")
    query_embs = torch.tensor(request.query_embs).view(request.bsz, -1)
    #global_index.search_knn(request.query, request.topk)
    relevant_docs, scores = global_index.search_knn(query_embs, request.topk)
    return [relevant_docs, scores]

@app.post("/rebuild")
def rebuild(request:rebuildRequest):
    index = DistributedFaissIndex(passage_path, request.checkpoint_path)
    global global_index
    global_index = index
    requests.post(request.response_url, json={"status": "success"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=29501)
