import requests
import multiprocessing
import numpy as np
import torch

def call_retrieve_api(query_embs=None,topk=10):
    # 定义请求数据
    
    #query_embs = torch.rand(2,1024)
    bsz = query_embs.size(0)
    query_embs = query_embs.to(torch.float32)
    query_emb_list = query_embs.cpu().numpy().flatten().tolist()

    data = {
        "query_embs": query_emb_list,
        "bsz": bsz,
        "topk": topk
    }

    # 发送请求
    response = requests.post("http://paraai-n32-h-01-agent-2:29501/retrieve", json=data)

    # 处理响应
    if response.status_code == 200:
        results = response.json()
        #print(results[0])
        
        return results[0],results[1]
        
    else:
        print(f"请求失败，状态码: {response.status_code}")