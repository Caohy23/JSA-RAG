import json

def calculate_recall_and_mrr(file1_path, file2_path):
    # 读取文件1
    with open(file1_path, 'r', encoding='utf-8') as f1:
        lines1 = f1.readlines()
        data1 = [json.loads(line) for line in lines1]

    # 读取文件2
    with open(file2_path, 'r', encoding='utf-8') as f2:
        lines2 = f2.readlines()
        data2 = [json.loads(line) for line in lines2]

    # 创建一个字典，用于快速查找文件1中每个问题对应的gold_doc
    question_gold_doc = {item["question"]: item["gold_doc"] for item in data1}

    total_recall_at_1 = 0
    total_recall_at_10 = 0
    total_mrr = 0
    valid_count = 0

    # 遍历文件2中的每个问题
    for item2 in data2:
        query = item2["question"]
        passages = item2["passages"]
        passage_texts = [passage["id"] for passage in passages]
        
        # 检查该问题是否存在于文件1中
        if query in question_gold_doc:
            gold_doc = question_gold_doc[query]
            gold_text = gold_doc#["text"]
            
            # 计算recall@1
            if gold_text in passage_texts[:1]:
                total_recall_at_1 += 1
            
            # 计算recall@10
            if gold_text in passage_texts[:10]:
                total_recall_at_10 += 1
            
            # 计算MRR@10
            mrr = 0
            try:
                rank = passage_texts.index(gold_text) + 1
                if rank <= 10:
                    mrr = 1.0 / rank
            except ValueError:
                pass
            total_mrr += mrr
            
            valid_count += 1

    # 计算平均指标
    if valid_count > 0:
        recall_at_1 = total_recall_at_1 / valid_count
        recall_at_10 = total_recall_at_10 / valid_count
        average_mrr = total_mrr / valid_count
    else:
        recall_at_1 = 0
        recall_at_10 = 0
        average_mrr = 0

    return recall_at_1, recall_at_10, average_mrr

# 替换为你的文件路径
file1_path = '/home/bingxing2/home/scx7124/nlp_workspace/caohy/RAG/test_woCNA_gold.jsonl'
file2_path = '/home/bingxing2/home/scx7124/nlp_workspace/caohy/RAG/test_woCNA_with_top100_post.jsonl'
recall_at_1, recall_at_10, mrr = calculate_recall_and_mrr(file1_path, file2_path)
print(f"R@1: {recall_at_1}")
print(f"R@10: {recall_at_10}")
print(f"MRR@10: {mrr}")