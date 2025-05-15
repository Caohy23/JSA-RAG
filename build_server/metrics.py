def coverage(passages, ground_truths):
    total = len(ground_truths)
    passages = [p['text'].lower() for p in passages]
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

def evaluate_retriever(dataset):
    topk4eval = [5, 10, 20, 50]
    coverage_list = [[] for _ in range(len(topk4eval))]
    for sample in dataset:
        for i, topk in enumerate(topk4eval):
            c = coverage(sample['passages'][:topk], sample['answers'])
            coverage_list[i].append(c)
    for i, topk in enumerate(topk4eval):
        cov = sum(coverage_list[i])/len(coverage_list[i])
        print(f'Top{topk} coverage:', cov)
