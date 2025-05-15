import json
import unicodedata
import matplotlib.pyplot as plt

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

def calculate_metrics(path):
    topk = 20
    results = json.load(open(path, 'r', encoding='utf-8'))
    recall_list = []
    gen_recall_list = []
    for sample in results:
        sample['passages'] = [unicodedata.normalize('NFKD', p) for p in sample['passages'][:topk]]
        sample['answers'] = [unicodedata.normalize('NFKD', a) for a in sample['answers']]
        r = recall(sample['passages'], sample['answers'])
        recall_list.append(r)
        # gen_r = recall(sample['passages'], [sample['generation']])
        # gen_recall_list.append(gen_r)
    # print(path)
    print(f'Cover_ora_{topk}:', sum(recall_list)/len(recall_list))
    # print(f'Cover_gen_{topk}:', sum(gen_recall_list)/len(gen_recall_list))


def calculate_mul_metrics():
    topk = 20
    OracleRecalls = []
    GenRecalls = []
    total = 20000
    interval = 1000
    step_list = [i for i in range(interval, total+interval, interval)]
    for step in step_list:
        path = f'atlas_data/experiments/nq-full-jsa-vanilla-20/dev-step-{step}.json'
        # path = "atlas_data/experiments/nq-full-refresh/test-step-20000.json"
        results = json.load(open(path, 'r', encoding='utf-8'))
        recall_list = []
        gen_recall_list = []
        for sample in results:
            sample['passages'] = [unicodedata.normalize('NFKD', p) for p in sample['passages'][:topk]]
            sample['answers'] = [unicodedata.normalize('NFKD', a) for a in sample['answers']]
            r = recall(sample['passages'], sample['answers'])
            recall_list.append(r)
            gen_r = recall(sample['passages'], [sample['generation']])
            gen_recall_list.append(gen_r)
        # print(path)
        r1 = sum(recall_list)/len(recall_list)
        r2 = sum(gen_recall_list)/len(gen_recall_list)
        print(f'Step {step} Recall{topk}:', sum(recall_list)/len(recall_list))
        print(f'Step {step} Generation Recall{topk}:', sum(gen_recall_list)/len(gen_recall_list))
        OracleRecalls.append(r1)
        GenRecalls.append(r2)
    plt.figure()
    plt.plot(step_list, OracleRecalls, label='OracleRecall')
    plt.plot(step_list, GenRecalls, label='GenRecall')
    plt.legend()
    plt.savefig(f'Recall{topk}.png')

if __name__=='__main__':
    calculate_metrics('atlas_data/experiments/nq-full-jsa-8/test-step-10000.json')


