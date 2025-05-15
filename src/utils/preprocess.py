import os
import json

def FormatConversion(path):
    data = []
    with open(path, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            sample = json.loads(line)
            data.append(sample)
    save_path = path[:-1]
    json.dump(data, open(save_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

def ReorganizeData(path):
    data = json.load(open(path, 'r', encoding='utf-8'))
    single_answer_data, multi_answer_data = [], []
    for sample in data:
        if len(sample['answers'])<=1:
            single_answer_data.append(sample)
        else:
            multi_answer_data.append(sample)
    print(path)
    print('Samples with single answer:', len(single_answer_data))
    print('Samples with multiple answers:', len(multi_answer_data))
    save_path1 = path[:-5] + '_single.json'
    save_path2 = path[:-5] + '_multi.json'
    json.dump(single_answer_data, open(save_path1, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(multi_answer_data, open(save_path2, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    sa = [json.dumps(item, ensure_ascii=False)+'\n' for item in single_answer_data]
    ma = [json.dumps(item, ensure_ascii=False)+'\n' for item in multi_answer_data]
    with open(save_path1+'l', 'w', encoding='utf-8') as fp:
        fp.writelines(sa)
    with open(save_path2+'l', 'w', encoding='utf-8') as fp:
        fp.writelines(ma)


if __name__=='__main__':
    root_path = '/mnt/workspace/liuhong/atlas-spmi/atlas_data/data/nq_data'
    for split in ['train', 'test', 'dev']:
        data_path = os.path.join(root_path, f'{split}.jsonl')
        FormatConversion(data_path)
        new_path = os.path.join(root_path, f'{split}.json')
        ReorganizeData(new_path)