import json
# for step in range(1000, 21000, 1000):
for step in range(1):
    path = f'atlas_data/experiments/emdr-gpt/test_with_top100_nomic-step-10000.jsonl'
    data = []
    with open(path, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            sample = json.loads(line)
            # passages = []
            # for p in sample['passages']:
            #     passages.append('{} {}'.format(p['title'], p['text']))
            # sample['passages'] = passages
            # if len(sample['metadata'])==0:
            #     sample.pop('metadata')
            data.append(sample)
    save_path = path[:-1]
    json.dump(data, open(save_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
