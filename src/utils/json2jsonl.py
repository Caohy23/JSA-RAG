import json
# for step in range(1000, 21000, 1000):
for step in range(1):
    path = f'atlas_data/experiments/generator-only/test_with_top100_nomic-step-1000.jsonl'
    save_path = path + 'l'
    data = json.load(open(path, 'r'))
    with open(save_path, 'w') as fp:
        for item in data:
            fp.write(json.dumps(item)+'\n')
