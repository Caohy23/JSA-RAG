import json
path = 'atlas_data/data/nq_data/train_with_top100_bge.json'
data = json.load(open(path, 'r'))
print('Total training data:', len(data))
bound = 19*len(data)//20
train, dev = data[:bound], data[bound:]
train_path = "atlas_data/data/nq_data/train_with_top100_bge_train.jsonl"
with open(train_path, 'w') as fp:
    for item in train:
        fp.write(json.dumps(item)+'\n')
dev_path = "atlas_data/data/nq_data/train_with_top100_bge_dev.jsonl"
with open(dev_path, 'w') as fp:
    for item in dev:
        fp.write(json.dumps(item)+'\n')