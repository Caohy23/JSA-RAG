from cProfile import label
import matplotlib.pyplot as plt
import re
import numpy as np
import os
# reader_loss, retriever_loss, train_loss = [], [], []
# steps = []
# with open('atlas_data/experiments/nq-full-std/run.log', 'r') as fp:
#     for line in fp.readlines():
#         if 'loss/train_loss' in line:
#             steps.append(float(re.findall(r'(\d+) / 10000', line)[0]))
#             reader_loss.append(float(re.findall(r'loss/reader_loss: (\d+\.?\d*)', line)[0]))
#             retriever_loss.append(float(re.findall(r'loss/retriever_loss: (\d+\.?\d*)', line)[0]))
#             train_loss.append(float(re.findall(r'loss/train_loss: (\d+\.?\d*)', line)[0]))
# plt.figure()
# plt.plot(steps, reader_loss, label='reader_loss')
# plt.savefig('reader_loss.png')
# plt.plot(steps, retriever_loss, label='retriever_loss')
# plt.plot(steps, train_loss, label='train_loss')
# plt.legend()
# plt.savefig('train_loss.png')
path = 'atlas_data/experiments/jsa-test-4-2/run.log'
dir_path = os.path.dirname(path)
recall_nums = [5, 10, 20, 50]
recalls = {}
for key in recall_nums:
    recalls[key] = []
steps = []
accept_rates = []
mrr_list = []
mrr_rev_list = []
reader_loss = []
retriever_loss = []
with open(path, 'r') as fp:
    for line in fp.readlines():
        if 'Sampling info' in line:
            for r in recall_nums:
                value = re.findall(r'\'recall{}\': (\d+\.?\d*)'.format(r), line)[0]
                recalls[r].append(float(value))
        if "accept rate:" in line:
            ac = re.findall(r'accept rate: (\d\.\d+)', line)[0]
            accept_rates.append(float(ac))
        if "MRR" in line:
            mrr_list.append(float(re.findall(r'MRR: (\d+\.?\d*)', line)[0]))
            mrr_rev_list.append(float(re.findall(r'MRR_rev: (\d+\.?\d*)', line)[0]))
        if 'reader_loss' in line:
            steps.append(float(re.findall(r'(\d+) / 10000', line)[0]))
            reader_loss.append(float(re.findall(r'loss/reader_loss: (\d+\.?\d*)', line)[0]))
            retriever_loss.append(float(re.findall(r'loss/retriever_loss: (\d+\.?\d*)', line)[0]))

        
# print(recalls[5])
smooth_window = 1
# for r in recall_nums:
#     smooth_recall = []
#     for i in range(0, len(recalls[r]), smooth_window):
#         smooth_recall.append(np.mean(recalls[r][i:i+smooth_window]))
#     plt.figure()
#     plt.plot(smooth_recall)
#     plt.savefig(os.path.join(dir_path, f'recall{r}.png'))
plt.figure()
acr = []
for i in range(0, len(accept_rates), smooth_window):
    acr.append(np.mean(accept_rates[i:i+smooth_window]))
plt.plot(acr)
plt.savefig(os.path.join(dir_path, 'accept_rate.png'))
plt.figure()
plt.plot(steps, reader_loss, label='reader_loss')
plt.savefig(os.path.join(dir_path, 'reader_loss.png'))
plt.figure()
plt.plot(steps, retriever_loss, label='retriever_loss')
plt.savefig(os.path.join(dir_path, 'retriever_loss.png'))
plt.figure()
plt.plot(steps, mrr_list, label='mrr')
plt.plot(steps, mrr_rev_list, label='mrr_rev')
plt.legend()
plt.savefig(os.path.join(dir_path, 'MRR.png'))