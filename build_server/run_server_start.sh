#!/bin/bash




PASSAGES="/home/bingxing2/home/scx7124/nlp_workspace/caohy/tasi_rag_platform-master/data/corpora/wiki/enwiki-dec2018-doc/text-list-100-sec.jsonl /home/bingxing2/home/scx7124/nlp_workspace/caohy/tasi_rag_platform-master/kb/wiki/enwiki-dec2018-doc/infobox.jsonl"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python  server_start.py \