#!/bin/bash


DATA_DIR='./egs'
TRAIN_FILE="${DATA_DIR}/NaturalQuestion/nq_data/train.jsonl"
EVAL_FILES="${DATA_DIR}/NaturalQuestion/nq_data/test.jsonl"
SAVE_DIR="${DATA_DIR}/NaturalQuestion/JSA"
PRECISION="bf16" # "bf16"
PASSAGES="/home/bingxing2/home/scx7124/nlp_workspace/caohy/tasi_rag_platform-master/kb/wiki/enwiki-dec2018-doc/text-list-100-sec-int.jsonl" #"/home/bingxing2/home/scx7124/nlp_workspace/caohy/tasi_rag_platform-master/egs/NaturalQuestion/nq_data/nq_wiki_union_int.jsonl"
TOTAL_STEPS=20000
PRETRAINED_INDEX="kb/wiki/enwiki-dec2018"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 --nnodes=1 --master_port=29504 train.py \
    --shuffle \
    --shard_optim \
    --gold_score_mode jsa \
    --gen_method  rag_seq_xy \
    --mis_step 50 \
    --mis_topk 0 \
    --use_all_mis \
    --decouple_encoder \
    --unil_postandprior \
    --use_gradient_checkpoint_generator \
    --use_gradient_checkpoint_retriever \
    --precision ${PRECISION} \
    --query_side_retriever_training \
    --temperature_gold 1 --temperature_score 1 \
    --temperature_jsa  0.1 \
    --refresh_index 0-40000:40000 \
    --target_maxlength 256 \
    --passages ${PASSAGES} \
    --dropout 0.1 \
    --separate_learning_rates \
    --lr 2e-5 --lr_retriever 1e-5 \
    --epsilon 1e-7 --ret_epsilon 1e-7 \
    --beta2 0.95 \
    --scheduler cosine \
    --weight_decay 0.01 \
    --text_maxlength 512 \
    --train_data ${TRAIN_FILE} \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 1 \
    --accumulation_steps 1 \
    --per_gpu_embedder_batch_size 64 \
    --n_context 10 --retriever_n_context 100 \
    --n_context_gen 10 \
    --eval_freq 5000 \
    --log_freq 10 \
    --total_epochs 10 \
    --total_steps ${TOTAL_STEPS} \
    --warmup_steps 1000 \
    --save_freq 5000 \
    --write_results \
    --task qa \
    --index_mode faiss \
    --faiss_index_type ivfpq \
    --faiss_code_size 32 \
    --decoder_only \
    --generator_model_type "model/generator/mistralai" \
    --qa_prompt_format "{question}" \
    --retriever_model_path "model/embedding/bge-large-en" \
    --checkpoint_dir ${SAVE_DIR} \
    --name "JSA-1" \
    --use_lora \
    --train_retriever \
    --server \
    --gen_doc_scores 0.001 \
