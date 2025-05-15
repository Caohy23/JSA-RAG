#!/bin/bash



DATA_DIR='./egs'
EVAL_FILES="${DATA_DIR}/NaturalQuestion/nq_data/test.jsonl"
SAVE_DIR="${DATA_DIR}/NaturalQuestion/JSA"
PRECISION="bf16" # "bf16"
PASSAGES="/home/bingxing2/home/scx7124/nlp_workspace/caohy/tasi_rag_platform-master/egs/NaturalQuestion/nq_data/nq_wiki_union_int.jsonl"
TOTAL_STEPS=20000
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 --nnodes=1 --master_port=29504 evaluate.py \
    --shuffle \
    --shard_optim \
    --gold_score_mode jsa \
    --gen_method  rag_seq_lh \
    --mis_step 50 \
    --mis_topk 0 \
    --use_all_mis \
    --decouple_encoder \
    --unil_postandprior \
    --use_gradient_checkpoint_generator \
    --use_gradient_checkpoint_retriever \
    --precision ${PRECISION} \
    --temperature_gold 1 \
    --temperature_score 1 \
    --temperature_jsa 0.1 \
    --refresh_index 0-40000:2001 \
    --target_maxlength 256 \
    --passages ${PASSAGES} \
    --dropout 0.1 \
    --lr 2e-5 --lr_retriever 2e-5 \
    --epsilon 1e-6 --ret_epsilon 1e-6 \
    --beta2 0.95 \
    --scheduler cosine \
    --weight_decay 0.01 \
    --eval_batch_size 1 \
    --text_maxlength 512 \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 1 \
    --accumulation_steps 1 \
    --per_gpu_embedder_batch_size 128 \
    --n_context 10 --retriever_n_context 100 \
    --n_context_gen 10 \
    --write_results \
    --task qa \
    --index_mode flat \
    --decoder_only \
    --generator_model_type "/home/bingxing2/home/scx7124/nlp_workspace/caohy/tasi_rag_platform-master/model/generator/mistralai" \
    --qa_prompt_format "{question}" \
    --retriever_model_path "/home/bingxing2/home/scx7124/nlp_workspace/caohy/tasi_rag_platform-master/model/embedding/bge-large-en" \
    --checkpoint_dir ${SAVE_DIR} \
    --name "EVAL-3-6-norebuild" \
    --use_lora \
    --train_retriever \
    --gen_doc_scores 1 \
    --model_path /home/bingxing2/home/scx7124/nlp_workspace/caohy/tasi_rag_platform-master/egs/NaturalQuestion/JSA/JSA-3-5-no-rebuild/checkpoint/step-10000 \
    --load_pretrained_weights \
