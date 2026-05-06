#!/bin/bash

# Switch to script directory (required for TAIJI platform)
cd "$(dirname "$0")"

python train.py \
    --data_dir ./data \
    --ckpt_dir ./checkpoints \
    --log_dir ./logs \
    --batch_size 256 \
    --lr 1e-4 \
    --sparse_lr 0.05 \
    --num_epochs 999 \
    --patience 5 \
    --num_workers 16 \
    --buffer_batches 20 \
    --valid_ratio 0.1 \
    --d_model 128 \
    --emb_dim 64 \
    --ns_len 10 \
    --seq_len 64 \
    --num_heads 4 \
    --ffn_hidden 256 \
    --multi_num 4 \
    --mask_type paper_causal \
    --num-pyramid-layers 6 \
    --pyramid-align 32 \
    --dropout_rate 0.01 \
    --loss_type bce \
    --use_time_buckets \
    --emb_skip_threshold 1000000 \
    --seq_id_threshold 10000 \
    --use-checkpoint \
    --seed 42
