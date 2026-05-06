#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# OneTrans training entry point
# Default hyperparameters tuned for OneTrans architecture
python3 -u "${SCRIPT_DIR}/train.py" \
    --d_model 128 \
    --emb_dim 64 \
    --ns_len 4 \
    --seq_len 64 \
    --num_heads 4 \
    --ffn_hidden 256 \
    --multi_num 4 \
    --mask_type origin \
    --batch_size 256 \
    --lr 1e-4 \
    --sparse_lr 0.05 \
    --num_workers 8 \
    --num_epochs 999 \
    --patience 5 \
    --loss_type bce \
    --dropout_rate 0.01 \
    --emb_skip_threshold 1000000 \
    --seq_id_threshold 10000 \
    --use_time_buckets \
    "$@"
