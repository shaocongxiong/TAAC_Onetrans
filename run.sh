#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# TAIJI Platform: Memory optimization for 19GiB GPU limit
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Debug info (will be visible in TAIJI logs)
echo "[TAIJI] SCRIPT_DIR: ${SCRIPT_DIR}"
echo "[TAIJI] TRAIN_DATA_PATH: ${TRAIN_DATA_PATH}"
echo "[TAIJI] TRAIN_CKPT_PATH: ${TRAIN_CKPT_PATH}"
echo "[TAIJI] TRAIN_LOG_PATH: ${TRAIN_LOG_PATH}"
echo "[TAIJI] TRAIN_TF_EVENTS_PATH: ${TRAIN_TF_EVENTS_PATH}"

# Run training with unbuffered output
python3 -u "${SCRIPT_DIR}/train.py" \
    --data_dir "${TRAIN_DATA_PATH}" \
    --ckpt_dir "${TRAIN_CKPT_PATH}" \
    --log_dir "${TRAIN_LOG_PATH}" \
    --tf_events_dir "${TRAIN_TF_EVENTS_PATH}" \
    --batch_size 64 \
    --lr 1e-4 \
    --sparse_lr 0.05 \
    --num_epochs 999 \
    --patience 5 \
    --num_workers 4 \
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
