#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================================"
echo "[TAIJI] run.sh STARTED at $(date)"
echo "[TAIJI] PWD: $(pwd)"
echo "[TAIJI] SCRIPT_DIR: ${SCRIPT_DIR}"
echo "[TAIJI] whoami: $(whoami)"
echo "[TAIJI] python3: $(which python3 2>/dev/null || echo 'NOT FOUND')"
echo "[TAIJI] PYTHONPATH: ${PYTHONPATH}"
echo "[TAIJI] TRAIN_DATA_PATH: ${TRAIN_DATA_PATH}"
echo "[TAIJI] TRAIN_CKPT_PATH: ${TRAIN_CKPT_PATH}"
echo "[TAIJI] TRAIN_LOG_PATH: ${TRAIN_LOG_PATH}"
echo "[TAIJI] TRAIN_TF_EVENTS_PATH: ${TRAIN_TF_EVENTS_PATH}"
echo "================================================================"

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

EXIT_CODE=$?
echo "================================================================"
echo "[TAIJI] python3 exited with code: ${EXIT_CODE}"
echo "================================================================"

exit ${EXIT_CODE}
