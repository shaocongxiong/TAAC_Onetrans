"""OneTrans inference script for the TAIJI evaluation platform.

Rebuilds the model from ``schema.json`` + ``train_config.json`` found in the
checkpoint directory, loads the saved weights, runs inference on the test
data, and writes ``predictions.json``.

Environment variables:
    MODEL_OUTPUT_PATH  Checkpoint directory (points at the ``global_step``
                       sub-directory containing ``model.pt`` / ``train_config.json``).
    EVAL_DATA_PATH     Test data directory (*.parquet + schema.json).
    EVAL_RESULT_PATH   Directory for the generated ``predictions.json``.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FeatureSchema, PCVRParquetDataset, NUM_TIME_BUCKETS
from model import OneTransPCVR, ModelInput

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# ── Fallback defaults (must match train.py argparse defaults) ──────────────

_FALLBACK_MODEL_CFG = {
    'd_model': 128,
    'emb_dim': 64,
    'ns_len': 4,
    'seq_len': 64,
    'num_heads': 4,
    'ffn_hidden': 256,
    'multi_num': 4,
    'mask_type': 'paper_causal',
    'num_pyramid_layers': 6,
    'pyramid_align': 32,
    'dropout_rate': 0.01,
    'num_time_buckets': NUM_TIME_BUCKETS,
    'emb_skip_threshold': 1000000,
    'seq_id_threshold': 10000,
    'use_checkpoint': False,
}

_FALLBACK_SEQ_MAX_LENS = None  # Will default to seq_len per domain
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 16

_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build ``feature_specs = [(vocab_size, offset, length), ...]``."""
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    """Parse ``'seq_a:256,seq_b:256,...'`` into a dict."""
    seq_max_lens: Dict[str, int] = {}
    for pair in sml_str.split(','):
        k, v = pair.split(':')
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def load_train_config(model_dir: str) -> Dict[str, Any]:
    """Load ``train_config.json`` from the ckpt directory."""
    train_config_path = os.path.join(model_dir, 'train_config.json')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg
    logging.warning(
        f"train_config.json not found in {model_dir}, "
        f"falling back to hardcoded defaults.")
    return {}


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model hyperparameters; missing keys fall back to defaults."""
    cfg: Dict[str, Any] = {}
    for key in _MODEL_CFG_KEYS:
        if key == 'num_time_buckets':
            if 'num_time_buckets' in train_config:
                cfg[key] = train_config['num_time_buckets']
            elif 'use_time_buckets' in train_config:
                cfg[key] = NUM_TIME_BUCKETS if train_config['use_time_buckets'] else 0
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
            continue

        if key in train_config:
            cfg[key] = train_config[key]
        else:
            cfg[key] = _FALLBACK_MODEL_CFG[key]
            logging.warning(
                f"train_config missing '{key}', using fallback = {cfg[key]}")
    return cfg


def build_model(
    dataset: PCVRParquetDataset,
    model_cfg: Dict[str, Any],
    device: str = 'cpu',
) -> OneTransPCVR:
    """Construct ``OneTransPCVR`` from the dataset schema and resolved config."""
    user_int_feature_specs = build_feature_specs(
        dataset.user_int_schema, dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        dataset.item_int_schema, dataset.item_int_vocab_sizes)

    logging.info(f"Building OneTransPCVR with cfg: {model_cfg}")
    model = OneTransPCVR(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        **model_cfg,
    ).to(device)

    return model


def load_model_state_strict(
    model: nn.Module,
    ckpt_path: str,
    device: str,
) -> None:
    """Strictly load ``state_dict``; any mismatch fails fast."""
    state_dict = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        logging.error(
            "Failed to load state_dict in strict mode. The model constructed "
            "by build_model does NOT match the checkpoint. Check that "
            "train_config.json in the ckpt dir matches the training hyperparameters.")
        raise e


def get_ckpt_path() -> Optional[str]:
    """Locate the first ``*.pt`` file inside ``$MODEL_OUTPUT_PATH``."""
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if not ckpt_path:
        return None
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)
    return None


def _batch_to_model_input(
    batch: Dict[str, Any],
    device: str,
) -> ModelInput:
    """Convert a batch dict to ``ModelInput``."""
    device_batch: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            device_batch[k] = v.to(device, non_blocking=True)
        else:
            device_batch[k] = v

    seq_domains = device_batch['_seq_domains']
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f'{domain}_len']
        B, _, L = device_batch[domain].shape
        seq_time_buckets[domain] = device_batch.get(
            f'{domain}_time_bucket',
            torch.zeros(B, L, dtype=torch.long, device=device))

    return ModelInput(
        user_int_feats=device_batch['user_int_feats'],
        item_int_feats=device_batch['item_int_feats'],
        user_dense_feats=device_batch['user_dense_feats'],
        item_dense_feats=device_batch['item_dense_feats'],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
    )


def main() -> None:
    # ── Read environment variables ──
    model_dir = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')

    os.makedirs(result_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Device: {device}")

    # ── Schema: prefer ckpt dir (exact match with training) ──
    schema_path = os.path.join(model_dir, 'schema.json')
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, 'schema.json')
    logging.info(f"Using schema: {schema_path}")

    # ── Load train_config.json ──
    train_config = load_train_config(model_dir)

    # ── Parse seq_max_lens ──
    sml_str = train_config.get('seq_max_lens', None)
    seq_max_lens: Dict[str, int] = {}
    if sml_str:
        seq_max_lens = _parse_seq_max_lens(sml_str)
    if not seq_max_lens:
        # Default to the model's seq_len for all domains
        model_cfg_preview = resolve_model_cfg(train_config)
        default_len = model_cfg_preview.get('seq_len', 64)
        logging.info(f"seq_max_lens not in config, using seq_len={default_len} for all domains")

    # ── Data loading ──
    batch_size = int(train_config.get('batch_size', _FALLBACK_BATCH_SIZE))
    num_workers = int(train_config.get('num_workers', _FALLBACK_NUM_WORKERS))

    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens if seq_max_lens else None,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
        default_seq_len=resolve_model_cfg(train_config).get('seq_len', 64),
    )
    total_test_samples = test_dataset.num_rows
    logging.info(f"Total test samples: {total_test_samples}")

    # ── Build model ──
    model_cfg = resolve_model_cfg(train_config)
    model = build_model(test_dataset, model_cfg=model_cfg, device=device)

    # ── Load weights ──
    ckpt_path = get_ckpt_path()
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No *.pt file found under MODEL_OUTPUT_PATH={model_dir!r}. "
            f"Directory contents: {os.listdir(model_dir) if model_dir and os.path.isdir(model_dir) else 'N/A'}")
    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state_strict(model, ckpt_path, device)
    model.eval()
    logging.info("Model loaded successfully")

    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=torch.cuda.is_available(),
    )

    # ── Inference ──
    all_probs = []
    all_user_ids = []
    logging.info("Starting inference...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            model_input = _batch_to_model_input(batch, device)
            user_ids = batch.get('user_id', [])

            logits, _ = model.predict(model_input)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_user_ids.extend(user_ids)

            if (batch_idx + 1) % 100 == 0:
                logging.info(f"  Processed {(batch_idx + 1) * batch_size} samples")

    logging.info(f"Inference complete: {len(all_probs)} predictions")

    # ── Save predictions.json ──
    predictions = {
        "predictions": dict(zip(all_user_ids, all_probs)),
    }

    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    logging.info(f"Saved {len(all_probs)} predictions to {output_path}")


if __name__ == "__main__":
    main()
