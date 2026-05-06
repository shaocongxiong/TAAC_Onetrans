# OneTrans_Pytorch 适配 onetrans_v1 比赛格式实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 OneTrans_Pytorch 的核心模型架构和数据处理逻辑适配到 onetrans_v1 的比赛文件格式中，保持 TAIJI 平台兼容性。

**Architecture:** 模块化迁移方案 - 保留 onetrans_v1 的数据读取、训练循环、推理脚本，替换骨干网络实现，新增特征工程模块。

**Tech Stack:** PyTorch, PyArrow, numpy, sklearn, argparse, logging

---

## Task 1: 创建特征工程模块

**Files:**
- Create: `onetrans_v1/feature_engineering.py`

**Step 1: 创建 feature_engineering.py**

```python
"""Feature engineering module adapted from OneTrans_Pytorch.

Implements scalar/array/sequence feature processing:
- safe_float: Numerical safety conversion
- squash_numeric: Log compression
- summarize_array: 6 statistical summaries
- sanitize_sequence: Sequence cleaning
"""

import hashlib
import math
from typing import Any, Iterable

ARRAY_STATS = ("mean", "std", "min", "max", "last", "length")


def safe_float(value: Any) -> float:
    """Safely convert any value to float.
    
    - None/NaN/Inf → 0.0
    - Bool → float(value)
    - String → float(value) or SHA1 hash normalized to [0, 1]
    """
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
            return int(digest[:12], 16) / float(16**12)
    return 0.0


def squash_numeric(value: float) -> float:
    """Log compression: copysign(log1p(abs(x)), x).
    
    Preserves sign while compressing extreme values.
    """
    if value == 0.0:
        return 0.0
    return math.copysign(math.log1p(abs(value)), value)


def scalar_feature(value: Any) -> float:
    """Process scalar feature: safe_float → squash_numeric."""
    return squash_numeric(safe_float(value))


def sanitize_sequence(values: Any) -> list[float]:
    """Clean sequence values: [squash_numeric(safe_float(v)) for v in values]."""
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, dict)):
        return []
    return [squash_numeric(safe_float(v)) for v in values]


def summarize_array(values: Any) -> list[float]:
    """Compute 6 statistical summaries for array features.
    
    Returns: [mean, std, min, max, last, log_length]
    """
    arr = sanitize_sequence(values)
    if not arr:
        return [0.0] * len(ARRAY_STATS)
    
    mean = sum(arr) / len(arr)
    variance = sum((v - mean) ** 2 for v in arr) / len(arr)
    return [
        mean,
        math.sqrt(variance),
        min(arr),
        max(arr),
        arr[-1],
        math.log1p(len(arr)),
    ]


def build_feature_vector(
    scalar_values: list[Any],
    array_values: list[Any],
    seq_values: list[Any],
    seq_len: int,
) -> tuple[list[float], list[list[float]]]:
    """Build feature vector from raw values.
    
    Args:
        scalar_values: List of scalar feature values
        array_values: List of array feature values
        seq_values: List of sequence feature values
        seq_len: Maximum sequence length
    
    Returns:
        (non_seq_vec, seq_matrix)
        - non_seq_vec: scalar features + array summaries
        - seq_matrix: [seq_len, num_seq_channels]
    """
    # Non-sequence features
    non_seq = [scalar_feature(v) for v in scalar_values]
    for arr in array_values:
        non_seq.extend(summarize_array(arr))
    
    # Sequence features
    seq_channels = [sanitize_sequence(v) for v in seq_values]
    max_len = min(seq_len, max((len(ch) for ch in seq_channels), default=0))
    max_len = max(max_len, 1)
    
    seq_matrix = [[0.0] * len(seq_channels) for _ in range(max_len)]
    for ch_idx, channel in enumerate(seq_channels):
        for step_idx, value in enumerate(channel[:max_len]):
            seq_matrix[step_idx][ch_idx] = value
    
    return non_seq, seq_matrix
```

**Step 2: 验证模块导入**

Run: `cd onetrans_v1 && python -c "from feature_engineering import safe_float, squash_numeric, summarize_array; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
cd onetrans_v1
git add feature_engineering.py
git commit -m "feat: add feature_engineering module from OneTrans_Pytorch"
```

---

## Task 2: 替换骨干网络实现 (RMSNorm, CMA, OneTransBlock)

**Files:**
- Modify: `onetrans_v1/model.py`

**Step 1: 替换 RMSNorm 为 FP32 稳定版**

Locate: `model.py:45-55`

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (FP32 stable version)."""
    
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_fp32 * rms).to(input_dtype) * self.weight
```

**Step 2: 替换 CausalMaskAttention 使用 SDPA**

Locate: `model.py:71-186`

```python
class CausalMaskAttention(nn.Module):
    def __init__(
        self,
        ns_len: int,
        d_model: int = 128,
        num_heads: int = 4,
        if_mask: bool = True,
        mask_type: str = "paper_causal",
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if mask_type not in VALID_MASK_TYPES:
            raise ValueError(f"Unsupported mask_type: {mask_type}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.ns_len = ns_len
        self.if_mask = if_mask
        self.mask_type = mask_type
        self.dense = nn.Linear(d_model, d_model)
        self.kqv_list = nn.ModuleList(
            [nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)]) for _ in range(ns_len + 1)]
        )

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def create_attention_mask(self, query_len: int, key_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.mask_type == "paper_causal":
            row_idx = torch.arange(query_len, device=device).unsqueeze(1)
            col_idx = torch.arange(key_len, device=device).unsqueeze(0)
            q_abs = row_idx + (key_len - query_len)
            allowed = col_idx <= q_abs
            mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
            return mask.masked_fill(~allowed, torch.finfo(dtype).min)

        row_idx = torch.arange(query_len, device=device).unsqueeze(1)
        col_idx = torch.arange(key_len, device=device).unsqueeze(0)
        origin_allowed = (col_idx - row_idx) <= (self.ns_len - 1)
        if self.mask_type == "origin":
            return origin_allowed.to(dtype=dtype) + 1e-9
        if self.mask_type == "hard_mask":
            mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
            return mask.masked_fill(~origin_allowed, torch.finfo(dtype).min)

        ns_query_rows = row_idx < self.ns_len
        strict_causal_allowed = col_idx <= row_idx
        if self.mask_type == "bimask_soft":
            mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
            seq_allowed = (~ns_query_rows) & strict_causal_allowed
            return mask + seq_allowed.to(dtype=dtype)

        allowed = ns_query_rows | strict_causal_allowed
        mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
        return mask.masked_fill(~allowed, torch.finfo(dtype).min)

    def _cal_kqv(self, x: torch.Tensor, group_idx: int, proj_idx: int) -> torch.Tensor:
        return self.kqv_list[group_idx][proj_idx](x)

    def _project_one(self, x: torch.Tensor, proj_idx: int) -> torch.Tensor:
        seq_len = x.size(1) - self.ns_len
        shared_group_idx = self.ns_len
        res = []
        if seq_len > 0:
            res.append(self._cal_kqv(x[:, :seq_len, :], shared_group_idx, proj_idx))
        for i in range(self.ns_len):
            start = seq_len + i
            res.append(self._cal_kqv(x[:, start : start + 1, :], i, proj_idx))
        return torch.cat(res, dim=1)

    def cal_mix_param_kqv(self, x: tuple) -> tuple:
        return self._project_one(x[0], 0), self._project_one(x[1], 1), self._project_one(x[2], 2)

    def forward(self, x: tuple) -> torch.Tensor:
        seq_len_k = x[0].size(1)
        seq_len_q = x[1].size(1)

        k, q, v = self.cal_mix_param_kqv(x)
        k = self.split_heads(k)
        q = self.split_heads(q)
        v = self.split_heads(v)

        attention_mask = None
        if self.if_mask:
            attention_mask = self.create_attention_mask(
                seq_len_q, seq_len_k, device=q.device, dtype=q.dtype
            )
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        )
        output = output.transpose(1, 2).contiguous()
        output = output.view(output.size(0), -1, self.d_model)
        return self.dense(output)
```

**Step 3: 替换 OneTransBlock 使用简洁参数投影**

Locate: `model.py:189-254`

```python
class OneTransBlock(nn.Module):
    def __init__(
        self,
        ns_len: int,
        d_model: int,
        num_heads: int = 4,
        ffn_units: tuple = (256, 128),
        pyramid_stack_len: Optional[int] = None,
        mask_type: str = "paper_causal",
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.ns_len = ns_len
        self.d_model = d_model
        self.pyramid_stack_len = pyramid_stack_len
        self.use_checkpoint = use_checkpoint
        self.rms_0 = RMSNorm(d_model)
        self.rms_1 = RMSNorm(d_model)
        self.cma = CausalMaskAttention(
            ns_len=ns_len,
            d_model=d_model,
            num_heads=num_heads,
            mask_type=mask_type,
        )
        self.ffn_list = nn.ModuleList(
            [FFNLayer(input_dim=d_model, unit_1=ffn_units[0], unit_2=ffn_units[1]) for _ in range(ns_len + 1)]
        )

    def cal_mix_param_ffn(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        seq_len = x.size(1) - self.ns_len
        if seq_len > 0:
            res.append(self.ffn_list[self.ns_len](x[:, :seq_len, :]))
        for i in range(self.ns_len):
            start = seq_len + i
            res.append(self.ffn_list[i](x[:, start : start + 1, :]))
        return torch.cat(res, dim=1)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rms_0(x)
        k_x, q_x, v_x = x, x, x
        if self.pyramid_stack_len is not None and self.pyramid_stack_len >= self.ns_len:
            q_x = x[:, -self.pyramid_stack_len :, :]
        origin_x = q_x

        x = self.cma((k_x, q_x, v_x))
        x = origin_x + x
        origin_x = x

        x = self.rms_1(x)
        x = self.cal_mix_param_ffn(x)
        x = origin_x + x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)
```

**Step 4: 替换 linear_pyramid_schedule**

Locate: `model.py:27-42`

```python
def linear_pyramid_schedule(
    total_tokens: int,
    ns_len: int,
    num_layers: int,
    align_to: int = 32,
) -> List[int]:
    """Calculate pyramid compression schedule with alignment.
    
    Returns list of stack lengths for each compression step.
    """
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if total_tokens < ns_len:
        raise ValueError("total_tokens must be >= ns_len")
    if align_to <= 0:
        raise ValueError("align_to must be positive")
    
    if num_layers == 1:
        return [ns_len]
    
    schedule = [total_tokens]
    for layer_idx in range(1, num_layers - 1):
        raw = total_tokens + (ns_len - total_tokens) * layer_idx / (num_layers - 1)
        target_len = int(round(raw))
        if align_to > 1 and total_tokens > align_to:
            target_len = int(round(target_len / align_to) * align_to)
        target_len = max(ns_len, min(schedule[-1], target_len))
        schedule.append(target_len)
    schedule.append(ns_len)
    return schedule
```

**Step 5: 验证语法**

Run: `cd onetrans_v1 && python -m py_compile model.py`
Expected: No output (success)

**Step 6: Commit**

```bash
cd onetrans_v1
git add model.py
git commit -m "refactor: replace backbone with OneTrans_Pytorch implementation (RMSNorm, CMA+SDPA, pyramid)"
```

---

## Task 3: 更新 OneTransPCVR 支持金字塔压缩

**Files:**
- Modify: `onetrans_v1/model.py:447-724` (OneTransPCVR class)

**Step 1: 更新 __init__ 添加金字塔参数**

```python
class OneTransPCVR(nn.Module):
    def __init__(
        self,
        # Data schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: Dict[str, List[int]],
        # OneTrans hyperparameters
        d_model: int = 128,
        emb_dim: int = 64,
        ns_len: int = 4,
        seq_len: int = 64,
        num_heads: int = 4,
        ffn_hidden: int = 256,
        multi_num: int = 4,
        mask_type: str = "paper_causal",
        num_pyramid_layers: int = 6,
        pyramid_align: int = 32,
        use_checkpoint: bool = False,
        # Optional
        num_time_buckets: int = 0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        dropout_rate: float = 0.01,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.ns_len = ns_len
        self.seq_len = seq_len
        self.num_time_buckets = num_time_buckets
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.seq_domains = sorted(seq_vocab_sizes.keys())

        # Compute input dimensions
        user_int_dim = sum(length for _, _, length in user_int_feature_specs)
        item_int_dim = sum(length for _, _, length in item_int_feature_specs)

        # NS tokenizer
        self.ns_tokenizer = SimpleNSTokenizer(
            user_int_dim=user_int_dim,
            item_int_dim=item_int_dim,
            user_dense_dim=user_dense_dim,
            item_dense_dim=item_dense_dim,
            ns_len=ns_len,
            d_model=d_model,
        )

        # Learnable SEP token
        self.sep_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Sequence tokenizers (per-domain)
        self._seq_tokenizers = nn.ModuleDict()
        for domain in self.seq_domains:
            self._seq_tokenizers[domain] = SeqDomainTokenizer(
                vocab_sizes=seq_vocab_sizes[domain],
                d_model=d_model,
                emb_dim=emb_dim,
                num_time_buckets=num_time_buckets,
                emb_skip_threshold=emb_skip_threshold,
                seq_id_threshold=seq_id_threshold,
            )

        # Base block (no compression)
        self.base_block = MultiOneTransBlock(
            ns_len=ns_len,
            d_model=d_model,
            num_heads=num_heads,
            ffn_units=(ffn_hidden, d_model),
            n=multi_num,
            mask_type=mask_type,
            use_checkpoint=use_checkpoint,
        )

        # Pyramid stack blocks (dynamic schedule)
        total_tokens = ns_len + seq_len * len(self.seq_domains) + 1  # +1 for SEP
        pyramid_schedule = linear_pyramid_schedule(
            total_tokens=total_tokens,
            ns_len=ns_len,
            num_layers=num_pyramid_layers,
            align_to=pyramid_align,
        )
        self.pyramid_schedule = pyramid_schedule
        
        self.stack_blocks = nn.ModuleList(
            [
                MultiOneTransBlock(
                    ns_len=ns_len,
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_units=(ffn_hidden, d_model),
                    n=multi_num,
                    pyramid_stack_len=target_len,
                    mask_type=mask_type,
                    use_checkpoint=use_checkpoint,
                )
                for target_len in pyramid_schedule
            ]
        )

        # Binary classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, 1),
        )

        self.dropout = nn.Dropout(dropout_rate)

        self._init_params()

        logging.info(
            f"OneTransPCVR: d_model={d_model}, emb_dim={emb_dim}, "
            f"ns_len={ns_len}, seq_len={seq_len}, num_heads={num_heads}, "
            f"multi_num={multi_num}, mask_type={mask_type}, "
            f"num_pyramid_layers={num_pyramid_layers}, pyramid_align={pyramid_align}, "
            f"seq_domains={self.seq_domains}, use_checkpoint={use_checkpoint}"
        )
```

**Step 2: 更新 forward 方法**

```python
def forward(self, inputs: ModelInput) -> torch.Tensor:
    # 1. NS tokens
    ns_tokens = self.ns_tokenizer(
        inputs.user_int_feats.float(),
        inputs.item_int_feats.float(),
        inputs.user_dense_feats,
        inputs.item_dense_feats,
    )  # (B, ns_len, d_model)

    # 2. Embed sequence domains
    seq_tokens_list = []
    for domain in self.seq_domains:
        domain_tokens, _ = self._embed_seq_domain(
            domain,
            inputs.seq_data[domain],
            inputs.seq_lens[domain],
            inputs.seq_time_buckets.get(domain),
        )
        seq_tokens_list.append(domain_tokens)

    seq_tokens = torch.cat(seq_tokens_list, dim=1)  # (B, num_domains * seq_len, d_model)

    # 3. Dropout
    ns_tokens = self.dropout(ns_tokens)
    seq_tokens = self.dropout(seq_tokens)

    # 4. Backbone: [seq_tokens, sep_token, ns_tokens]
    batch_size = seq_tokens.size(0)
    sep_tokens = self.sep_token.expand(batch_size, -1, -1)
    x = torch.cat([seq_tokens, sep_tokens, ns_tokens], dim=1)
    
    x = self.base_block(x)
    for block in self.stack_blocks:
        x = block(x)

    # 5. Mean pooling + head
    pooled = x.mean(dim=1)
    logits = self.head(pooled)  # (B, 1)
    return logits
```

**Step 3: 更新 predict 方法 (同步 forward 逻辑)**

```python
def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
    self.eval()
    with torch.no_grad():
        ns_tokens = self.ns_tokenizer(
            inputs.user_int_feats.float(),
            inputs.item_int_feats.float(),
            inputs.user_dense_feats,
            inputs.item_dense_feats,
        )

        seq_tokens_list = []
        for domain in self.seq_domains:
            domain_tokens, _ = self._embed_seq_domain(
                domain,
                inputs.seq_data[domain],
                inputs.seq_lens[domain],
                inputs.seq_time_buckets.get(domain),
            )
            seq_tokens_list.append(domain_tokens)

        seq_tokens = torch.cat(seq_tokens_list, dim=1)
        batch_size = seq_tokens.size(0)
        sep_tokens = self.sep_token.expand(batch_size, -1, -1)
        x = torch.cat([seq_tokens, sep_tokens, ns_tokens], dim=1)
        
        x = self.base_block(x)
        for block in self.stack_blocks:
            x = block(x)

        pooled = x.mean(dim=1)
        logits = self.head(pooled)
        return logits, pooled
```

**Step 4: 验证语法**

Run: `cd onetrans_v1 && python -m py_compile model.py`
Expected: No output

**Step 5: Commit**

```bash
cd onetrans_v1
git add model.py
git commit -m "feat: add pyramid compression support to OneTransPCVR"
```

---

## Task 4: 在 dataset.py 中集成特征工程

**Files:**
- Modify: `onetrans_v1/dataset.py`

**Step 1: 添加 feature_engineering 导入**

Locate: `dataset.py:1-20` (imports section)

```python
import os
import logging
import random
import json
import gc
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

from feature_engineering import scalar_feature, summarize_array, sanitize_sequence
```

**Step 2: 在 _convert_batch 中应用特征工程 (标量特征)**

Locate: `dataset.py:413-422` (user_int processing)

```python
for ci, dim, offset, vs in self._user_int_plan:
    col = batch.column(ci)
    if dim == 1:
        arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.float64)
        # Apply feature engineering
        arr = np.array([scalar_feature(v) for v in arr], dtype=np.float32)
        if vs > 0:
            self._record_oob('user_int', ci, arr.astype(np.int64), vs)
        else:
            arr[:] = 0
        user_int[:, offset] = arr
    else:
        padded, _ = self._pad_varlen_int_column(col, dim, B)
        if vs > 0:
            self._record_oob('user_int', ci, padded, vs)
        else:
            padded[:] = 0
        user_int[:, offset:offset + dim] = padded
```

**Step 3: 在 _convert_batch 中应用特征工程 (数组特征)**

Locate: `dataset.py:451-456` (user_dense processing)

```python
user_dense = self._buf_user_dense[:B]
user_dense[:] = 0
for ci, dim, offset in self._user_dense_plan:
    col = batch.column(ci)
    padded = self._pad_varlen_float_column(col, dim, B)
    # Apply summarize_array to each row
    for i in range(B):
        stats = summarize_array(padded[i].tolist())
        user_dense[i, offset:offset + len(stats)] = stats
```

**Step 4: 验证语法**

Run: `cd onetrans_v1 && python -m py_compile dataset.py`
Expected: No output

**Step 5: Commit**

```bash
cd onetrans_v1
git add dataset.py
git commit -m "feat: integrate feature_engineering into dataset processing pipeline"
```

---

## Task 5: 更新 train.py 添加新参数

**Files:**
- Modify: `onetrans_v1/train.py`

**Step 1: 添加新参数到 parse_args**

Locate: `train.py:95-136` (after existing model hyperparameters)

```python
    # Pyramid compression.
    parser.add_argument('--num-pyramid-layers', type=int, default=6,
                        help='Number of pyramid compression layers')
    parser.add_argument('--pyramid-align', type=int, default=32,
                        help='Pyramid schedule alignment granularity')

    # Activation checkpointing.
    parser.add_argument('--use-checkpoint', action='store_true',
                        help='Enable activation checkpointing during training')
```

**Step 2: 更新 model_args 传递新参数**

Locate: `train.py:207-225` (model_args dict)

```python
    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "ns_len": args.ns_len,
        "seq_len": args.seq_len,
        "num_heads": args.num_heads,
        "ffn_hidden": args.ffn_hidden,
        "multi_num": args.multi_num,
        "mask_type": args.mask_type,
        "num_pyramid_layers": args.num_pyramid_layers,
        "pyramid_align": args.pyramid_align,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "dropout_rate": args.dropout_rate,
        "use_checkpoint": args.use_checkpoint,
    }
```

**Step 3: 验证语法**

Run: `cd onetrans_v1 && python -m py_compile train.py`
Expected: No output

**Step 4: Commit**

```bash
cd onetrans_v1
git add train.py
git commit -m "feat: add pyramid compression and checkpoint parameters to train.py"
```

---

## Task 6: 更新 infer.py 同步新参数

**Files:**
- Modify: `onetrans_v1/infer.py`

**Step 1: 更新 _FALLBACK_MODEL_CFG 添加新参数**

Locate: `infer.py:32-47`

```python
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
```

**Step 2: 验证语法**

Run: `cd onetrans_v1 && python -m py_compile infer.py`
Expected: No output

**Step 3: Commit**

```bash
cd onetrans_v1
git add infer.py
git commit -m "feat: sync new model parameters in infer.py fallback config"
```

---

## Task 7: 更新 run.sh 添加新参数示例

**Files:**
- Modify: `onetrans_v1/run.sh`

**Step 1: 更新 run.sh 添加新参数**

```bash
#!/bin/bash

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
    --ns_len 4 \
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
```

**Step 2: Commit**

```bash
cd onetrans_v1
git add run.sh
git commit -m "docs: update run.sh with new pyramid compression parameters"
```

---

## Task 8: 端到端验证

**Files:**
- All modified files

**Step 1: 验证所有模块导入**

Run: `cd onetrans_v1 && python -c "from model import OneTransPCVR, ModelInput; from dataset import PCVRParquetDataset; from feature_engineering import safe_float, squash_numeric, summarize_array; print('All imports OK')"`
Expected: `All imports OK`

**Step 2: 验证模型构建 (无数据)**

Run: `cd onetrans_v1 && python -c "
from model import OneTransPCVR
import torch

# Test model creation
model = OneTransPCVR(
    user_int_feature_specs=[(1000, 0, 10)],
    item_int_feature_specs=[(500, 0, 5)],
    user_dense_dim=60,
    item_dense_dim=0,
    seq_vocab_sizes={'seq_a': [100, 200, 300]},
    d_model=128,
    emb_dim=64,
    ns_len=4,
    seq_len=16,
    num_heads=4,
    ffn_hidden=256,
    multi_num=2,
    mask_type='paper_causal',
    num_pyramid_layers=3,
    pyramid_align=32,
)

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f'Model created successfully: {total:,} parameters')
print(f'Pyramid schedule: {model.pyramid_schedule}')
"`
Expected: Model creation success with parameter count and pyramid schedule

**Step 3: 验证前向传播 (随机数据)**

Run: `cd onetrans_v1 && python -c "
from model import OneTransPCVR, ModelInput
import torch

model = OneTransPCVR(
    user_int_feature_specs=[(1000, 0, 10)],
    item_int_feature_specs=[(500, 0, 5)],
    user_dense_dim=60,
    item_dense_dim=0,
    seq_vocab_sizes={'seq_a': [100, 200, 300]},
    d_model=128,
    emb_dim=64,
    ns_len=4,
    seq_len=8,
    num_heads=4,
    ffn_hidden=256,
    multi_num=2,
    mask_type='paper_causal',
    num_pyramid_layers=3,
    pyramid_align=32,
)
model.eval()

# Create dummy input
B = 2
inputs = ModelInput(
    user_int_feats=torch.randint(0, 1000, (B, 10)),
    item_int_feats=torch.randint(0, 500, (B, 5)),
    user_dense_feats=torch.randn(B, 60),
    item_dense_feats=torch.zeros(B, 0),
    seq_data={'seq_a': torch.randint(0, 100, (B, 3, 8))},
    seq_lens={'seq_a': torch.tensor([5, 7])},
    seq_time_buckets={'seq_a': torch.zeros(B, 8, dtype=torch.long)},
)

# Forward pass
with torch.no_grad():
    logits = model(inputs)
    print(f'Input shape: user_int={inputs.user_int_feats.shape}, seq={inputs.seq_data[\"seq_a\"].shape}')
    print(f'Output logits shape: {logits.shape}')
    print(f'Logits: {logits.squeeze().tolist()}')
"`
Expected: Forward pass success with correct output shape `(B, 1)`

**Step 4: 验证特征工程模块**

Run: `cd onetrans_v1 && python -c "
from feature_engineering import safe_float, squash_numeric, summarize_array, scalar_feature

# Test safe_float
assert safe_float(None) == 0.0
assert safe_float(float('nan')) == 0.0
assert safe_float(float('inf')) == 0.0
assert safe_float(42) == 42.0
assert 0 <= safe_float('hello') <= 1  # SHA1 hash

# Test squash_numeric
assert squash_numeric(0.0) == 0.0
assert abs(squash_numeric(1.0) - 0.693) < 0.001
assert abs(squash_numeric(-1.0) - (-0.693)) < 0.001

# Test summarize_array
stats = summarize_array([1, 2, 3, 4, 5])
assert len(stats) == 6
assert stats[0] == 3.0  # mean

# Test scalar_feature
assert scalar_feature(100) == squash_numeric(100.0)

print('All feature_engineering tests passed')
"`
Expected: `All feature_engineering tests passed`

**Step 5: 最终 Commit**

```bash
cd onetrans_v1
git add -A
git commit -m "test: verify end-to-end model and feature engineering"
```

---

## 验证清单

完成所有任务后，运行以下验证：

```bash
cd onetrans_v1

# 1. 语法检查
python -m py_compile model.py dataset.py train.py infer.py feature_engineering.py

# 2. 导入检查
python -c "from model import OneTransPCVR; from dataset import PCVRParquetDataset; from feature_engineering import *; print('OK')"

# 3. 模型构建和前向传播
python -c "from model import OneTransPCVR, ModelInput; import torch; ... (见 Task 8 Step 3)"

# 4. 特征工程测试
python -c "from feature_engineering import *; ... (见 Task 8 Step 4)"
```

---

## 后续步骤

1. 使用真实数据集进行小规模训练验证 (`--max_rows 100 --epochs 1`)
2. 验证推理脚本输出格式 (`predictions.json`)
3. 提交到比赛平台测试
