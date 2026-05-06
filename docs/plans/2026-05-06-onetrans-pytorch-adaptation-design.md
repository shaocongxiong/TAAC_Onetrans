# OneTrans_Pytorch 适配 onetrans_v1 比赛格式设计文档

**日期**: 2026-05-06  
**状态**: 已批准  
**作者**: AI Assistant

---

## 一、 目标

将 OneTrans_Pytorch 的核心设计（模型架构、数据处理逻辑）适配到 onetrans_v1 的比赛文件格式中，确保：
1. 完全兼容 TAIJI 评估平台（Parquet + schema.json + infer.py）
2. 遵守 OneTrans_Pytorch 的模型设计和数据处理逻辑
3. 保持 onetrans_v1 的训练流程、日志格式和资源监控

---

## 二、 架构决策

### 2.1 保留 onetrans_v1 的组件

| 组件 | 原因 |
|---|---|
| `dataset.py` | Parquet 流式读取、buffer shuffle、Row Group 划分 |
| `trainer.py` | 双优化器、GPU 内存日志、EarlyStopping、tqdm 进度条 |
| `infer.py` | TAIJI 平台适配、predictions.json 输出 |
| `utils.py` | 日志格式、EarlyStopping、Focal Loss |
| 环境变量 | `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH`, `TRAIN_LOG_PATH` |

### 2.2 替换为 OneTrans_Pytorch 的组件

| 组件 | 替换内容 |
|---|---|
| `model.py` 骨干网络 | RMSNorm (FP32), CausalMaskAttention (SDPA), OneTransBlock (金字塔压缩) |
| 特征工程 | 新增 `feature_engineering.py` (safe_float, squash_numeric, summarize_array) |
| 序列 Tokenizer | 混合模式：Linear 投影 + 可选 Embedding |
| 金字塔压缩 | 动态调度 `linear_pyramid_schedule` |

---

## 三、 核心改动点

### 3.1 `model.py` 改动

#### 保留部分
- `ModelInput` NamedTuple
- `OneTransPCVR` 主类接口
- `get_sparse_params()` / `get_dense_params()`
- `reinit_high_cardinality_params()`

#### 替换部分

**RMSNorm**:
```python
# 修改前 (简单实现)
norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
return self.weight * (x / norm)

# 修改后 (FP32 稳定版)
input_dtype = x.dtype
x_fp32 = x.float()
rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
return (x_fp32 * rms).to(input_dtype) * self.weight
```

**CausalMaskAttention**:
```python
# 修改前 (手动 softmax)
matmul_qk = torch.matmul(q, k.transpose(-2, -1))
scaled_attention_logits = matmul_qk / (self.depth ** 0.5)
attention_weights = torch.softmax(scaled_attention_logits + mask, dim=-1)
output = torch.matmul(attention_weights, v)

# 修改后 (PyTorch SDPA)
output = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
)
```

**OneTransBlock._project_one**:
```python
# 修改前 (显式处理 SEP)
seq_len = total_len - self.ns_len - 1  # Subtract ns_tokens and sep_token

# 修改后 (简洁实现)
seq_len = x.size(1) - self.ns_len
# 前 seq_len 个 token 共享 group[ns_len]
# 后 ns_len 个 token 各自使用 group[i]
```

**linear_pyramid_schedule**:
```python
# 修改前 (简单线性)
steps = seq_len - ns_len
schedule = [seq_len - 1 - i for i in range(steps)]

# 修改后 (动态调度 + 对齐)
schedule = [total_tokens]
for layer_idx in range(1, num_layers - 1):
    raw = total_tokens + (ns_len - total_tokens) * layer_idx / (num_layers - 1)
    target_len = int(round(raw))
    if align_to > 1 and total_tokens > align_to:
        target_len = int(round(target_len / align_to) * align_to)
    schedule.append(max(ns_len, min(schedule[-1], target_len)))
schedule.append(ns_len)
```

### 3.2 新增 `feature_engineering.py`

```python
ARRAY_STATS = ("mean", "std", "min", "max", "last", "length")

def safe_float(value) -> float:
    """数值安全转换：None/NaN/Inf → 0.0, 字符串 → SHA1 哈希"""

def squash_numeric(value: float) -> float:
    """Log 压缩：copysign(log1p(abs(x)), x)"""

def summarize_array(values) -> list[float]:
    """数组统计摘要：[mean, std, min, max, last, log_length]"""

def sanitize_sequence(values) -> list[float]:
    """序列清洗：[squash_numeric(safe_float(v)) for v in values]"""

def build_feature_vector(row, schema) -> tuple[list[float], list[list[float]]]:
    """构建特征向量：(non_seq_vec, seq_matrix)"""
```

### 3.3 `dataset.py` 微调

在 `_convert_batch()` 中添加特征工程处理：

```python
# 修改前
user_int[:, offset] = arr  # 直接使用原始值

# 修改后
from feature_engineering import squash_numeric, safe_float
processed = [squash_numeric(safe_float(v)) for v in arr]
user_int[:, offset] = processed
```

### 3.4 `train.py` 新增参数

```python
parser.add_argument('--use-linear-tokenizer', action='store_true',
                    help='Use Linear projection for sequence tokenizer')
parser.add_argument('--num-pyramid-layers', type=int, default=6,
                    help='Number of pyramid compression layers')
parser.add_argument('--pyramid-align', type=int, default=32,
                    help='Pyramid schedule alignment granularity')
parser.add_argument('--use-sdpa', action='store_true', default=True,
                    help='Use PyTorch SDPA for attention')
```

---

## 四、 数据流

### 4.1 训练数据流

```
Parquet 文件
  │
  ▼ dataset.py (PCVRParquetDataset)
  │   - Parquet 流式读取
  │   - buffer shuffle
  │   - Row Group 划分
  │
  ▼ _convert_batch() + feature_engineering.py
  │   - safe_float → squash_numeric (标量)
  │   - summarize_array (数组 → 6 统计量)
  │   - sanitize_sequence (序列)
  │
  ▼ ModelInput
  │   - user_int_feats: [B, user_int_dim]
  │   - item_int_feats: [B, item_int_dim]
  │   - user_dense_feats: [B, user_dense_dim]
  │   - seq_data[domain]: [B, n_features, seq_len]
  │
  ▼ OneTransPCVR.forward()
  │   - SimpleNSTokenizer: cat([user_int, item_int, user_dense]) → Linear → [B, ns_len, d_model]
  │   - HybridSeqTokenizer: Linear 投影 或 Embedding → [B, seq_len, d_model]
  │   - 拼接: [seq_tokens, sep_token, ns_tokens]
  │   - base_block: MultiOneTransBlock (无压缩)
  │   - stack_blocks: MultiOneTransBlock x num_pyramid_layers (金字塔压缩)
  │   - Mean Pooling → Head → [B, 1] logits
  │
  ▼ trainer.py
  │   - BCEWithLogits / Focal Loss
  │   - 双优化器: Adagrad (sparse) + AdamW (dense)
  │   - GPU 内存日志
  │   - EarlyStopping
  │
  ▼ Checkpoint
      - global_step 目录
      - model.pt + schema.json + train_config.json
```

### 4.2 推理数据流

```
Parquet 文件
  │
  ▼ infer.py
  │   - 加载 train_config.json
  │   - 重建模型
  │   - 加载 checkpoint
  │
  ▼ PCVRParquetDataset (is_training=False)
  │
  ▼ model.predict()
  │   - logits → sigmoid → probs
  │
  ▼ predictions.json
      - {user_id: probability}
```

---

## 五、 兼容性保证

### 5.1 训练脚本兼容性

| 特性 | 保持 |
|---|---|
| 环境变量 | ✅ `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH`, `TRAIN_LOG_PATH` |
| 参数格式 | ✅ 所有原有参数保持不变 |
| 日志格式 | ✅ `logging.info` + GPU 内存监控 |
| Checkpoint | ✅ `global_step` 目录 + `model.pt` |
| 早停 | ✅ `EarlyStopping` 类 |

### 5.2 推理脚本兼容性

| 特性 | 保持 |
|---|---|
| 环境变量 | ✅ `MODEL_OUTPUT_PATH`, `EVAL_DATA_PATH`, `EVAL_RESULT_PATH` |
| 输出格式 | ✅ `predictions.json` |
| 模型加载 | ✅ `train_config.json` + `schema.json` |
| 严格模式 | ✅ `load_state_dict(strict=True)` |

### 5.3 比赛平台兼容性

| 特性 | 保持 |
|---|---|
| 数据格式 | ✅ Parquet + schema.json |
| 数据读取 | ✅ `PCVRParquetDataset` (IterableDataset) |
| Shuffle | ✅ `buffer_batches` |
| 双优化器 | ✅ Adagrad (sparse) + AdamW (dense) |
| Embedding 重建 | ✅ `reinit_high_cardinality_params()` |

---

## 六、 实施步骤

1. 创建 `feature_engineering.py` (特征工程模块)
2. 修改 `model.py` (骨干网络替换)
3. 修改 `dataset.py` (添加特征工程处理)
4. 修改 `train.py` (添加新参数)
5. 更新 `run.sh` (添加新参数示例)
6. 验证训练流程
7. 验证推理流程

---

## 七、 风险与缓解

| 风险 | 缓解措施 |
|---|---|
| SDPA 兼容性问题 | 保留 `--use-sdpa` 开关，可回退到手动实现 |
| 金字塔压缩显存 | 使用 `--activation-checkpoint` 节省显存 |
| 特征工程性能 | 使用 numpy 向量化操作，避免 Python 循环 |
| 双优化器兼容 | 保留 `get_sparse_params()` / `get_dense_params()` 接口 |
