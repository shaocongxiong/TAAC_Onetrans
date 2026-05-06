# Version V1

## 数据处理方式
- **标量特征**: `safe_float()` → `squash_numeric()` (Log 压缩: `copysign(log1p(abs(x)), x)`)
- **数组特征**: `summarize_array()` → 6 个统计量 `[mean, std, min, max, last, log_length]`
- **序列特征**: `sanitize_sequence()` → 多通道对齐 + Padding 到 `seq_len`
- **核心模块**: `feature_engineering.py`

## 核心超参数
| 参数 | 值 | 说明 |
|---|---|---|
| `batch_size` | 256 | 批次大小 |
| `d_model` | 128 | 隐藏维度 |
| `emb_dim` | 64 | 嵌入维度 |
| `ns_len` | 10 | 非序列 token 数量 |
| `seq_len` | 64 | 序列长度 |
| `num_heads` | 4 | 注意力头数 |
| `ffn_hidden` | 256 | FFN 隐藏层维度 |
| `multi_num` | 4 | 每个阶段 block 数量 |
| `mask_type` | paper_causal | 注意力掩码类型 |
| `num_pyramid_layers` | 6 | 金字塔压缩层数 |
| `pyramid_align` | 32 | 金字塔对齐粒度 |
| `dropout_rate` | 0.01 | Dropout 率 |
| `lr` | 1e-4 | 学习率 (AdamW) |
| `sparse_lr` | 0.05 | 稀疏学习率 (Adagrad) |
| `loss_type` | bce | 损失函数 |
| `emb_skip_threshold` | 1000000 | 跳过 embedding 阈值 |
| `seq_id_threshold` | 10000 | 序列 ID 阈值 |
| `use_checkpoint` | True | 激活检查点 |

## 架构特点
- 使用 OneTrans_Pytorch 骨干网络 (RMSNorm FP32, SDPA 注意力)
- 动态金字塔压缩调度 (`linear_pyramid_schedule`)
- 双优化器策略 (Adagrad + AdamW)
- TAIJI 平台适配 (`run.sh` 使用 `PYTHONPATH` 和绝对路径)
