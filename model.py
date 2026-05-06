"""OneTrans model adapted to the official PCVRHyFormer input/output format."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, NamedTuple, Tuple, Optional, Dict


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict        # {domain: tensor [B, S, L]}
    seq_lens: dict        # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}


# ═══════════════════════════════════════════════════════════════════════════════
# OneTrans Backbone (inlined from main_pytorch.py)
# ═══════════════════════════════════════════════════════════════════════════════

VALID_MASK_TYPES = {"origin", "hard_mask", "bimask_soft", "bimask_hard", "paper_causal"}


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


class FFNLayer(nn.Module):
    def __init__(self, input_dim: int, unit_1: int = 256, unit_2: int = 128) -> None:
        super().__init__()
        self.proj_1 = nn.Linear(input_dim, unit_1)
        self.proj_2 = nn.Linear(unit_1, unit_2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.proj_1(x))
        x = self.act(self.proj_2(x))
        return x


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


class MultiOneTransBlock(nn.Module):
    def __init__(
        self,
        ns_len: int = 4,
        d_model: int = 128,
        num_heads: int = 4,
        ffn_units: tuple = (256, 128),
        n: int = 4,
        pyramid_stack_len: Optional[int] = None,
        mask_type: str = "origin",
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.otb_list = nn.ModuleList(
            [
                OneTransBlock(
                    ns_len=ns_len,
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_units=ffn_units,
                    pyramid_stack_len=pyramid_stack_len,
                    mask_type=mask_type,
                    use_checkpoint=use_checkpoint,
                )
                for _ in range(n)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sequential execution instead of stack+mean
        for otb in self.otb_list:
            x = otb(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Tokenizer: per-domain embedding + projection
# ═══════════════════════════════════════════════════════════════════════════════


class SeqDomainTokenizer(nn.Module):
    """Embeds and projects a single sequence domain to d_model.

    Each domain has multiple features (fids), each with its own embedding table.
    Feature embeddings are concatenated then projected to d_model.
    Optional time bucket embedding is added.
    """

    def __init__(
        self,
        vocab_sizes: List[int],
        d_model: int,
        emb_dim: int,
        num_time_buckets: int = 0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.num_time_buckets = num_time_buckets
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold

        # Create embedding tables (skip high-cardinality if configured)
        embs_raw = []
        for vs in vocab_sizes:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs_raw.append(None)
            else:
                embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs_raw if e is not None])
        self._emb_index = []
        real_idx = 0
        for e in embs_raw:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        self.is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
        self.vocab_sizes = vocab_sizes

        # Projection: (num_features * emb_dim) -> d_model
        # Always use full feature count; skipped features just contribute zeros.
        self.proj = nn.Sequential(
            nn.Linear(len(vocab_sizes) * emb_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Time bucket embedding
        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        # Dropout for id features
        self.seq_id_emb_dropout = nn.Dropout(0.02)  # 2 * default dropout_rate

    def forward(
        self,
        seq: torch.Tensor,
        time_bucket_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed and project a sequence domain.

        Args:
            seq: (B, num_features, L) integer feature indices.
            time_bucket_ids: (B, L) time bucket indices.

        Returns:
            (B, L, d_model) token embeddings.
        """
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = self._emb_index[i] if i < len(self._emb_index) else -1
            if real_idx == -1:
                emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = self.embs[real_idx]
                e = emb(seq[:, i, :])  # (B, L, emb_dim)
                if self.is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)

        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, active_features * emb_dim)
        token_emb = F.gelu(self.proj(cat_emb))  # (B, L, d_model)

        if self.num_time_buckets > 0 and time_bucket_ids is not None:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)

        return token_emb


# ═══════════════════════════════════════════════════════════════════════════════
# NS Tokenizer: simple embedding-based approach for OneTrans
# ═══════════════════════════════════════════════════════════════════════════════


class SimpleNSTokenizer(nn.Module):
    """Projects non-sequence features to ns_len tokens.

    Concatenates all non-sequence features (user_int + item_int + user_dense + item_dense)
    then projects to (ns_len, d_model).
    """

    def __init__(
        self,
        user_int_dim: int,
        item_int_dim: int,
        user_dense_dim: int,
        item_dense_dim: int,
        ns_len: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.ns_len = ns_len
        self.d_model = d_model
        total_input_dim = user_int_dim + item_int_dim + user_dense_dim + item_dense_dim
        self.proj = nn.Linear(total_input_dim, ns_len * d_model)

    def forward(
        self,
        user_int: torch.Tensor,
        item_int: torch.Tensor,
        user_dense: torch.Tensor,
        item_dense: torch.Tensor,
    ) -> torch.Tensor:
        """Project non-sequence features to ns_len tokens.

        Args:
            user_int: (B, user_int_dim)
            item_int: (B, item_int_dim)
            user_dense: (B, user_dense_dim)
            item_dense: (B, item_dense_dim)

        Returns:
            (B, ns_len, d_model)
        """
        combined = torch.cat([user_int, item_int, user_dense, item_dense], dim=-1)
        tokens = self.proj(combined).view(-1, self.ns_len, self.d_model)
        return tokens


# ═══════════════════════════════════════════════════════════════════════════════
# OneTransPCVR: Main Model
# ═══════════════════════════════════════════════════════════════════════════════


class OneTransPCVR(nn.Module):
    """OneTrans model adapted to the official PCVRHyFormer input format.

    Accepts ModelInput (user/item int/dense features + multi-domain sequences),
    converts to OneTrans internal format, processes through OneTrans blocks,
    and outputs binary classification logits.
    """

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

    def _init_params(self) -> None:
        for tokenizer in self._seq_tokenizers.values():
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

            if self.num_time_buckets > 0:
                nn.init.xavier_normal_(tokenizer.time_embedding.weight.data)
                tokenizer.time_embedding.weight.data[0, :] = 0

    def get_sparse_params(self) -> List[nn.Parameter]:
        """Returns all embedding table parameters (optimized with Adagrad)."""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """Returns all non-embedding parameters (optimized with AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def reinit_high_cardinality_params(self, cardinality_threshold: int = 10000) -> set:
        """Reinitializes only high-cardinality embeddings."""
        reinit_ptrs = set()
        for tokenizer in self._seq_tokenizers.values():
            for i, vs in enumerate(tokenizer.vocab_sizes):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
        return reinit_ptrs

    def _embed_seq_domain(
        self,
        domain: str,
        seq_data: torch.Tensor,
        seq_lens: torch.Tensor,
        time_buckets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed a single sequence domain and pad/truncate to seq_len.

        Returns:
            tokens: (B, seq_len, d_model)
            padding_mask: (B, seq_len), True = padding
        """
        tokenizer = self._seq_tokenizers[domain]
        B, S, L = seq_data.shape

        # Truncate to seq_len BEFORE embedding to save memory
        if L > self.seq_len:
            seq_data = seq_data[:, :, :self.seq_len]
            L = self.seq_len
            if time_buckets is not None:
                time_buckets = time_buckets[:, :self.seq_len]

        # Embed domain
        tokens = tokenizer(seq_data, time_buckets)  # (B, L, d_model)

        # Pad to seq_len if needed
        if L < self.seq_len:
            pad = torch.zeros(B, self.seq_len - L, self.d_model, device=tokens.device)
            tokens = torch.cat([tokens, pad], dim=1)

        # Build padding mask
        effective_len = torch.clamp(seq_lens, max=self.seq_len)
        idx = torch.arange(self.seq_len, device=seq_lens.device).unsqueeze(0)
        padding_mask = idx >= effective_len.unsqueeze(1)

        return tokens, padding_mask

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """Forward pass accepting official ModelInput format.

        1. Embed non-sequence features -> ns_len tokens
        2. Embed each sequence domain, concatenate along sequence dimension
        3. Run through OneTrans blocks
        4. Binary classification head
        """
        # 1. NS tokens from non-sequence features
        ns_tokens = self.ns_tokenizer(
            inputs.user_int_feats.float(),
            inputs.item_int_feats.float(),
            inputs.user_dense_feats,
            inputs.item_dense_feats,
        )  # (B, ns_len, d_model)

        # 2. Embed and concatenate all sequence domains
        seq_tokens_list = []
        for domain in self.seq_domains:
            domain_tokens, _ = self._embed_seq_domain(
                domain,
                inputs.seq_data[domain],
                inputs.seq_lens[domain],
                inputs.seq_time_buckets.get(domain),
            )
            seq_tokens_list.append(domain_tokens)

        # Concatenate all domain tokens along sequence dimension
        # Each domain contributes seq_len tokens, total = num_domains * seq_len
        seq_tokens = torch.cat(seq_tokens_list, dim=1)  # (B, num_domains * seq_len, d_model)

        # 3. Apply dropout
        ns_tokens = self.dropout(ns_tokens)
        seq_tokens = self.dropout(seq_tokens)

        # 4. OneTrans backbone: [seq_tokens, sep_token, ns_tokens]
        batch_size = seq_tokens.size(0)
        sep_tokens = self.sep_token.expand(batch_size, -1, -1)
        x = torch.cat([seq_tokens, sep_tokens, ns_tokens], dim=1)
        x = self.base_block(x)
        for block in self.stack_blocks:
            x = block(x)

        # 5. Mean pooling + classification head
        pooled = x.mean(dim=1)
        logits = self.head(pooled)  # (B, 1)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference mode without dropout, returns logits and embeddings."""
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
