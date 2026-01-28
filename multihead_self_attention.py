from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from rope import RotaryPositionalEmbedding
from scaled_dot_product_attention import scaled_dot_product_attention


def _make_causal_mask(max_seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Return shape (max_seq_len, max_seq_len) bool mask.
    True means "masked out" (upper triangle, excluding diagonal).
    """
    return torch.triu(
        torch.ones(max_seq_len, max_seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )


def _expand_positions_for_heads(
    token_positions: torch.Tensor,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert token_positions to shape (B, H, T) for RoPE.
    Accept: (T,), (B,T), (B,H,T)
    Output: (B,H,T) long on `device`.
    """
    if token_positions.dim() == 1:
        if token_positions.shape[0] != seq_len:
            raise ValueError(f"token_positions {tuple(token_positions.shape)} != (T,) with T={seq_len}")
        pos = token_positions.to(device=device)
        if pos.dtype not in (torch.int32, torch.int64):
            pos = pos.long()
        return pos.view(1, 1, seq_len).expand(batch_size, num_heads, seq_len)

    if token_positions.dim() == 2:
        if token_positions.shape != (batch_size, seq_len):
            raise ValueError(f"token_positions {tuple(token_positions.shape)} != (B,T)={(batch_size, seq_len)}")
        pos = token_positions.to(device=device)
        if pos.dtype not in (torch.int32, torch.int64):
            pos = pos.long()
        return pos.view(batch_size, 1, seq_len).expand(batch_size, num_heads, seq_len)

    if token_positions.dim() == 3:
        if token_positions.shape != (batch_size, num_heads, seq_len):
            raise ValueError(
                f"token_positions {tuple(token_positions.shape)} != (B,H,T)={(batch_size, num_heads, seq_len)}"
            )
        pos = token_positions.to(device=device)
        if pos.dtype not in (torch.int32, torch.int64):
            pos = pos.long()
        return pos

    raise ValueError(f"token_positions must be 1d/2d/3d, got dim={token_positions.dim()}")


class CasualMultiheadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention with optional RoPE.

    input:  x (B, T, d_model)
    output: y (B, T, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        rope_theta: float = 10000.0,
        use_rope: bool = True,
        max_seq_len: int = 4096,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, got {self.d_model}/{self.num_heads}")
        self.head_dim = self.d_model // self.num_heads

        self.use_rope = bool(use_rope)
        self.rope_theta = float(rope_theta)
        self.max_seq_len = int(max_seq_len)

        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        # Projections
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False, **factory_kwargs)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False, **factory_kwargs)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False, **factory_kwargs)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False, **factory_kwargs)

        # RoPE
        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=self.rope_theta,
                d_k=self.head_dim,
                max_seq_len=self.max_seq_len,
                device=device,
            )
        else:
            self.rope = None

        # ✅ Cache a big causal mask once (as buffer). Slice in forward.
        # persistent=False: 不写进 state_dict（mask 可复现）
        mask = _make_causal_mask(self.max_seq_len, device=(device if device is not None else torch.device("cpu")))
        self.register_buffer("causal_mask", mask, persistent=False)

    def _maybe_grow_rope_cache(self, needed_max_pos: int, device: torch.device) -> None:
        """
        If token_positions exceed current RoPE cache, grow it dynamically.
        """
        if not self.use_rope:
            return
        assert self.rope is not None

        # 这里假设你的 RotaryPositionalEmbedding 有 max_seq_len 属性
        if needed_max_pos <= self.rope.max_seq_len:
            return

        new_max = int(needed_max_pos)
        self.max_seq_len = new_max
        self.rope = RotaryPositionalEmbedding(
            theta=self.rope_theta,
            d_k=self.head_dim,
            max_seq_len=new_max,
            device=device,
        )

        # 同步扩展 causal mask（否则 T 变大时 mask 不够切）
        # 注意：buffer 会跟随 .to(device) 移动，所以这里用 self.causal_mask.device 即可
        new_mask = _make_causal_mask(new_max, device=self.causal_mask.device)
        self.causal_mask = new_mask  # 直接替换 buffer 张量（允许）

    def forward(
        self,
        x: torch.Tensor,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, T, d_model)
        token_positions: (T,) or (B,T) or (B,H,T)
        """
        if x.dim() != 3:
            raise ValueError(f"x must be (B,T,D), got {tuple(x.shape)}")
        B, T, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Expected last dim d_model={self.d_model}, got {D}")

        device = x.device

        # default positions: (T,)
        if token_positions is None:
            token_positions = torch.arange(T, device=device, dtype=torch.int64)

        # Ensure mask capacity
        if T > self.causal_mask.shape[0]:
            # grow both rope cache and causal mask (safe even if use_rope=False)
            self._maybe_grow_rope_cache(needed_max_pos=T, device=device)

        # 1) Project: (B,T,D)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 2) Split heads: (B,T,D)->(B,H,T,hd)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) RoPE on Q/K
        if self.use_rope:
            assert self.rope is not None
            pos_bht = _expand_positions_for_heads(
                token_positions=token_positions,
                batch_size=B,
                num_heads=self.num_heads,
                seq_len=T,
                device=device,
            )
            needed_max_pos = int(pos_bht.max().item()) + 1 if pos_bht.numel() > 0 else T
            self._maybe_grow_rope_cache(needed_max_pos=needed_max_pos, device=device)

            q = self.rope(q, pos_bht)
            k = self.rope(k, pos_bht)

        # 4) Causal mask slice: (T,T)
        causal_mask = self.causal_mask[:T, :T]
        # 确保 mask 跟 x 在同一 device（一般 buffer 会自动跟随 module.to(device)）
        if causal_mask.device != device:
            causal_mask = causal_mask.to(device)

        # 5) Attention
        out = scaled_dot_product_attention(q, k, v, mask=causal_mask)  # (B,H,T,hd)

        # 6) Merge heads: (B,H,T,hd)->(B,T,D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # 7) Output proj
        y = self.w_o(out)
        return y

