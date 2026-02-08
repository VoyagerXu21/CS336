from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from rope import RotaryPositionalEmbedding
from scaled_dot_product_attention import scaled_dot_product_attention


def _make_causal_keep_mask(max_seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Return shape (max_seq_len, max_seq_len) bool mask.

    IMPORTANT (aligned with your _masked_softmax semantics):
      - mask == True  : keep / allowed attention
      - mask == False : masked out (probability forced to 0)

    Causal self-attention keep-mask = lower triangle (including diagonal).
    i attends to j iff j <= i.
    """
    return torch.tril(
        torch.ones(max_seq_len, max_seq_len, device=device, dtype=torch.bool),
        diagonal=0,
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


class CausalMultiheadSelfAttention(nn.Module):
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
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=True, **factory_kwargs)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=True, **factory_kwargs)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=True, **factory_kwargs)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=True, **factory_kwargs)

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

        # ✅ Cache a big causal KEEP mask once (as buffer). Slice in forward.
        # persistent=False: not saved in state_dict (re-creatable)
        # NOTE: if device is None, create on CPU; buffer will move with module.to(device).
        init_device = device if device is not None else torch.device("cpu")
        mask = _make_causal_keep_mask(self.max_seq_len, device=init_device)
        self.register_buffer("causal_mask", mask, persistent=False)

    def _maybe_grow_rope_cache(self, needed_max_pos: int, device: torch.device) -> None:
        """
        If token_positions exceed current RoPE cache, grow it dynamically.
        Also grow the causal mask accordingly.
        """
        # grow causal mask regardless of rope, since T might exceed max_seq_len
        if needed_max_pos <= self.causal_mask.shape[0]:
            return

        new_max = int(needed_max_pos)
        self.max_seq_len = new_max

        # grow RoPE cache if used
        if self.use_rope:
            assert self.rope is not None
            self.rope = RotaryPositionalEmbedding(
                theta=self.rope_theta,
                d_k=self.head_dim,
                max_seq_len=new_max,
                device=device,
            )

        # grow causal KEEP mask on the same device as existing buffer
        new_mask = _make_causal_keep_mask(new_max, device=self.causal_mask.device)
        self.causal_mask = new_mask  # replace buffer tensor (allowed)

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

        # Ensure mask capacity (and rope cache if needed)
        if T > self.causal_mask.shape[0]:
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
            if needed_max_pos > self.causal_mask.shape[0]:
                self._maybe_grow_rope_cache(needed_max_pos=needed_max_pos, device=device)

            q = self.rope(q, pos_bht)
            k = self.rope(k, pos_bht)

        # 4) Causal KEEP mask slice: (T,T), True=keep, False=mask out
        causal_mask = self.causal_mask[:T, :T]
        if causal_mask.device != device:
            causal_mask = causal_mask.to(device)

        # 5) Attention
        out = scaled_dot_product_attention(q, k, v, mask=causal_mask)  # (B,H,T,hd)

        # 6) Merge heads: (B,H,T,hd)->(B,T,D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # 7) Output proj
        y = self.w_o(out)
        return y

