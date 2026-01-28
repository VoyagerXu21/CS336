# transformer_block.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from RMSNorm import RMSNorm
from positionwise_feedforward import PositionwiseFeedForward  # 你的 SwiGLU FFN
from multihead_self_attention import CasualMultiheadSelfAttention  # ✅ 你已改成“无 lazy init”的 MHA


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block:
      y = x + MHA(RMSNorm(x))
      z = y + FFN(RMSNorm(y))

    Required init params:
      - d_model: int
      - num_heads: int
      - d_ff: int
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
        eps: float = 1e-5,
        # RoPE-related
        rope_theta: float = 10000.0,
        use_rope: bool = True,
        max_seq_len: int = 4096,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)

        self.dropout = float(dropout)
        self.bias = bool(bias)
        self.eps = float(eps)

        self.rope_theta = float(rope_theta)
        self.use_rope = bool(use_rope)
        self.max_seq_len = int(max_seq_len)

        # ---- factory kwargs (for Linear etc.) ----
        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        # 1) Two pre-norms
        self.norm1 = RMSNorm(self.d_model, eps=self.eps, device=device, dtype=dtype)
        self.norm2 = RMSNorm(self.d_model, eps=self.eps, device=device, dtype=dtype)

        # 2) MHA (✅ no lazy init)
        self.attn = CasualMultiheadSelfAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            rope_theta=self.rope_theta,
            use_rope=self.use_rope,
            max_seq_len=self.max_seq_len,
            device=device,
            dtype=dtype,
        )

        # 3) FFN (reuse your SwiGLU FFN, but force d_ff to match external argument)
        self.ff = PositionwiseFeedForward(
            d_model=self.d_model,
            dropout=self.dropout,
            multiple_of=1,  # don't force alignment to avoid mismatch with explicit d_ff in tests
            bias=self.bias,
        )
        self._force_ff_dff(**factory_kwargs)

        # 4) Residual dropouts
        self.resid_dropout1 = nn.Dropout(self.dropout)
        self.resid_dropout2 = nn.Dropout(self.dropout)

    def _force_ff_dff(self, **factory_kwargs) -> None:
        """
        Force PositionwiseFeedForward to use the externally provided d_ff.
        This avoids test failures when tests explicitly pass d_ff.
        """
        if not hasattr(self.ff, "d_ff"):
            return

        if int(self.ff.d_ff) == self.d_ff:
            return

        # rebuild layers: in_proj: d_model -> 2*d_ff, out_proj: d_ff -> d_model
        self.ff.d_ff = int(self.d_ff)
        self.ff.in_proj = nn.Linear(self.d_model, 2 * self.d_ff, bias=self.bias, **factory_kwargs)
        self.ff.out_proj = nn.Linear(self.d_ff, self.d_model, bias=self.bias, **factory_kwargs)
        self.ff.dropout = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, d_model)
        token_positions: optional positions for RoPE
          - (T,) or (B,T) or (B,H,T)  (match your MHA)
        """
        if x.dim() != 3:
            raise ValueError(f"x must be (B,T,d_model), got {tuple(x.shape)}")

        B, T, D = x.shape
        if D != self.d_model:
            raise ValueError(f"Expected x last dim d_model={self.d_model}, got {D}")

        # --- Sublayer 1: x + MHA(RMSNorm(x)) ---
        h = self.norm1(x)
        h = self.attn(h, token_positions=token_positions)
        x = x + self.resid_dropout1(h)

        # --- Sublayer 2: x + FFN(RMSNorm(x)) ---
        h2 = self.norm2(x)
        h2 = self.ff(h2)
        x = x + self.resid_dropout2(h2)

        return x

