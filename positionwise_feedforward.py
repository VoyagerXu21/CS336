# positionwise_feedforward.py
from __future__ import annotations

import math
import torch
import torch.nn as nn


def _round_to_multiple(x: int, multiple: int) -> int:
    """Round x up to the nearest multiple."""
    return ((x + multiple - 1) // multiple) * multiple


class PositionwiseFeedForward(nn.Module):
    """
    SwiGLU Position-wise Feed-Forward Network.

    Implements:
        y = W_out( (xW_g) * SiLU(xW_v) )
    where SiLU(z) = z * sigmoid(z)

    Notes:
    - d_ff ≈ (8/3) * d_model
    - d_ff must be a multiple of 64
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        multiple_of: int = 64,
        bias: bool = True,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        # d_ff ≈ 8/3 * d_model, then round up to multiple_of (default 64)
        d_ff_raw = int((8.0 / 3.0) * d_model)
        d_ff = _round_to_multiple(max(1, d_ff_raw), multiple_of)

        self.d_model = int(d_model)
        self.d_ff = int(d_ff)

        # One projection produces both gate/value parts: [*, 2*d_ff] -> split
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_ff, bias=bias)
        self.out_proj = nn.Linear(self.d_ff, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _silu(x: torch.Tensor) -> torch.Tensor:
        # numerically stable SiLU using sigmoid
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d_model)
        return: (..., d_model)
        """
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Last dim of x must be d_model={self.d_model}, got {x.shape[-1]}"
            )

        h = self.in_proj(x)                # (..., 2*d_ff)
        x_g, x_v = h.chunk(2, dim=-1)      # (..., d_ff), (..., d_ff)

        # SwiGLU: gate * SiLU(value)
        y = x_g * self._silu(x_v)          # (..., d_ff)

        y = self.out_proj(y)               # (..., d_model)
        y = self.dropout(y)
        return y


# 兼容一些项目可能用的别名/工厂函数（不影响测试）
SwiGLUFeedForward = PositionwiseFeedForward


def positionwise_feedforward(d_model: int, dropout: float = 0.0, **kwargs) -> PositionwiseFeedForward:
    """Factory function (optional)."""
    return PositionwiseFeedForward(d_model=d_model, dropout=dropout, **kwargs)
