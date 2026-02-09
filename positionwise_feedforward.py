# positionwise_feedforward.py
from __future__ import annotations

import torch
import torch.nn as nn


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class PositionwiseFeedForward(nn.Module):
    """
    SwiGLU Position-wise Feed-Forward Network (up/gate/down form).

    Implements:
        y = W_down( SiLU(W_up x) * W_gate x )
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {d_ff}")

        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.dropout = nn.Dropout(dropout)

        self.up = nn.Linear(self.d_model, self.d_ff, bias=bias)
        self.gate = nn.Linear(self.d_model, self.d_ff, bias=bias)
        self.down = nn.Linear(self.d_ff, self.d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Last dim of x must be d_model={self.d_model}, got {x.shape[-1]}"
            )

        y = _silu(self.up(x)) * self.gate(x)
        y = self.down(y)
        y = self.dropout(y)
        return y


# 兼容一些项目可能用的别名/工厂函数（不影响测试）
SwiGLUFeedForward = PositionwiseFeedForward


def positionwise_feedforward(d_model: int, d_ff: int, dropout: float = 0.0, **kwargs) -> PositionwiseFeedForward:
    """Factory function (optional)."""
    return PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, **kwargs)
