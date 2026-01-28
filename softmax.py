# softmax.py
from __future__ import annotations

import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Numerically stable softmax along a given dimension.

    Args:
        x: input tensor of any shape
        dim: the dimension along which to apply softmax (can be negative)

    Returns:
        Tensor with the same shape as x. Values along `dim` sum to 1.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
    if x.numel() == 0:
        return x  # 空张量直接返回（一般测试不会用到）

    # 1) 取 dim 上的最大值，保留维度用于广播
    x_max = torch.amax(x, dim=dim, keepdim=True)

    # 2) 平移：让最大值变成 0，避免 exp(很大) -> inf
    x_shifted = x - x_max

    # 3) 指数
    exp_x = torch.exp(x_shifted)

    # 4) 归一化
    denom = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / denom
