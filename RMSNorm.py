# RMSNorm.py
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    - 不减均值，只按 RMS(均方根) 做缩放
    - 输出形状与输入相同
    - 只有一个可学习参数 gamma（也叫 weight），形状 (d_model,)
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.d_model = int(d_model)
        self.eps = float(eps)

        # gamma 参数：形状 (d_model,)
        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        self.weight = nn.Parameter(torch.ones(self.d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d_model) 典型是 (batch, seq, d_model)
        返回: 同 shape
        注意作业要求：归一化计算前 upcast 到 float32，之后再 cast 回原 dtype
        """
        orig_dtype = x.dtype

        # 1) upcast 到 float32 做数值稳定的归一化
        x_fp32 = x.to(torch.float32)

        # 2) 计算均方：mean(x^2, dim=-1, keepdim=True)
        #    shape: (..., 1)
        mean_sq = x_fp32.pow(2).mean(dim=-1, keepdim=True)

        # 3) rms = sqrt(mean_sq + eps)，但我们通常用 rsqrt 更快更稳
        #    inv_rms = 1 / sqrt(mean_sq + eps)
        inv_rms = torch.rsqrt(mean_sq + self.eps)

        # 4) 归一化：x_hat = x * inv_rms
        x_hat = x_fp32 * inv_rms

        # 5) 乘可学习缩放 gamma（weight）
        #    weight shape (d_model,) 会广播到 (..., d_model)
        y_fp32 = x_hat * self.weight.to(torch.float32)

        # 6) cast 回原 dtype（比如 fp16/bf16）
        return y_fp32.to(orig_dtype)

