# linear.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    一个“无 bias”的线性层：
      输入:  (..., in_features)
      输出:  (..., out_features)

    注意：题目要求把参数存成 W（不是 W^T）：
      W.shape = (in_features, out_features)
    这样 forward 里自然是 x @ W
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()  # ✅ 必须调用父类构造函数（nn.Module 需要注册参数等）

        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # ✅ 按题意：存 W 而不是 W^T
        # W: (in_features, out_features)
        w = torch.empty(self.in_features, self.out_features, device=device, dtype=dtype)

        # ✅ 题目要求用 trunc_normal_ 初始化（并使用“上面”的设置）
        # 常见设置（Transformer/GPT 风格）：mean=0, std=0.02, 截断区间 [-2σ, 2σ]
        torch.nn.init.trunc_normal_(w, mean=0.0, std=0.02, a=-0.04, b=0.04)

        # ✅ 注册成可训练参数
        self.W = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features)
        return: (..., out_features)
        """
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Linear expected last dim = {self.in_features}, got {x.size(-1)}"
            )

        # ✅ 不用 nn.functional.linear（题目禁止）
        # torch.matmul / @ 会自动处理 batch 维度：(..., in) @ (in, out) -> (..., out)
        return x @ self.W

