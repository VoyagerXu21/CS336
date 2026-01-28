# Embedding.py
from __future__ import annotations

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    一个最小版 nn.Embedding：
    - weight: (num_embeddings, embedding_dim)
    - forward(token_ids): 返回 (..., embedding_dim)
    注意：不允许用 nn.Embedding 或 nn.functional.embedding
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        # 记录超参数（有些测试/调试会用到）
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)

        # factory_kwargs：把 device/dtype 统一传给 tensor 构造
        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        # 关键：embedding 矩阵的最后一维必须是 embedding_dim
        # shape = (V, D)
        self.weight = nn.Parameter(
            torch.empty((self.num_embeddings, self.embedding_dim), **factory_kwargs)
        )

        # 按作业要求：用 trunc_normal_ 初始化
        # 这里用常见设置：mean=0, std=0.02, 截断到 [-2*std, 2*std]
        std = 0.02
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: 任意形状的整数张量，如 (batch, seq) 或 (seq,)
        返回: token_ids.shape + (embedding_dim,)
        """

        if token_ids.numel() == 0:
            # 空输入：返回正确形状的空张量
            out_shape = (*token_ids.shape, self.embedding_dim)
            return self.weight.new_empty(out_shape)

        # embedding lookup 必须用 long 索引
        if token_ids.dtype != torch.long:
            token_ids = token_ids.to(torch.long)

        # 一般测试会让 token_ids 和 weight 在同一 device；不一致会报错，这里显式提示
        if token_ids.device != self.weight.device:
            raise RuntimeError(
                f"token_ids device {token_ids.device} != weight device {self.weight.device}"
            )

        # 核心：高级索引（不是 nn.functional.embedding）
        # weight: (V, D)
        # token_ids: (...,)
        # weight[token_ids] -> (..., D)
        return self.weight[token_ids]

