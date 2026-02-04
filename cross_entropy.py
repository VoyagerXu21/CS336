# cross_entropy.py
from __future__ import annotations

import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Numerically-stable cross entropy for logits.

    Args:
        logits: (*batch_dims, V)
        targets: (*batch_dims,)  int class indices in [0, V)

    Returns:
        scalar tensor: mean loss over all batch-like elements.
    """
    if logits.ndim < 1:
        raise ValueError(f"logits must have at least 1 dim, got {tuple(logits.shape)}")
    if targets.shape != logits.shape[:-1]:
        raise ValueError(
            f"Shape mismatch: logits {tuple(logits.shape)} expects targets {tuple(logits.shape[:-1])}, "
            f"but got {tuple(targets.shape)}"
        )

    # targets 必须是整型索引
    if targets.dtype != torch.long:
        targets = targets.to(torch.long)

    # 半精度下提升稳定性
    calc_dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
    logits_ = logits.to(calc_dtype)

    # 1) subtract max for numerical stability
    # max_logits: (*batch_dims,)
    max_logits = logits_.amax(dim=-1)
    logits_shifted = logits_ - max_logits.unsqueeze(-1)

    # 2) logsumexp = max + log(sum(exp(shifted)))
    sum_exp = torch.exp(logits_shifted).sum(dim=-1)
    logsumexp = torch.log(sum_exp) + max_logits

    # 3) pick correct class logit: logits[..., y]
    correct = logits_.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # 4) CE per element: logsumexp - correct
    loss = logsumexp - correct

    # 5) average across all batch-like dims
    return loss.mean()

