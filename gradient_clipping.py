# gradient_clipping.py
from __future__ import annotations

from typing import Iterable, List, Optional

import torch


@torch.no_grad()
def clip_grad_l2_norm_(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    eps: float = 1e-6,
) -> float:
    """
    Clip gradients by global L2 norm (in-place), like:
        if ||g||_2 <= max_norm: do nothing
        else: g <- g * (max_norm / (||g||_2 + eps))

    Args:
        parameters: iterable of nn.Parameter (or anything with `.grad`)
        max_norm: maximum allowed global L2 norm (M)
        eps: numerical stability term (default 1e-6)

    Returns:
        total_norm (float): the pre-clipping global L2 norm of all grads.
    """
    if max_norm < 0:
        raise ValueError(f"max_norm must be non-negative, got {max_norm}")

    # Materialize in case it's a generator
    params: List[torch.nn.Parameter] = [p for p in parameters if p is not None]

    # Collect grads that exist
    grads: List[torch.Tensor] = []
    for p in params:
        if getattr(p, "grad", None) is None:
            continue
        g = p.grad
        if g is None:
            continue
        # Some grads can be sparse; this assignment typically doesn't use sparse.
        if g.is_sparse:
            g = g.coalesce().values()
        grads.append(g)

    if len(grads) == 0:
        return 0.0

    # Compute global L2 norm: sqrt(sum_i ||g_i||_2^2)
    # Use float32 accumulation for stability.
    device = grads[0].device
    total_sq = torch.zeros((), device=device, dtype=torch.float32)
    for g in grads:
        total_sq += g.detach().float().pow(2).sum()
    total_norm = torch.sqrt(total_sq)  # scalar tensor

    # Compute scaling coefficient
    # If total_norm <= max_norm -> coef >= 1, we do nothing
    coef = float(max_norm) / (float(total_norm.item()) + float(eps))

    if coef < 1.0:
        for p in params:
            if getattr(p, "grad", None) is None:
                continue
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                # scale sparse grad values in-place
                p.grad = p.grad.coalesce()
                p.grad._values().mul_(coef)
            else:
                p.grad.mul_(coef)

    return float(total_norm.item())

