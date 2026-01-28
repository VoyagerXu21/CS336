# scaled_dot_product_attention.py
from __future__ import annotations

import math
from typing import Optional

import torch


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    A numerically stable masked softmax:
    - mask == True : keep
    - mask == False: probability forced to 0
    Handles the (rare) case where an entire row is masked (sum = 0) -> returns all zeros.
    """
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be bool, got {mask.dtype}")

    # Put a very negative value where masked out.
    # Using -1e9 avoids -inf propagation issues in half precision.
    neg = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
    logits = torch.where(mask, logits, neg)

    # Stable softmax: exp(logits - max) * mask
    maxv = logits.max(dim=dim, keepdim=True).values
    exps = torch.exp(logits - maxv) * mask.to(dtype=logits.dtype)

    denom = exps.sum(dim=dim, keepdim=True)
    # If denom == 0 (all masked), avoid NaN and return all zeros.
    denom = denom.clamp_min(torch.finfo(exps.dtype).tiny)

    return exps / denom


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention.

    Args:
        q: (batch_size, ..., seq_len, d_k)
        k: (batch_size, ..., seq_len, d_k)
        v: (batch_size, ..., seq_len, d_v)
        mask: optional boolean mask of shape (seq_len, seq_len)
              mask[i, j] == True means query position i can attend to key position j.

    Returns:
        out: (batch_size, ..., seq_len, d_v)
    """
    if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
        raise ValueError("q/k/v must be at least 2D tensors with (..., seq_len, d).")

    if q.shape[:-1] != k.shape[:-1]:
        raise ValueError(f"q and k must match on all dims except last: {q.shape} vs {k.shape}")
    if q.shape[:-2] != v.shape[:-2]:
        raise ValueError(f"q and v must match on batch-like dims: {q.shape} vs {v.shape}")

    seq_len = q.shape[-2]
    d_k = q.shape[-1]

    if k.shape[-2] != seq_len:
        raise ValueError("This assignment expects self-attention style shapes: k.shape[-2] == q.shape[-2].")
    if v.shape[-2] != seq_len:
        raise ValueError("This assignment expects v.shape[-2] == q.shape[-2].")

    # Compute in float32 for stability if using half/bfloat16; cast back at end.
    compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    q_ = q.to(dtype=compute_dtype)
    k_ = k.to(dtype=compute_dtype)
    v_ = v.to(dtype=compute_dtype)

    scale = 1.0 / math.sqrt(d_k)

    # logits: (..., seq_len, seq_len)
    logits = torch.matmul(q_, k_.transpose(-1, -2)) * scale

    if mask is not None:
        if mask.shape != (seq_len, seq_len):
            raise ValueError(f"mask must have shape ({seq_len}, {seq_len}), got {tuple(mask.shape)}")
        if mask.dtype != torch.bool:
            mask = mask.to(dtype=torch.bool)

        # Broadcast mask to logits shape: (1,1,...,seq_len,seq_len)
        # logits.ndim = len(batch_like_dims) + 2
        expand_shape = (1,) * (logits.ndim - 2) + (seq_len, seq_len)
        mask_b = mask.view(expand_shape)
        attn = _masked_softmax(logits, mask_b, dim=-1)
    else:
        attn = torch.softmax(logits, dim=-1)

    # out: (..., seq_len, d_v)
    out = torch.matmul(attn, v_)

    return out.to(dtype=v.dtype)