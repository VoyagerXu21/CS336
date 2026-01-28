# rope.py
from __future__ import annotations

import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """
    Apply RoPE (Rotary Positional Embedding) to an input tensor x.

    x: (..., seq_len, d_k)
    token_positions: (..., seq_len)  (same leading batch dims as x, excluding the last feature dim)
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        super().__init__()
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)

        # RoPE works on pairs; if d_k is odd, we rotate the first d_k-1 dims and keep the last dim unchanged.
        self.rope_dim = self.d_k if (self.d_k % 2 == 0) else (self.d_k - 1)
        if self.rope_dim < 0:
            raise ValueError(f"d_k must be >= 1, got {d_k}")

        half_dim = self.rope_dim // 2

        # inv_freq shape: (half_dim,)
        # Standard: inv_freq[i] = 1 / (theta ** (2*i/rope_dim))
        # Equivalent: 1 / (theta ** (arange(0, rope_dim, 2) / rope_dim))
        freq_seq = torch.arange(0, self.rope_dim, 2, dtype=torch.float32, device=device)  # (half_dim,)
        inv_freq = 1.0 / (self.theta ** (freq_seq / self.rope_dim))  # (half_dim,)

        # positions: (max_seq_len,)
        positions = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)  # (L,)

        # angles: (L, half_dim)
        angles = positions[:, None] * inv_freq[None, :]

        cos_cached = torch.cos(angles)  # (L, half_dim)
        sin_cached = torch.sin(angles)  # (L, half_dim)

        # Register buffers so they move with .to(device) / .cuda()
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len)
        returns: same shape as x
        """
        if x.dim() < 2:
            raise ValueError(f"x must have at least 2 dims (..., seq_len, d_k). Got shape {tuple(x.shape)}")
        if x.size(-1) != self.d_k:
            raise ValueError(f"Expected x last dim d_k={self.d_k}, got {x.size(-1)}")

        seq_len = x.size(-2)
        expected_pos_shape = x.shape[:-1]  # (..., seq_len)
        if token_positions.shape != expected_pos_shape:
            raise ValueError(
                f"token_positions must have shape {tuple(expected_pos_shape)}, got {tuple(token_positions.shape)}"
            )

        if self.rope_dim == 0:
            return x

        # Ensure positions are on same device, integer type for indexing
        pos = token_positions.to(device=x.device)
        if pos.dtype not in (torch.int32, torch.int64):
            pos = pos.long()

        # Safety check: positions within cache
        max_pos = int(pos.max().item()) if pos.numel() > 0 else 0
        if max_pos >= self.max_seq_len:
            raise ValueError(
                f"token_positions has value {max_pos} but max_seq_len={self.max_seq_len}. "
                f"Increase max_seq_len."
            )

        half_dim = self.rope_dim // 2

        # Gather cos/sin for each token position
        # pos_flat: (N,) where N = prod(batch_dims)*seq_len
        pos_flat = pos.reshape(-1)  # (N,)
        cos_flat = self.cos_cached.index_select(0, pos_flat)  # (N, half_dim)
        sin_flat = self.sin_cached.index_select(0, pos_flat)  # (N, half_dim)

        # Reshape back to (..., seq_len, half_dim)
        cos = cos_flat.view(*x.shape[:-1], half_dim).to(dtype=x.dtype)
        sin = sin_flat.view(*x.shape[:-1], half_dim).to(dtype=x.dtype)

        # Apply RoPE on the first rope_dim dims
        x_main = x[..., : self.rope_dim]  # (..., seq_len, rope_dim)
        x_rest = x[..., self.rope_dim :]  # (..., seq_len, d_k - rope_dim) possibly empty

        # Split into even/odd
        x_even = x_main[..., 0::2]  # (..., seq_len, half_dim)
        x_odd = x_main[..., 1::2]   # (..., seq_len, half_dim)

        # Rotate
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # Interleave back
        out_main = torch.empty_like(x_main)
        out_main[..., 0::2] = out_even
        out_main[..., 1::2] = out_odd

        if x_rest.numel() == 0:
            return out_main
        return torch.cat([out_main, x_rest], dim=-1)