import torch
import torch.nn as nn

from torch import Tensor
from jaxtyping import Float, Int


class RotrayPositionalEmbedding(nn.Module):
    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000.0, device: torch.device | str | None = None, ):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for Rotary Positional Embedding")
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        t = torch.arange(max_seq_len, device=device)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.oneslike(freqs), freqs)
        self.register_buffer(
            "freqs_cis", freqs_cis, persistent=False
        )

    def forward(self, x:Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "...seq_len"],) -> Float[Tensor, "... seq_len d_k"]:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, -2))
        freqs_cis = self.freqs_cis[token_positions]
        if x_complex.dim() == 4:
            freqs_cis = freqs_cis.unsqueeze(1)
        x_rotated = x_complex * freqs_cis
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.reshape(*x.shape)
        return x_out.type_as(x)
