# transformer_lm.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from RMSNorm import RMSNorm
from transformer_block import TransformerBlock
from Embedding import Embedding
from Linear import Linear


class TransformerLM(nn.Module):
    """
    A GPT-style Transformer Language Model (causal LM).

    Forward:
      input_ids: (B, T) int64
      returns logits: (B, T, vocab_size)

    Components:
      tok_emb: (vocab_size, d_model)
      pos_emb: (context_length, d_model)
      blocks: num_layers * TransformerBlock
      norm_f: RMSNorm(d_model) or Identity
      lm_head: Linear(d_model -> vocab_size)
        - if tie_weights=True, lm_head has NO own parameter; it uses tok_emb.weight.T

    Notes:
      - Causality is enforced inside TransformerBlock's MHA (your implementation).
      - token_positions are generated as arange(T) and passed into blocks (for RoPE).
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        bias: bool = True,
        eps: float = 1e-5,
        rope_theta: float = 10000.0,
        use_rope: bool = True,
        max_seq_len: int = 4096,
        tie_weights: bool = True,
        use_final_norm: bool = True,
        emb_dropout: Optional[float] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.vocab_size = int(vocab_size)
        self.context_length = int(context_length)
        self.num_layers = int(num_layers)

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)

        self.dropout = float(dropout)
        self.bias = bool(bias)
        self.eps = float(eps)

        self.rope_theta = float(rope_theta)
        self.use_rope = bool(use_rope)
        self.max_seq_len = int(max_seq_len)

        self.tie_weights = bool(tie_weights)
        self.use_final_norm = bool(use_final_norm)

        if emb_dropout is None:
            emb_dropout = dropout
        self.emb_dropout = float(emb_dropout)

        factory_kwargs: Dict[str, Any] = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        # ---- Embeddings ----
        self.tok_emb = Embedding(self.vocab_size, self.d_model, **factory_kwargs)
        # self.pos_emb = Embedding(self.context_length, self.d_model, **factory_kwargs)
        self.drop = nn.Dropout(self.emb_dropout)

        # ---- Blocks ----
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    bias=self.bias,
                    eps=self.eps,
                    rope_theta=self.rope_theta,
                    use_rope=self.use_rope,
                    max_seq_len=max(self.max_seq_len, self.context_length),
                    device=device,
                    dtype=dtype,
                )
                for _ in range(self.num_layers)
            ]
        )

        # ---- Final norm ----
        self.norm_f = (
            RMSNorm(self.d_model, eps=self.eps, **factory_kwargs)
            if self.use_final_norm
            else nn.Identity()
        )

        # ---- LM head ----
        # Always create Linear so your structure stays the same;
        # if tie_weights=True, we REMOVE its parameter and bind W to tok_emb.weight.T.
        self.lm_head = Linear(self.d_model, self.vocab_size, device=device, dtype=dtype)

        # ---- Init ----
        self._init_weights()

    def _init_weights(self) -> None:
        # Embeddings
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        # nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

        # lm_head:
        # - if tie_weights=True: lm_head has no own parameter; do NOT init it.
        # - else: init its independent W.
        if not self.tie_weights:
            nn.init.normal_(self.lm_head.W, mean=0.0, std=0.02)

        # (Optional) If you want to init other modules, do it inside those modules' own __init__.
        # Here we keep it minimal and safe given custom Linear stores weight as .W.

    @torch.no_grad()
    def get_token_position(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.arange(T, device=device, dtype=torch.long)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        targets: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        shift_targets: bool = True,
        ignore_index: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be (B,T), got {tuple(input_ids.shape)}")
        if input_ids.dtype != torch.long:
            raise TypeError(f"input_ids must be torch.long, got {input_ids.dtype}")

        B, T = input_ids.shape
        if T > self.context_length:
            raise ValueError(f"Sequence length T={T} exceeds context_length={self.context_length}")

        token_positions = self.get_token_position(T, device=input_ids.device)

        # (B,T,D)
        tok = self.tok_emb(input_ids)
        # pos = self.pos_emb(token_positions).unsqueeze(0)
        x = self.drop(tok)

        for blk in self.blocks:
            x = blk(x, token_positions=token_positions)

        x = self.norm_f(x)
        if self.tie_weights:
            # tok_emb.weight: (V, D)  ->  (D, V)
            logits = x @ self.tok_emb.weight.T  # (B,T,V)
        else:
            logits = self.lm_head(x)  # (B,T,V)

        if not (return_loss and targets is not None):
            return logits

        if targets.shape != input_ids.shape:
            raise ValueError(f"targets must have shape (B,T) like input_ids, got {tuple(targets.shape)}")
        if targets.dtype != torch.long:
            raise TypeError(f"targets must be torch.long, got {targets.dtype}")

        if ignore_index is None:
            if (targets == -100).any():
                ignore_index = -100
            elif (targets == -1).any():
                ignore_index = -1
            else:
                ignore_index = -100

        if shift_targets:
            logits_use = logits[:, :-1, :].contiguous()
            targets_use = targets[:, 1:].contiguous()
        else:
            logits_use = logits
            targets_use = targets

        loss = F.cross_entropy(
            logits_use.view(-1, self.vocab_size),
            targets_use.view(-1),
            ignore_index=ignore_index,
        )
        return logits, loss
