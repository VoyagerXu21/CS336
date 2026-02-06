from __future__ import annotations

from typing import Optional, Sequence

import torch

from tokenizerclass import Tokenizer
from transformer_lm import TransformerLM


def _get_end_token_id(tokenizer: Tokenizer, end_token: Optional[str]) -> Optional[int]:
    if end_token is None:
        return None
    token_ids = tokenizer.encode(end_token)
    if len(token_ids) != 1:
        raise ValueError(
            f"end_token must map to a single id, got {len(token_ids)} ids for {end_token!r}"
        )
    return token_ids[0]


def _sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())

    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative <= top_p
        mask[..., 0] = True
        filtered_probs = sorted_probs * mask
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        next_idx = torch.multinomial(filtered_probs, num_samples=1).item()
        return int(sorted_indices[next_idx].item())

    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_token: Optional[str] = "<|endoftext|>",
    device: Optional[torch.device] = None,
) -> str:
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    if top_p <= 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    model.eval()

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("prompt must produce at least one token")

    end_token_id = _get_end_token_id(tokenizer, end_token)

    if device is None:
        device = next(model.parameters()).device

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        if input_ids.shape[1] > model.context_length:
            input_slice = input_ids[:, -model.context_length :]
        else:
            input_slice = input_ids

        logits = model(input_slice)
        next_logits = logits[0, -1]

        next_token = _sample_next_token(
            next_logits,
            temperature=temperature,
            top_p=top_p,
        )

        next_token_tensor = torch.tensor([[next_token]], device=device, dtype=torch.long)
        input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

        if end_token_id is not None and next_token == end_token_id:
            break

    generated_ids: Sequence[int] = input_ids[0].tolist()
    return tokenizer.decode(list(generated_ids))


