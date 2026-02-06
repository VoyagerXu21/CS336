# run_decode_with_gt.py
from __future__ import annotations

import random
from typing import List, Optional, Sequence

import numpy as np
import torch

from tokenizerclass import Tokenizer
from transformer_lm import TransformerLM
from data_loader import load_tokenized_ids  # 你训练时用的 loader


def _get_end_token_id(tokenizer: Tokenizer, end_token: Optional[str]) -> Optional[int]:
    if end_token is None:
        return None
    ids = tokenizer.encode(end_token)
    if len(ids) != 1:
        raise ValueError(f"end_token must map to a single id, got {len(ids)} ids for {end_token!r}")
    return ids[0]


def _sample_next_token(logits: torch.Tensor, *, temperature: float, top_p: float) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())

    probs = torch.softmax(logits / float(temperature), dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum <= float(top_p)
        mask[..., 0] = True
        filtered = sorted_probs * mask
        filtered = filtered / filtered.sum(dim=-1, keepdim=True)
        j = int(torch.multinomial(filtered, 1).item())
        return int(sorted_idx[j].item())

    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate_completion_from_ids(
    model: TransformerLM,
    prompt_ids: List[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    end_token_id: Optional[int],
    device: torch.device,
) -> List[int]:
    model.eval()
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_len = input_ids.shape[1]

    for _ in range(max_new_tokens):
        if input_ids.shape[1] > model.context_length:
            x = input_ids[:, -model.context_length :]
        else:
            x = input_ids

        logits = model(x)          # (1, T, V)
        next_logits = logits[0, -1]  # (V,)
        nxt = _sample_next_token(next_logits, temperature=temperature, top_p=top_p)

        input_ids = torch.cat([input_ids, torch.tensor([[nxt]], device=device)], dim=1)

        if end_token_id is not None and nxt == end_token_id:
            break

    gen_ids = input_ids[0].tolist()
    return gen_ids[prompt_len:]  # only completion ids


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== tokenizer =====
    tokenizer = Tokenizer.from_files(
        "bpe_tinystories_10k/vocab.json",
        "bpe_tinystories_10k/merges.json",
        special_tokens=["<|endoftext|>"],  # 训练时确实加过才保留；不确定就改 None
    )
    end_token = "<|endoftext|>"           # 不确定就改 None
    end_token_id = _get_end_token_id(tokenizer, end_token)

    # ===== model (要与 ckpt 匹配) =====
    model = TransformerLM(
        vocab_size=10000,
        context_length=256,
        num_layers=4,
        d_model=512,
        num_heads=8,
        d_ff=1408,
        dropout=0.0,
        bias=True,
        eps=1e-5,
        rope_theta=10000.0,
        use_rope=True,
        max_seq_len=4096,
        tie_weights=True,
        emb_dropout=None,
    ).to(device)

    ckpt = torch.load("checkpoints/ckpt_step_2000.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # ===== load val tokens =====
    val_tokens = load_tokenized_ids("tinystories_dev_uint16.npy", root_dir="tokenized_ids", mmap=True)  # ←按你真实文件名改
    val_tokens = np.asarray(val_tokens, dtype=np.int64)

    context_length = model.context_length
    max_new_tokens = 100

    # 随机选一个起点，保证后面够长
    max_start = len(val_tokens) - (context_length + max_new_tokens + 1)
    start = random.randint(0, max_start)

    prompt_ids = val_tokens[start : start + context_length].tolist()
    gt_ids = val_tokens[start + context_length : start + context_length + max_new_tokens].tolist()

    # ===== model generate =====
    gen_ids = generate_completion_from_ids(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.65,
        top_p=0.85,
        end_token_id=end_token_id,
        device=device,
    )

    # ===== decode =====
    prompt_text = tokenizer.decode(prompt_ids)
    gt_text = tokenizer.decode(gt_ids)
    gen_text = tokenizer.decode(gen_ids)

    print("===== PROMPT (from val set) =====")
    print(prompt_text)
    print("\n===== GT (true continuation in val set) =====")
    print(gt_text)
    print("\n===== MODEL (generated continuation) =====")
    print(gen_text)


if __name__ == "__main__":
    main()

