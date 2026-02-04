# train_loop.py
from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional
from checkpointing import save_checkpoint, load_checkpoint
import numpy as np
import torch

from transformer_lm import TransformerLM  # your GPT-style LM


from cross_entropy import cross_entropy as user_cross_entropy  # <- adjust name if needed



# optimizer choices (your implementations)
try:
    from AdamW import AdamW  # your AdamW
except Exception:
    AdamW = None

try:
    from SGD import SGD  # your SGD
except Exception:
    SGD = None

# grad clipping (your implementation)
try:
    from gradient_clipping import clip_grad_l2_norm_
except Exception:
    clip_grad_l2_norm_ = None

from learning_rate_schedule import lr_cosine_with_warmup

try:
    import checkpointing as user_ckpt
except Exception:
    user_ckpt = None


# =========================
# Config dataclasses
# =========================
@dataclass
class ModelCfg:
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    dropout: float
    bias: bool
    eps: float
    rope_theta: float
    use_rope: bool
    max_seq_len: int
    tie_weights: bool
    emb_dropout: Optional[float]


@dataclass
class OptimCfg:
    optimizer: str  # "adamw" or "sgd"
    lr_max: float
    lr_min: float
    warmup_steps: int
    cosine_end_step: int  # Tc
    weight_decay: float
    betas: Tuple[float, float]
    eps: float
    momentum: float  # for SGD


@dataclass
class TrainCfg:
    train_tokens_path: str
    val_tokens_path: str
    token_dtype: str
    batch_size: int
    total_steps: int
    grad_accum_steps: int
    max_grad_norm: float
    log_every: int
    eval_every: int
    eval_iters: int
    save_every: int
    save_dir: str
    resume: str
    device: str
    seed: int
    wandb: bool
    wandb_project: str
    wandb_name: str


# =========================
# Utilities
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dtype_from_str(s: str) -> np.dtype:
    s = s.lower()
    if s in ["uint16", "u2"]:
        return np.uint16
    if s in ["int32", "i4"]:
        return np.int32
    if s in ["int64", "i8"]:
        return np.int64
    raise ValueError(f"Unsupported token dtype: {s} (use uint16/int32/int64)")


def load_tokens_memmap(path: str, dtype: np.dtype) -> np.ndarray:
    """
    Load 1D token array with memory-mapped IO.
    Supports:
      - .npy  (np.load(..., mmap_mode='r'))
      - .bin  (np.memmap)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.endswith(".npy"):
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 1:
            raise ValueError(f"{path} must be a 1D token array, got shape={arr.shape}")
        return arr

    if path.endswith(".bin"):
        itemsize = np.dtype(dtype).itemsize
        n = os.path.getsize(path) // itemsize
        arr = np.memmap(path, dtype=dtype, mode="r", shape=(n,))
        return arr

    raise ValueError(f"Unsupported token file: {path} (use .npy or .bin)")


@torch.no_grad()
def get_batch(tokens_1d: np.ndarray, batch_size: int, context_length: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample random (x, y) from a long 1D token stream.
      x: (B, T)
      y: (B, T) = next-token labels (aligned with logits at each position)
    """
    n = int(tokens_1d.shape[0])
    if n <= context_length + 1:
        raise ValueError(f"Token stream too short: n={n}, context_length={context_length}")

    starts = np.random.randint(0, n - context_length - 1, size=(batch_size,))
    x_np = np.stack([tokens_1d[s : s + context_length] for s in starts], axis=0)
    y_np = np.stack([tokens_1d[s + 1 : s + 1 + context_length] for s in starts], axis=0)

    x = torch.from_numpy(np.asarray(x_np)).long().to(device, non_blocking=True)
    y = torch.from_numpy(np.asarray(y_np)).long().to(device, non_blocking=True)
    return x, y


def _zero_grad_compat(optimizer: Any) -> None:
    if not hasattr(optimizer, "zero_grad"):
        return
    try:
        optimizer.zero_grad(set_to_none=True)
    except TypeError:
        optimizer.zero_grad()


def set_optimizer_lr(optimizer: Any, lr: float) -> None:
    if hasattr(optimizer, "param_groups"):
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        return
    if hasattr(optimizer, "lr"):
        optimizer.lr = lr
        return
    if hasattr(optimizer, "set_lr"):
        optimizer.set_lr(lr)
        return
    raise AttributeError("Optimizer has no known way to set lr (param_groups / lr / set_lr).")


def compute_lr(step: int, opt_cfg: OptimCfg) -> float:
    if lr_cosine_with_warmup is not None:
        return float(
            lr_cosine_with_warmup(
                t=step,
                alpha_max=opt_cfg.lr_max,
                alpha_min=opt_cfg.lr_min,
                Tw=opt_cfg.warmup_steps,
                Tc=opt_cfg.cosine_end_step,
            )
        )

    # fallback (assignment definition)
    t = step
    Tw = opt_cfg.warmup_steps
    Tc = opt_cfg.cosine_end_step
    amax = opt_cfg.lr_max
    amin = opt_cfg.lr_min

    if t < Tw:
        return (t / max(1, Tw)) * amax
    if t <= Tc:
        progress = (t - Tw) / max(1, (Tc - Tw))
        return amin + 0.5 * (1.0 + math.cos(progress * math.pi)) * (amax - amin)
    return amin

def build_optimizer(model: torch.nn.Module, opt_cfg: OptimCfg):
    params = list(model.parameters())

    if opt_cfg.optimizer.lower() == "adamw":
        if AdamW is None:
            raise ImportError("Could not import your AdamW from AdamW.py")
        return AdamW(
            params,
            lr=opt_cfg.lr_max,
            betas=opt_cfg.betas,
            eps=opt_cfg.eps,
            weight_decay=opt_cfg.weight_decay,
        )

    if opt_cfg.optimizer.lower() == "sgd":
        if SGD is None:
            raise ImportError("Could not import your SGD from SGD.py")
        # 你的 SGD 只支持 lr
        return SGD(params, lr=opt_cfg.lr_max)

    raise ValueError(f"Unknown optimizer: {opt_cfg.optimizer} (use adamw/sgd)")




def round_up_to_multiple(x: int, multiple: int) -> int:
    if multiple <= 1:
        return x
    return ((x + multiple - 1) // multiple) * multiple


def swiglu_default_dff(d_model: int, multiple: int) -> int:
    # d_ff ≈ (8/3) * d_model, then round up for matmul efficiency
    raw = int(math.ceil((8.0 * d_model) / 3.0))
    return round_up_to_multiple(raw, multiple)


def compute_loss_with_user_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Use your cross_entropy.py:
      cross_entropy(logits: (*, V), targets: (*,))
    Here we feed logits (B,T,V) and targets (B,T) directly.
    """
    if user_cross_entropy is None:
        raise ImportError(f"Failed to import cross_entropy from cross_entropy.py")
    return user_cross_entropy(logits, targets)




@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
) -> Dict[str, float]:
    model.eval()
    out: Dict[str, float] = {}
    for split, tokens in [("train", train_tokens), ("val", val_tokens)]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(tokens, batch_size, context_length, device)
            logits = model(x)  # (B,T,V)
            loss = compute_loss_with_user_ce(logits, y)
            losses.append(float(loss.item()))
        out[split] = float(np.mean(losses))
    model.train()
    return out


# =========================
# Argparse
# =========================
def parse_args() -> Tuple[ModelCfg, OptimCfg, TrainCfg, int]:
    p = argparse.ArgumentParser("CS336 training loop (assembled from your components)")

    # data
    p.add_argument("--train_tokens", type=str, required=True, help="Path to 1D token ids (.npy or .bin)")
    p.add_argument("--val_tokens", type=str, required=True, help="Path to 1D token ids (.npy or .bin)")
    p.add_argument("--token_dtype", type=str, default="uint16", help="uint16/int32/int64 (for .bin)")

    # model core
    p.add_argument("--vocab_size", type=int, required=True)
    p.add_argument("--context_length", type=int, required=True)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--num_heads", type=int, default=6)

    # SwiGLU d_ff
    p.add_argument("--d_ff", type=int, default=-1, help="FFN hidden size. If -1, use SwiGLU default (ceil(8/3*d_model))")
    p.add_argument("--d_ff_multiple", type=int, default=64, help="Round d_ff up to this multiple (default 64)")

    # model extras (aligned with your TransformerLM)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--bias", action="store_true", help="Use bias in Linear layers (default True in your LM)")
    p.add_argument("--no_bias", action="store_true", help="Disable bias (overrides --bias)")
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument("--no_rope", action="store_true")
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--no_tie_weights", action="store_true")
    p.add_argument("--emb_dropout", type=float, default=-1.0, help="If -1, LM uses dropout as emb_dropout")

    # train
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--total_steps", type=int, default=20000)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--save_every", type=int, default=1000)

    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")

    # optim + schedule
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--lr_max", type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--cosine_end_step", type=int, default=-1, help="Tc; default=total_steps-1")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--optim_eps", type=float, default=1e-8)
    p.add_argument("--momentum", type=float, default=0.9)

    # system
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)

    # logging
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="cs336")
    p.add_argument("--wandb_name", type=str, default="run")

    args = p.parse_args()

    # d_ff
    if args.d_ff is None or args.d_ff < 0:
        d_ff = swiglu_default_dff(args.d_model, args.d_ff_multiple)
    else:
        d_ff = round_up_to_multiple(int(args.d_ff), int(args.d_ff_multiple))

    # bias flag resolution (your LM default bias=True)
    bias = True
    if args.no_bias:
        bias = False
    elif args.bias:
        bias = True

    use_rope = not args.no_rope
    tie_weights = not args.no_tie_weights

    emb_dropout = None if args.emb_dropout < 0 else float(args.emb_dropout)

    model_cfg = ModelCfg(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=d_ff,
        dropout=float(args.dropout),
        bias=bool(bias),
        eps=float(args.eps),
        rope_theta=float(args.rope_theta),
        use_rope=bool(use_rope),
        max_seq_len=int(args.max_seq_len),
        tie_weights=bool(tie_weights),
        emb_dropout=emb_dropout,
    )

    Tc = args.cosine_end_step if args.cosine_end_step >= 0 else (args.total_steps - 1)
    opt_cfg = OptimCfg(
        optimizer=args.optimizer,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        warmup_steps=args.warmup_steps,
        cosine_end_step=Tc,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.optim_eps,
        momentum=args.momentum,
    )

    train_cfg = TrainCfg(
        train_tokens_path=args.train_tokens,
        val_tokens_path=args.val_tokens,
        token_dtype=args.token_dtype,
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_iters=args.eval_iters,
        save_every=args.save_every,
        save_dir=args.save_dir,
        resume=args.resume,
        device=args.device,
        seed=args.seed,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )

    return model_cfg, opt_cfg, train_cfg, args.d_ff_multiple


# =========================
# Main
# =========================
def main() -> None:
    model_cfg, opt_cfg, train_cfg, dff_multiple = parse_args()
    set_seed(train_cfg.seed)

    device = train_cfg.device
    print(f"[device] {device}")

    # ---- load tokens with memmap ----
    np_dtype = _dtype_from_str(train_cfg.token_dtype)
    train_tokens = load_tokens_memmap(train_cfg.train_tokens_path, dtype=np_dtype)
    val_tokens = load_tokens_memmap(train_cfg.val_tokens_path, dtype=np_dtype)
    print(f"[data] train_tokens={train_tokens.shape} val_tokens={val_tokens.shape} dtype={train_tokens.dtype}")

    # ---- build model (aligned with your TransformerLM) ----
    print(f"[model] d_ff={model_cfg.d_ff} (SwiGLU default rounded to multiple={dff_multiple})")
    model = TransformerLM(
        vocab_size=model_cfg.vocab_size,
        context_length=model_cfg.context_length,
        num_layers=model_cfg.num_layers,
        d_model=model_cfg.d_model,
        num_heads=model_cfg.num_heads,
        d_ff=model_cfg.d_ff,
        dropout=model_cfg.dropout,
        bias=model_cfg.bias,
        eps=model_cfg.eps,
        rope_theta=model_cfg.rope_theta,
        use_rope=model_cfg.use_rope,
        max_seq_len=model_cfg.max_seq_len,
        tie_weights=model_cfg.tie_weights,
        emb_dropout=model_cfg.emb_dropout,
    ).to(device)

    # ---- optimizer ----
    optimizer = build_optimizer(model, opt_cfg)

    # ---- optional wandb ----
    wandb_run = None
    if train_cfg.wandb:
        try:
            import wandb  # type: ignore
            wandb_run = wandb.init(
                project=train_cfg.wandb_project,
                name=train_cfg.wandb_name,
                config={**asdict(model_cfg), **asdict(opt_cfg), **asdict(train_cfg)},
            )
        except Exception as e:
            print(f"[wandb] disabled (import/init failed): {e}")
            wandb_run = None

    # ---- resume ----
    start_step = 0
    if train_cfg.resume:
        if user_ckpt is None or not hasattr(user_ckpt, "load_checkpoint"):
            ckpt = torch.load(train_cfg.resume, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = int(ckpt["iteration"])
        else:
            start_step = int(
                user_ckpt.load_checkpoint(
                    src=train_cfg.resume,
                    model=model,
                    optimizer=optimizer,
                )
            )

        print(f"[resume] loaded from {train_cfg.resume} at iteration={start_step}")

    model.train()

    # ---- training loop ----
    t0 = time.time()
    running_loss = 0.0

    for step in range(start_step, train_cfg.total_steps):
        lr = compute_lr(step, opt_cfg)
        set_optimizer_lr(optimizer, lr)

        _zero_grad_compat(optimizer)

        loss_accum = 0.0
        for _micro in range(train_cfg.grad_accum_steps):
            x, y = get_batch(train_tokens, train_cfg.batch_size, model_cfg.context_length, device)
            logits = model(x)  # (B,T,V)

            loss = compute_loss_with_user_ce(logits, y) / train_cfg.grad_accum_steps
            loss.backward()
            loss_accum += float(loss.item())

        # clip grads
        if train_cfg.max_grad_norm > 0:
            if clip_grad_l2_norm_ is not None:
                clip_grad_l2_norm_(model.parameters(), max_norm=train_cfg.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.max_grad_norm)

        optimizer.step()

        running_loss += loss_accum

        # ---- logging ----
        if (step + 1) % train_cfg.log_every == 0:
            dt = time.time() - t0
            avg_loss = running_loss / train_cfg.log_every
            ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

            tok_per_step = train_cfg.batch_size * model_cfg.context_length * train_cfg.grad_accum_steps
            tok_s = tok_per_step * train_cfg.log_every / max(1e-9, dt)

            print(
                f"[step {step+1:>7}/{train_cfg.total_steps}] "
                f"lr={lr:.3e} loss={avg_loss:.4f} ppl={ppl:.2f} tok/s={tok_s:.1f}"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/lr": lr,
                        "train/loss": avg_loss,
                        "train/ppl": ppl,
                        "perf/tok_s": tok_s,
                        "step": step + 1,
                    },
                    step=step + 1,
                )

            running_loss = 0.0
            t0 = time.time()

        # ---- eval ----
        if (step + 1) % train_cfg.eval_every == 0:
            losses = estimate_loss(
                model=model,
                train_tokens=train_tokens,
                val_tokens=val_tokens,
                batch_size=train_cfg.batch_size,
                context_length=model_cfg.context_length,
                device=device,
                eval_iters=train_cfg.eval_iters,
            )
            train_loss = losses["train"]
            val_loss = losses["val"]
            train_ppl = math.exp(train_loss) if train_loss < 20 else float("inf")
            val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")

            print(
                f"[eval step {step+1}] "
                f"train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} | "
                f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/train_loss": train_loss,
                        "eval/val_loss": val_loss,
                        "eval/train_ppl": train_ppl,
                        "eval/val_ppl": val_ppl,
                        "step": step + 1,
                    },
                    step=step + 1,
                )

        # ---- checkpoint ----
        if (step + 1) % train_cfg.save_every == 0 or (step + 1) == train_cfg.total_steps:
            ckpt_path = os.path.join(train_cfg.save_dir, f"ckpt_step_{step + 1}.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

            if user_ckpt is None or not hasattr(user_ckpt, "save_checkpoint"):
                # fallback（万一没导入成功）
                torch.save(
                    {
                        "iteration": step + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    ckpt_path,
                )
            else:
                user_ckpt.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=step + 1,
                    out=ckpt_path,
                )

            print(f"[ckpt] saved to {ckpt_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

