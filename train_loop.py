# train_loop.py
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch

from checkpointing import save_checkpoint, load_checkpoint
from cross_entropy import cross_entropy as user_cross_entropy
from data_loader import load_tokenized_ids, get_batch
from learning_rate_schedule import lr_cosine_with_warmup
from transformer_lm import TransformerLM  # your GPT-style LM


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
    batch_size: int
    total_steps: int
    grad_accum_steps: int
    max_grad_norm: float
    log_every: int
    eval_every: int
    eval_iters: int
    save_every: int
    save_dir: str              # base output dir
    run_name: str              # used to create per-run dir
    resume: str
    device: str
    seed: int
    wandb: bool
    wandb_project: str
    wandb_name: str


# =========================
# Logging helpers
# =========================
class JSONLLogger:
    """Append-only JSONL logger: one dict per line, flushed immediately."""
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def log(self, record: Dict[str, Any]) -> None:
        self.f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass


def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# =========================
# Utilities
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _zero_grad_compat(optimizer: Any) -> None:
    if not hasattr(optimizer, "zero_grad"):
        return
    try:
        optimizer.zero_grad(set_to_none=True)
    except TypeError:
        optimizer.zero_grad()


def set_optimizer_lr(optimizer: Any, lr: float) -> None:
    # Most torch-like optimizers
    if hasattr(optimizer, "param_groups"):
        for pg in optimizer.param_groups:
            # some custom student optimizers may name lr as "alpha"
            if "alpha" in pg:
                pg["alpha"] = lr
            else:
                pg["lr"] = lr
        return
    # custom optimizers
    if hasattr(optimizer, "lr"):
        optimizer.lr = lr
        return
    if hasattr(optimizer, "set_lr"):
        optimizer.set_lr(lr)
        return
    raise AttributeError("Optimizer has no known way to set lr (param_groups / lr / set_lr).")


def compute_lr(step: int, opt_cfg: OptimCfg) -> float:
    return float(
        lr_cosine_with_warmup(
            t=step,
            alpha_max=opt_cfg.lr_max,
            alpha_min=opt_cfg.lr_min,
            Tw=opt_cfg.warmup_steps,
            Tc=opt_cfg.cosine_end_step,
        )
    )


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
            loss = user_cross_entropy(logits, y)  # per-token mean CE (unscaled)
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
    p.add_argument(
        "--train_tokens",
        type=str,
        required=True,
        help="Tokenized ids filename (under tokenized_ids/) or a full path to a .npy/.npz file.",
    )
    p.add_argument(
        "--val_tokens",
        type=str,
        required=True,
        help="Tokenized ids filename (under tokenized_ids/) or a full path to a .npy/.npz file.",
    )

    # model core
    p.add_argument("--vocab_size", type=int, required=True)
    p.add_argument("--context_length", type=int, required=True)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=16)

    # SwiGLU d_ff
    p.add_argument(
        "--d_ff",
        type=int,
        default=-1,
        help="FFN hidden size. If -1, use SwiGLU default (ceil(8/3*d_model))",
    )
    p.add_argument(
        "--d_ff_multiple",
        type=int,
        default=64,
        help="Round d_ff up to this multiple (default 64)",
    )

    # model extras (aligned with your TransformerLM)
    p.add_argument("--dropout", type=float, default=0.0)
    # bias flags: default True; --no_bias turns it off; --bias keeps it on (explicit, optional)
    p.add_argument("--bias", action="store_true", help="Explicitly enable bias (default: enabled)")
    p.add_argument("--no_bias", action="store_true", help="Disable bias in Linear layers")
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument("--no_rope", action="store_true")
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--no_tie_weights", action="store_true")
    p.add_argument("--emb_dropout", type=float, default=-1.0, help="If -1, LM uses dropout as emb_dropout")

    # train
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--total_steps", type=int, default=2000)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--save_dir", type=str, default="runs", help="Base directory for outputs (each run makes a subdir)")
    p.add_argument("--run_name", type=str, default="run", help="Run name (used for output subdir naming)")
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)

    # optimizer + schedule
    p.add_argument("--optimizer", type=str, default="adamw", help="adamw/sgd")
    p.add_argument("--lr_max", type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--cosine_end_step", type=int, default=-1, help="Tc (if -1, uses total_steps-1)")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--optim_eps", type=float, default=1e-8)
    p.add_argument("--momentum", type=float, default=0.0)

    # wandb
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="cs336")
    p.add_argument("--wandb_name", type=str, default="run")

    args = p.parse_args()

    dff_multiple = int(args.d_ff_multiple)
    if int(args.d_ff) < 0:
        d_ff = swiglu_default_dff(int(args.d_model), multiple=dff_multiple)
    else:
        d_ff = int(args.d_ff)

    # flags
    bias = True
    if bool(args.no_bias):
        bias = False
    # args.bias is kept for explicitness but default is already True

    use_rope = False if bool(args.no_rope) else True
    tie_weights = False if bool(args.no_tie_weights) else True
    emb_dropout = None if float(args.emb_dropout) < 0 else float(args.emb_dropout)

    model_cfg = ModelCfg(
        vocab_size=int(args.vocab_size),
        context_length=int(args.context_length),
        num_layers=int(args.num_layers),
        d_model=int(args.d_model),
        num_heads=int(args.num_heads),
        d_ff=int(d_ff),
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
        cosine_end_step=int(Tc),
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.optim_eps,
        momentum=args.momentum,
    )

    train_cfg = TrainCfg(
        train_tokens_path=args.train_tokens,
        val_tokens_path=args.val_tokens,
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_iters=args.eval_iters,
        save_every=args.save_every,
        save_dir=args.save_dir,
        run_name=args.run_name,
        resume=args.resume,
        device=args.device,
        seed=args.seed,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )

    return model_cfg, opt_cfg, train_cfg, dff_multiple


# =========================
# Main
# =========================
def main() -> None:
    model_cfg, opt_cfg, train_cfg, dff_multiple = parse_args()
    set_seed(train_cfg.seed)

    device = train_cfg.device
    print(f"[device] {device}")

    # ---- create per-run output dir ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(train_cfg.save_dir, f"{train_cfg.run_name}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    config_path = os.path.join(run_dir, "config.json")
    print(f"[run_dir] {run_dir}")
    print(f"[logs] metrics -> {metrics_path}")
    print(f"[logs] config  -> {config_path}")

    logger = JSONLLogger(metrics_path)

    # ---- save config once ----
    full_config: Dict[str, Any] = {
        "time_created": now_iso(),
        "cmd": " ".join(sys.argv),
        "run_dir": run_dir,
        "model_cfg": asdict(model_cfg),
        "opt_cfg": asdict(opt_cfg),
        "train_cfg": asdict(train_cfg),
    }
    write_json(config_path, full_config)

    # ---- load tokens with memmap ----
    train_root = os.path.dirname(train_cfg.train_tokens_path) or "tokenized_ids"
    train_name = os.path.basename(train_cfg.train_tokens_path)
    val_root = os.path.dirname(train_cfg.val_tokens_path) or "tokenized_ids"
    val_name = os.path.basename(train_cfg.val_tokens_path)

    train_tokens = load_tokenized_ids(train_name, root_dir=train_root, mmap=True)
    val_tokens = load_tokenized_ids(val_name, root_dir=val_root, mmap=True)
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
                config={
                    **asdict(model_cfg),
                    **asdict(opt_cfg),
                    **asdict(train_cfg),
                    "run_dir": run_dir,
                },
            )
        except Exception as e:
            print(f"[wandb] disabled (import/init failed): {e}")
            wandb_run = None

    # ---- resume ----
    start_step = 0
    if train_cfg.resume:
        start_step = int(load_checkpoint(src=train_cfg.resume, model=model, optimizer=optimizer))
        print(f"[resume] loaded from {train_cfg.resume} at iteration={start_step}")

    model.train()

    # ---- training loop timers ----
    run_start_t = time.perf_counter()  # for elapsed wallclock
    interval_t0 = time.perf_counter()  # for throughput in each logging interval

    # bookkeeping for correct loss logging under grad accumulation
    running_loss = 0.0  # sum of per-update (unscaled) losses over log interval

    # tokens processed per *parameter update step* (one optimizer.step)
    tokens_per_update = train_cfg.batch_size * model_cfg.context_length * train_cfg.grad_accum_steps

    # log run start event
    logger.log(
        {
            "event": "start",
            "time": now_iso(),
            "step": start_step,
            "elapsed_s": 0.0,
            "tokens_per_update": int(tokens_per_update),
        }
    )

    for step in range(start_step, train_cfg.total_steps):
        lr = compute_lr(step, opt_cfg)
        set_optimizer_lr(optimizer, lr)

        _zero_grad_compat(optimizer)

        # ---- grad accumulation ----
        micro_loss_sum = 0.0  # sum of *unscaled* micro losses
        for _micro in range(train_cfg.grad_accum_steps):
            x, y = get_batch(train_tokens, train_cfg.batch_size, model_cfg.context_length, device)
            logits = model(x)  # (B,T,V)

            loss_unscaled = user_cross_entropy(logits, y)  # correct per-token mean CE
            (loss_unscaled / train_cfg.grad_accum_steps).backward()
            micro_loss_sum += float(loss_unscaled.item())

        # per-update loss = average over microbatches (still per-token mean CE)
        step_loss = micro_loss_sum / max(1, train_cfg.grad_accum_steps)

        # clip grads
        if train_cfg.max_grad_norm > 0:
            if clip_grad_l2_norm_ is not None:
                clip_grad_l2_norm_(model.parameters(), max_norm=train_cfg.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.max_grad_norm)

        optimizer.step()
        running_loss += step_loss

        # ---- logging ----
        if (step + 1) % train_cfg.log_every == 0:
            interval_dt = time.perf_counter() - interval_t0
            elapsed_s = time.perf_counter() - run_start_t

            avg_loss = running_loss / train_cfg.log_every
            ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

            tok_s = (tokens_per_update * train_cfg.log_every) / max(1e-9, interval_dt)
            tokens_seen = (step + 1) * tokens_per_update

            msg = (
                f"[step {step+1:>7}/{train_cfg.total_steps}] "
                f"loss={avg_loss:.4f} ppl={ppl:.2f} "
                f"lr={lr:.3e} tok/s={tok_s:,.0f} "
                f"elapsed={elapsed_s:.1f}s"
            )
            print(msg)

            rec = {
                "event": "train",
                "time": now_iso(),
                "step": step + 1,
                "elapsed_s": float(elapsed_s),
                "interval_s": float(interval_dt),
                "train/loss": float(avg_loss),
                "train/ppl": float(ppl),
                "train/lr": float(lr),
                "train/tok_s": float(tok_s),
                "train/tokens_seen": int(tokens_seen),
            }
            logger.log(rec)

            if wandb_run is not None:
                wandb_run.log(rec, step=step + 1)

            running_loss = 0.0
            interval_t0 = time.perf_counter()

        # ---- eval ----
        if train_cfg.eval_every > 0 and ((step + 1) % train_cfg.eval_every == 0):
            eval_t0 = time.perf_counter()
            losses = estimate_loss(
                model=model,
                train_tokens=train_tokens,
                val_tokens=val_tokens,
                batch_size=train_cfg.batch_size,
                context_length=model_cfg.context_length,
                device=device,
                eval_iters=train_cfg.eval_iters,
            )
            eval_dt = time.perf_counter() - eval_t0
            elapsed_s = time.perf_counter() - run_start_t

            train_loss = losses["train"]
            val_loss = losses["val"]
            train_ppl = math.exp(train_loss) if train_loss < 20 else float("inf")
            val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")

            print(
                f"[eval step {step+1}] "
                f"train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} | "
                f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} "
                f"(eval_time={eval_dt:.2f}s, elapsed={elapsed_s:.1f}s)"
            )

            rec = {
                "event": "eval",
                "time": now_iso(),
                "step": step + 1,
                "elapsed_s": float(elapsed_s),
                "eval/seconds": float(eval_dt),
                "eval/train_loss": float(train_loss),
                "eval/val_loss": float(val_loss),
                "eval/train_ppl": float(train_ppl),
                "eval/val_ppl": float(val_ppl),
            }
            logger.log(rec)

            if wandb_run is not None:
                wandb_run.log(rec, step=step + 1)

        # ---- checkpoint ----
        if (step + 1) % train_cfg.save_every == 0 or (step + 1) == train_cfg.total_steps:
            ckpt_path = os.path.join(run_dir, f"ckpt_step_{step + 1}.pt")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=step + 1,
                out=ckpt_path,
            )
            elapsed_s = time.perf_counter() - run_start_t
            print(f"[ckpt] saved to {ckpt_path}")

            rec = {
                "event": "ckpt",
                "time": now_iso(),
                "step": step + 1,
                "elapsed_s": float(elapsed_s),
                "ckpt_path": ckpt_path,
            }
            logger.log(rec)
            if wandb_run is not None:
                wandb_run.log(rec, step=step + 1)

    # end
    elapsed_s = time.perf_counter() - run_start_t
    logger.log({"event": "end", "time": now_iso(), "step": train_cfg.total_steps, "elapsed_s": float(elapsed_s)})
    logger.close()

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

