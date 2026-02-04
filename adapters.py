# adapters.py
from __future__ import annotations

from Linear import Linear

from Embedding import Embedding

from RMSNorm import RMSNorm

from positionwise_feedforward import PositionwiseFeedForward

from rope import RotaryPositionalEmbedding

from softmax import softmax

from scaled_dot_product_attention import scaled_dot_product_attention

from transformer_block import TransformerBlock

from typing import Any, Dict, Optional, Tuple

from transformer_lm import TransformerLM

from learning_rate_schedule import lr_cosine_with_warmup

from typing import Iterable

from typing import Tuple
import numpy as np

from typing import BinaryIO, IO, Union
import os
import torch

from checkpointing import save_checkpoint, load_checkpoint


PathOrFile = Union[str, os.PathLike, BinaryIO, IO[bytes]]


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: PathOrFile,
) -> None:
    """
    Adapter wrapper required by the assignment.
    """
    save_checkpoint(model=model, optimizer=optimizer, iteration=iteration, out=out)


def run_load_checkpoint(
    src: PathOrFile,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Adapter wrapper required by the assignment.
    """
    return load_checkpoint(src=src, model=model, optimizer=optimizer)



def run_get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch from a 1D token id sequence.

    Args:
        x: 1D numpy array of token ids, shape (n,), integer dtype
        batch_size: B
        context_length: T
        device: 'cpu' | 'cuda:0' | 'mps' | ...

    Returns:
        x_batch: (B, T) torch.long on device
        y_batch: (B, T) torch.long on device, where y is x shifted by 1
    """
    if not isinstance(x, (np.ndarray, np.memmap)):
        raise TypeError(f"x must be a numpy array or memmap, got {type(x)}")
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape={x.shape}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if context_length <= 0:
        raise ValueError(f"context_length must be > 0, got {context_length}")
    if not np.issubdtype(x.dtype, np.integer):
        raise TypeError(f"x must contain integer token ids, got dtype={x.dtype}")

    n = int(x.shape[0])
    max_start = n - (context_length + 1)  # need i+T and i+T+1 to be valid
    if max_start < 0:
        raise ValueError(
            f"Sequence too short: len(x)={n}, context_length={context_length} "
            f"(need at least context_length+1 tokens)"
        )

    # sample B start indices uniformly from [0, max_start]
    starts = np.random.randint(0, max_start + 1, size=(batch_size,), dtype=np.int64)

    # build batches in numpy (works for memmap too)
    xb = np.empty((batch_size, context_length), dtype=np.int64)
    yb = np.empty((batch_size, context_length), dtype=np.int64)

    for b, i in enumerate(starts):
        i = int(i)
        xb[b, :] = x[i : i + context_length]
        yb[b, :] = x[i + 1 : i + context_length + 1]

    # convert to torch and move to device
    x_batch = torch.from_numpy(xb).to(device=device, dtype=torch.long)
    y_batch = torch.from_numpy(yb).to(device=device, dtype=torch.long)
    return x_batch, y_batch



def run_gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    eps: float = 1e-6,
) -> float:
    """
    Adapter required by the assignment.

    It should call your implementation and return the (pre-clip) norm.
    """
    # 兼容两种项目结构：
    # 1) adapters.py 和 gradient_clipping.py 同级：from gradient_clipping import ...
    # 2) adapters.py 在包内：from .gradient_clipping import ...
    try:
        from .gradient_clipping import clip_grad_l2_norm_
    except Exception:
        from gradient_clipping import clip_grad_l2_norm_

    return clip_grad_l2_norm_(parameters, max_norm=max_norm, eps=eps)

def get_lr_cosine_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    Tw: int,
    Tc: int,
) -> float:
    """
    Adapter API expected by the assignment/tests.
    Delegates to the actual implementation in learning_rate_schedule.py
    """
    return lr_cosine_with_warmup(t=t, alpha_max=alpha_max, alpha_min=alpha_min, Tw=Tw, Tc=Tc)


from cross_entropy import cross_entropy


def run_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Assignment entrypoint. Delegates to cross_entropy.cross_entropy.
    """
    return cross_entropy(logits, targets)




def _count_params_unique(model: torch.nn.Module) -> int:
    """
    Count parameters without double-counting shared Parameters (e.g., weight tying).
    """
    seen = set()
    total = 0
    for p in model.parameters():
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        total += p.numel()
    return total


def run_transformer_lm(
    *args,
    verbose: bool = True,
    check_forward: bool = False,
    forward_shape: Tuple[int, int] = (2, 8),   # (B,T) for quick check
    device: Optional[str] = None,              # override device for quick check only
    **kwargs,
) -> TransformerLM:
    """
    Test adapter for TransformerLM, plus optional reporting.

    Supported calls:
      1) run_transformer_lm(config_dict)
      2) run_transformer_lm(**config_kwargs)

    Extra adapter-only kwargs (won't be passed into TransformerLM):
      - verbose: print config + param counts
      - check_forward: run a tiny forward to verify logits shape
      - forward_shape: (B,T) used for check_forward
      - device: device string for check_forward (e.g., "cuda"); if None uses model's device

    Returns:
      TransformerLM instance.
    """
    # -------- parse config --------
    if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
        cfg: Dict[str, Any] = dict(args[0])  # copy
    elif len(args) == 0:
        cfg = dict(kwargs)  # copy
    else:
        raise TypeError("run_transformer_lm only supports run_transformer_lm(config_dict) or run_transformer_lm(**kwargs)")

    # Pop adapter-only keys if user mistakenly included them inside cfg
    cfg.pop("verbose", None)
    cfg.pop("check_forward", None)
    cfg.pop("forward_shape", None)
    cfg.pop("device", None)

    # -------- build model --------
    model = TransformerLM(**cfg)

    # -------- report --------
    if verbose:
        # Pull the most relevant hyperparams (if missing, show None)
        def g(k: str, default=None):
            return cfg.get(k, default)

        total_params = _count_params_unique(model)
        tied = bool(getattr(model, "tie_weights", False))

        print("\n[TransformerLM config]")
        print(f"  vocab_size     = {g('vocab_size')}")
        print(f"  context_length = {g('context_length')}")
        print(f"  num_layers     = {g('num_layers')}")
        print(f"  d_model        = {g('d_model')}")
        print(f"  num_heads      = {g('num_heads')}")
        print(f"  d_ff           = {g('d_ff')}")
        print(f"  dropout        = {g('dropout', 0.0)}")
        print(f"  use_rope       = {g('use_rope', True)}")
        print(f"  tie_weights    = {tied}")
        print(f"[Params] unique params = {total_params:,}  (~{total_params/1e6:.2f} M)\n")

    # -------- optional quick forward check --------
    if check_forward:
        V = int(cfg["vocab_size"])
        B, T = forward_shape
        # choose device for check
        if device is not None:
            model = model.to(device)
            dev = torch.device(device)
        else:
            dev = next(model.parameters()).device

        x = torch.randint(0, V, (B, T), dtype=torch.long, device=dev)
        logits = model(x)
        print(f"[Forward check] input: {(B,T)}  logits: {tuple(logits.shape)} (expected {(B,T,V)})")

    return model





def run_transformer_block(
    x: torch.Tensor,
    d_model: int,
    num_heads: int,
    d_ff: int,
    *,
    token_positions: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    bias: bool = True,
    eps: float = 1e-5,
    rope_theta: float = 10000.0,
    use_rope: bool = True,
    max_seq_len: int = 4096,
    seed: int = 0,
) -> torch.Tensor:
    """
    Adapter for tests.
    x: (B,T,d_model)
    """
    torch.manual_seed(int(seed))

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        bias=bias,
        eps=eps,
        rope_theta=rope_theta,
        use_rope=use_rope,
        max_seq_len=max_seq_len,
        device=x.device,
        dtype=x.dtype,
    )

    block.eval()
    with torch.no_grad():
        y = block(x, token_positions=token_positions)
    return y




def run_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Test adapter: the autograder calls this function.
    We simply forward to your implementation.
    """
    return scaled_dot_product_attention(q, k, v, mask=mask)

def run_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Test adapter: calls student's softmax implementation.
    """
    return softmax(x, dim)


def run_rope(x: torch.Tensor, token_positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """
    Test adapter: construct RoPE module and apply it.
    x: (..., seq_len, d_k)
    token_positions: (..., seq_len)
    """
    d_k = int(x.size(-1))
    # Make sure cache is large enough for given token_positions
    max_pos = int(token_positions.max().item()) if token_positions.numel() > 0 else 0
    max_seq_len = max(max_pos + 1, int(x.size(-2)))

    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=x.device)
    return rope(x, token_positions)


@torch.no_grad()
def run_swiglu(
    x: torch.Tensor,
    *,
    dropout: float = 0.0,
    multiple_of: int = 64,
    bias: bool = True,
) -> torch.Tensor:
    """
    Test adapter for SwiGLU FFN.

    Expected by tests: adapters.run_swiglu(x, ...) -> y
    where x shape is (..., d_model) and y shape is (..., d_model).
    """
    d_model = x.shape[-1]
    ffn = PositionwiseFeedForward(
        d_model=d_model,
        dropout=dropout,
        multiple_of=multiple_of,
        bias=bias,
    ).to(device=x.device, dtype=x.dtype)

    return ffn(x)



def run_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    x: (..., d_model)
    weight: (d_model,)  测试给的 gamma
    """
    d_model = weight.numel()
    m = RMSNorm(d_model=d_model, eps=eps, device=weight.device, dtype=weight.dtype)

    with torch.no_grad():
        m.weight.copy_(weight)

    return m(x)

def run_embedding(token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    测试用适配器：
    - weight: 形状 (num_embeddings, embedding_dim)
    - token_ids: 任意形状整数张量
    返回 embedding 输出
    """
    num_embeddings, embedding_dim = weight.shape

    emb = Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        device=weight.device,
        dtype=weight.dtype,
    )

    # 把测试给的权重加载进你的模块
    with torch.no_grad():
        emb.weight.copy_(weight)

    return emb(token_ids)

def run_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    测试适配器示例：
    - x: 输入张量 (..., in_features)
    - weight: 测试给定的权重（可能是 W 或 W^T）
    返回 Linear(x) 的输出
    """

    in_features = x.size(-1)

    # 允许 weight 两种常见形状：
    # 1) (in_features, out_features)  -> 正好是我们要的 W
    # 2) (out_features, in_features)  -> 这是 nn.Linear.weight，需要转置成 W
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D, got shape={tuple(weight.shape)}")

    if weight.shape[0] == in_features:
        # weight 已经是 W
        out_features = weight.shape[1]
        W_to_load = weight
    elif weight.shape[1] == in_features:
        # weight 是 W^T（nn.Linear.weight）
        out_features = weight.shape[0]
        W_to_load = weight.t().contiguous()
    else:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} incompatible with in_features={in_features}"
        )

    # 建模块（把参数放到和 x 一样的设备/类型，或者用显式传入的 device/dtype）
    dev = device if device is not None else x.device
    dt = dtype if dtype is not None else x.dtype

    m = Linear(in_features, out_features, device=dev, dtype=dt)

    # 用 load_state_dict 把权重塞进去（题目建议）
    state = {"W": W_to_load.to(device=dev, dtype=dt)}
    m.load_state_dict(state, strict=True)

    return m(x)

