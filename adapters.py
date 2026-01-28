# adapters.py
from __future__ import annotations

from typing import Optional


from Linear import Linear



from Embedding import Embedding


import torch
from RMSNorm import RMSNorm


from positionwise_feedforward import PositionwiseFeedForward

from rope import RotaryPositionalEmbedding


import torch
from softmax import softmax   # 如果你的文件路径不同，这里改成你的实际 import


from typing import Optional
import torch

from scaled_dot_product_attention import scaled_dot_product_attention


from typing import Optional
import torch

from transformer_block import TransformerBlock


from typing import Any, Dict

from transformer_lm import TransformerLM


def run_transformer_lm(*args, **kwargs) -> TransformerLM:
    """
    Test adapter for TransformerLM.

    Robust handling:
      - run_transformer_lm(config_dict)
      - run_transformer_lm(**config_kwargs)

    Expected keys at minimum:
      vocab_size, context_length, num_layers,
      d_model, num_heads, d_ff
    plus any TransformerBlock args (dropout, bias, eps, rope_theta, use_rope, max_seq_len, device, dtype, ...).
    """
    if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
        cfg: Dict[str, Any] = args[0]
        return TransformerLM(**cfg)

    if len(args) != 0:
        raise TypeError(
            "run_transformer_lm only supports run_transformer_lm(config_dict) or run_transformer_lm(**kwargs)"
        )

    return TransformerLM(**kwargs)



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

