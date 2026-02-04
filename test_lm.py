from __future__ import annotations
from adapters import run_transformer_lm

model = run_transformer_lm(
    vocab_size=10000,         # 你 tokenizer 的词表大小
    context_length=256,
    num_layers=4,
    d_model=512,
    num_heads=16,
    d_ff=1408,               # 典型 SwiGLU 取值（≈8/3*512 后对齐）
    dropout=0.0,
    use_rope=True,
    tie_weights=True,
    eps=1e-5,
    verbose=True,            # 打印配置 & 参数量
    check_forward=True,      # 可选：跑一遍小 forward 检查 shape
    forward_shape=(2, 16),   # 可选：检查用的 (B,T)
)

