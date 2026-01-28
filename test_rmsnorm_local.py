# test_rmsnorm_local.py
import pytest
import torch

from RMSNorm import RMSNorm

# 如果你在 adapters.py 里实现了 run_rmsnorm，可以顺便测一下适配器
try:
    from adapters import run_rmsnorm
    HAS_ADAPTER = True
except Exception:
    HAS_ADAPTER = False


def ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """参考实现：严格按作业描述（先 float32 计算，再 cast 回原 dtype）"""
    orig_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    w_fp32 = weight.to(torch.float32)

    mean_sq = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_sq + eps)
    y = x_fp32 * inv_rms * w_fp32  # 广播到 (..., d_model)

    return y.to(orig_dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_matches_reference(dtype):
    # 有些 CPU 环境对 bfloat16/float16 支持不完整，但我们实现里会 upcast 到 float32，所以一般没问题
    torch.manual_seed(0)
    device = torch.device("cpu")

    d_model = 8
    x = torch.randn(2, 3, d_model, device=device, dtype=dtype)
    weight = torch.randn(d_model, device=device, dtype=dtype)
    eps = 1e-5

    m = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
    with torch.no_grad():
        m.weight.copy_(weight)

    y = m(x)
    y_ref = ref_rmsnorm(x, weight, eps=eps)

    # 输出 dtype 必须和输入一致
    assert y.dtype == x.dtype

    # 数值匹配（因为内部 upcast，误差通常很小）
    assert torch.allclose(y.to(torch.float32), y_ref.to(torch.float32), atol=1e-5, rtol=1e-5)


def test_rmsnorm_rms_is_one_when_weight_is_one():
    torch.manual_seed(1)
    device = torch.device("cpu")
    d_model = 16
    x = torch.randn(4, 5, d_model, device=device, dtype=torch.float16)

    m = RMSNorm(d_model=d_model, eps=1e-5, device=device, dtype=torch.float16)
    with torch.no_grad():
        m.weight.fill_(1.0)

    y = m(x).to(torch.float32)

    # RMS(y) ≈ 1：mean(y^2) over last dim should be close to 1
    rms2 = y.pow(2).mean(dim=-1)  # shape (4,5)
    assert torch.allclose(rms2, torch.ones_like(rms2), atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not HAS_ADAPTER, reason="adapters.run_rmsnorm not found")
def test_adapter_run_rmsnorm():
    torch.manual_seed(2)
    device = torch.device("cpu")
    d_model = 10
    x = torch.randn(2, 1, d_model, device=device, dtype=torch.float16)
    w = torch.randn(d_model, device=device, dtype=torch.float16)

    y = run_rmsnorm(x, w, eps=1e-5)
    y_ref = ref_rmsnorm(x, w, eps=1e-5)

    assert torch.allclose(y.to(torch.float32), y_ref.to(torch.float32), atol=1e-5, rtol=1e-5)
# test_rmsnorm_local.py
import pytest
import torch

from RMSNorm import RMSNorm

# 如果你在 adapters.py 里实现了 run_rmsnorm，可以顺便测一下适配器
try:
    from adapters import run_rmsnorm
    HAS_ADAPTER = True
except Exception:
    HAS_ADAPTER = False


def ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """参考实现：严格按作业描述（先 float32 计算，再 cast 回原 dtype）"""
    orig_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    w_fp32 = weight.to(torch.float32)

    mean_sq = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_sq + eps)
    y = x_fp32 * inv_rms * w_fp32  # 广播到 (..., d_model)

    return y.to(orig_dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_matches_reference(dtype):
    # 有些 CPU 环境对 bfloat16/float16 支持不完整，但我们实现里会 upcast 到 float32，所以一般没问题
    torch.manual_seed(0)
    device = torch.device("cpu")

    d_model = 8
    x = torch.randn(2, 3, d_model, device=device, dtype=dtype)
    weight = torch.randn(d_model, device=device, dtype=dtype)
    eps = 1e-5

    m = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
    with torch.no_grad():
        m.weight.copy_(weight)

    y = m(x)
    y_ref = ref_rmsnorm(x, weight, eps=eps)

    # 输出 dtype 必须和输入一致
    assert y.dtype == x.dtype

    # 数值匹配（因为内部 upcast，误差通常很小）
    assert torch.allclose(y.to(torch.float32), y_ref.to(torch.float32), atol=1e-5, rtol=1e-5)


def test_rmsnorm_rms_is_one_when_weight_is_one():
    torch.manual_seed(1)
    device = torch.device("cpu")
    d_model = 16
    x = torch.randn(4, 5, d_model, device=device, dtype=torch.float16)

    m = RMSNorm(d_model=d_model, eps=1e-5, device=device, dtype=torch.float16)
    with torch.no_grad():
        m.weight.fill_(1.0)

    y = m(x).to(torch.float32)

    # RMS(y) ≈ 1：mean(y^2) over last dim should be close to 1
    rms2 = y.pow(2).mean(dim=-1)  # shape (4,5)
    assert torch.allclose(rms2, torch.ones_like(rms2), atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not HAS_ADAPTER, reason="adapters.run_rmsnorm not found")
def test_adapter_run_rmsnorm():
    torch.manual_seed(2)
    device = torch.device("cpu")
    d_model = 10
    x = torch.randn(2, 1, d_model, device=device, dtype=torch.float16)
    w = torch.randn(d_model, device=device, dtype=torch.float16)

    y = run_rmsnorm(x, w, eps=1e-5)
    y_ref = ref_rmsnorm(x, w, eps=1e-5)

    assert torch.allclose(y.to(torch.float32), y_ref.to(torch.float32), atol=1e-5, rtol=1e-5)

