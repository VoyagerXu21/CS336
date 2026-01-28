import torch

from Linear import Linear
from adapters import run_linear  # 你适配器里应该暴露这个函数名


def test_linear_forward_matches_matmul():
    m = Linear(3, 2)
    with torch.no_grad():
        m.W.copy_(torch.tensor([[1., 2.],
                                [3., 4.],
                                [5., 6.]]))
    x = torch.tensor([[10., 20., 30.],
                      [1., 2., 3.]])
    y = m(x)
    expected = torch.tensor([[220., 280.],
                             [22., 28.]])
    assert torch.allclose(y, expected)


def test_adapter_loads_weights():
    x = torch.tensor([[10., 20., 30.],
                      [1., 2., 3.]])

    # 模拟 nn.Linear.weight 的形状 (out, in) = (2, 3)
    weight_out_in = torch.tensor([[1., 3., 5.],
                                  [2., 4., 6.]])

    y = run_linear(x, weight_out_in)
    expected = torch.tensor([[220., 280.],
                             [22., 28.]])
    assert torch.allclose(y, expected)

