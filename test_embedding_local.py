import torch
from Embedding import Embedding
from adapters import run_embedding


def test_embedding_lookup_matches_rows():
    # V=5, D=3
    w = torch.tensor([
        [0., 0., 0.],   # id 0
        [1., 1., 1.],   # id 1
        [2., 2., 2.],   # id 2
        [3., 3., 3.],   # id 3
        [4., 4., 4.],   # id 4
    ])

    token_ids = torch.tensor([[1, 3, 1],
                              [0, 2, 4]])

    y = run_embedding(token_ids, w)

    # 期望输出形状 (2, 3, 3)
    assert y.shape == (2, 3, 3)

    # 检查几个位置是否等于对应行
    assert torch.allclose(y[0, 0], w[1])
    assert torch.allclose(y[0, 1], w[3])
    assert torch.allclose(y[1, 2], w[4])


def test_embedding_class_direct():
    emb = Embedding(5, 3)
    with torch.no_grad():
        emb.weight.copy_(torch.arange(15.).view(5, 3))
    token_ids = torch.tensor([4, 0, 2])
    y = emb(token_ids)
    assert torch.allclose(y[0], emb.weight[4])
    assert torch.allclose(y[1], emb.weight[0])
    assert torch.allclose(y[2], emb.weight[2])

