import torch

from ncopt.functions.quadratic import Quadratic

d = 10
b = 4


def test_quadratic():
    params = (torch.eye(d), torch.zeros(d), torch.tensor(1.0))
    model = Quadratic(params=params)
    inputs = torch.randn(b, d)

    expected = 0.5 * (inputs * inputs).sum(1) + 1.0
    out = model(inputs)

    assert torch.allclose(out, expected[:, None])
    assert out.shape == torch.Size([b, 1])
    return
