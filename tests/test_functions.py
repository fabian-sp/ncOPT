import torch

from ncopt.functions.quadratic import Quadratic
from ncopt.functions.rosenbrock import NonsmoothRosenbrock
from ncopt.utils import compute_batch_jacobian_vmap

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


def test_rosenbrock():
    inputs = torch.tensor(
        [
            [1.0866, 0.9348],
            [0.2838, 0.1436],
            [-0.4327, -1.1307],
            [-0.4813, -0.9599],
            [-0.4523, 0.7851],
        ],
        dtype=torch.float64,
    )

    expected_out = torch.tensor([[1.9746, 1.0174, 12.5960, 11.7266, 6.7533]], dtype=torch.float64).T

    expected_jac = torch.tensor(
        [[17.5588, -8.0], [-5.9732, 8.0], [-9.7886, -8.0], [-10.6634, -8.0], [4.3322, 8.0]],
        dtype=torch.float64,
    ).view(5, 1, 2)

    model = NonsmoothRosenbrock(a=8.0)
    jac, out = compute_batch_jacobian_vmap(model, inputs)

    assert torch.allclose(out, expected_out, atol=1e-3)
    assert torch.allclose(jac, expected_jac)

    return
