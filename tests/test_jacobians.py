import pytest
import torch

from ncopt.functions.quadratic import Quadratic
from ncopt.utils import (
    compute_batch_jacobian,
    compute_batch_jacobian_naive,
    compute_batch_jacobian_vmap,
)

d = 10
m = 3
b = 4


class DummyNet(torch.nn.Module):
    def __init__(self, d=7, C=1, num_classes=10):
        super(DummyNet, self).__init__()

        self.conv = torch.nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2)
        self.linear_input_dim = 8 * C * d * d
        self.linear = torch.nn.Linear(self.linear_input_dim, num_classes)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv(x))
        # x = x.view(x.shape[0], -1) # This would result in errors when computing Jacobian.
        x = x.view(-1, self.linear_input_dim)  # Batch dimension specified by -1.
        x = self.linear(x)
        return x


######################################################
# Tests start here


@pytest.mark.parametrize(
    "jac_fun", [compute_batch_jacobian_naive, compute_batch_jacobian_vmap, compute_batch_jacobian]
)
def test_linear_jacobian(jac_fun):
    torch.manual_seed(1)
    inputs = torch.randn(b, d)
    model = torch.nn.Linear(d, m)

    expected = torch.stack([model.weight.data for _ in range(b)])
    expected_out = model(inputs)

    jac, out = jac_fun(model, inputs)

    assert torch.allclose(expected, jac, rtol=1e-5, atol=1e-5)
    assert torch.allclose(expected_out, out)
    assert jac.shape == torch.Size([b, m, d])

    return


@pytest.mark.parametrize(
    "jac_fun", [compute_batch_jacobian_naive, compute_batch_jacobian_vmap, compute_batch_jacobian]
)
def test_quadratic_jacobian(jac_fun):
    torch.manual_seed(1)
    inputs = torch.randn(b, d)
    model = Quadratic(input_dim=d)

    # f(x) = 0.5 x^T A x + b.T x + c --> Df(x) = 0.5*(A+A.T)x + b
    expected = 0.5 * inputs @ (model.A.T + model.A) + model.b
    expected = expected.view(b, 1, d)
    expected_out = model(inputs)

    jac, out = jac_fun(model, inputs)

    assert torch.allclose(expected, jac, rtol=1e-5, atol=1e-5)
    assert torch.allclose(expected_out, out)
    assert jac.shape == torch.Size([b, 1, d])

    return


@pytest.mark.parametrize("jac_fun", [compute_batch_jacobian_vmap, compute_batch_jacobian])
def test_forward_backward_jacobian(jac_fun):
    torch.manual_seed(1)
    inputs = torch.randn(b, d)
    model = Quadratic(d)

    expected = 0.5 * inputs @ (model.A.T + model.A) + model.b
    expected = expected.view(b, 1, d)

    jac1, out1 = jac_fun(model, inputs, forward=False)
    jac2, out2 = jac_fun(model, inputs, forward=True)

    assert torch.allclose(expected, jac1, rtol=1e-5, atol=1e-5)
    assert torch.allclose(jac1, jac2, rtol=1e-5, atol=1e-5)
    assert torch.allclose(out1, out2, rtol=1e-5, atol=1e-5)

    return


def test_multidim_output():
    model = torch.nn.Sequential(torch.nn.Linear(d, m), torch.nn.Softmax(dim=-1))

    inputs = torch.randn(b, d)
    output = model(inputs)
    assert output.shape == torch.Size([b, m])

    jac1, out1 = compute_batch_jacobian_naive(model, inputs)
    jac2, out2 = compute_batch_jacobian_vmap(model, inputs)

    assert jac1.shape == torch.Size([b, m, d])
    assert jac2.shape == torch.Size([b, m, d])

    assert torch.allclose(jac1, jac2, rtol=1e-5, atol=1e-5)

    return


# TODO: Adapt
# def test_multidim_output_multiaxis_input():
#     pixel = 7
#     channel = 1
#     num_classes = 9
#     input_dim = (channel, pixel, pixel)
#     inputs = torch.randn(b, *input_dim)

#     model = DummyNet(d=pixel, C=channel, num_classes=num_classes)
#     output = model(inputs)

#     assert output.shape == torch.Size([b, num_classes])

#     jac1 = compute_batch_jacobian_naive(model, inputs)
#     jac2 = compute_batch_jacobian_vmap(model, inputs)

#     # this case has an extra dimension, due to the view operation
#     assert jac1.shape == torch.Size([b, 1, num_classes, *input_dim])
#     assert jac2.shape == torch.Size([b, 1, num_classes, *input_dim])

#     assert torch.allclose(jac1, jac2, rtol=1e-5, atol=1e-5)
