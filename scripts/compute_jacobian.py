import torch
from torch.autograd.functional import jacobian

d = 10
m = 3
b = 4


class Quadratic(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn((input_dim, input_dim)))

    def forward(self, x):
        return (1/2) * x @ self.A @ x

def compute_batch_jacobian_naive(model: torch.nn.Module, inputs: torch.Tensor):
    """Function for computing the Jacobian of model(inputs), with respect to inputs.
    Assume inputs has shape (batch, **d), where d itself could be a tuple. 
    Assume that model maps tensors of shape d to tensors of shape m (with m integer).

    Then the Jacobian of model has shape (m,d). 
    The output of this function should have shape (b,m,d)

    Parameters
    ----------
    model : torch.nn.Module
        The function of which to compute the Jacobian.
    inputs : torch.Tensor
        The inputs for model. First dimension should be batch dimension.
    """
    b = inputs.size(0)
    return torch.stack([jacobian(model, inputs[i]) for i in range(b)])


def test_quadratic_jacobian():
    torch.manual_seed(1)
    inputs = torch.randn(b,d)
    model = Quadratic(d)

    # f(x) = 0.5 x^T A x --> Df(x) = 0.5*(A+A.T)x
    expected = 0.5 * inputs @ (model.A.T + model.A) 
    jac =  compute_batch_jacobian_naive(model, inputs)   

    assert torch.allclose(expected, jac, rtol=1e-5, atol=1e-5)

    return


# TODO: test for multi-dim output, and d being a tuple (eg use convolution, softmax)
#%% multi-dim output

model = torch.nn.Sequential(torch.nn.Linear(d,m), 
                            torch.nn.Softmax())

inputs = torch.randn(b,d)
output = model(inputs)


jac = compute_batch_jacobian_naive(model, inputs)

assert jac.shape == torch.Size([b,m,d])