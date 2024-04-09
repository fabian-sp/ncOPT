import torch


class NormResidual(torch.nn.Module):
    """A residual error mapping. Implements the function

    x --> ||Ax-b||_q^p - offset
    """

    def __init__(
        self,
        input_dim: int = None,
        output_dim: int = None,
        params: tuple = None,
        q: float = 2.0,
        p: float = 1.0,
        offset: float = 1.0,
        dtype=torch.float32,
    ):
        """
        Either specify input_dim and output_dim, or specify params in the form
        (A, b) where A and b are both a torch.tensor.

        Parameters
        ----------
        input_dim : int, optional
            Input dimension of the linear layer, by default None
        output_dim : int, optional
            Output dimension of the linear layer, by default None
        params : tuple, optional
            If you want to fix the linear mapping, specify as (A, b).
            By default None.
        q : float, optional
            Order of the norm, by default 2.0 (standard Euclidean norm).
        q : float, optional
            Power of the residual, by default 1.0 .
        offset : float, optional
            A constant value to offset the function, by default 1.0.
        dtype : _type_, optional
            Will convert params to this type for the linear layer, by default torch.float32
        """
        super().__init__()

        assert q > 0, f"Order must be positive, but is given as {q}."
        assert p > 0, f"Power must be positive, but is given as {p}."

        self.p = p
        self.q = q
        self.offset = offset

        assert params is not None or (
            input_dim is not None and output_dim is not None
        ), "Specify either dimensions or parameters."

        if params is not None:
            assert (
                len(params) == 2
            ), f"params should contains two elements (A,b), but contains only {len(params)}."
            output_dim, input_dim = params[0].shape
            assert params[1].shape == torch.Size([output_dim]), "Shape of bias term does not match."

        self.linear = torch.nn.Linear(input_dim, output_dim)

        # Set the weights if the mapping is given
        # Default type of torch.nn.Linear is float32
        if params is not None:
            self.linear.weight.data = params[0].type(dtype)
            self.linear.bias.data = (-1) * params[1].type(dtype)

        return

    def forward(self, x):
        x = self.linear(x)
        x = torch.linalg.norm(x, ord=self.q, dim=1, keepdim=True) ** (self.p) - self.offset
        return x
