import torch


class MaxOfLinear(torch.nn.Module):
    """A composition of the maximum function with an affine mapping. Implements the mapping

        x --> max(Ax + b)

    where the maximum is taken over the components of Ax + b.
    """

    def __init__(
        self,
        input_dim: int = None,
        output_dim: int = None,
        params: tuple = None,
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
        dtype : _type_, optional
            Will convert params to this type for the linear layer, by default torch.float32
        """
        super().__init__()

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
            self.linear.bias.data = params[1].type(dtype)

        return

    def forward(self, x):
        x = self.linear(x)
        # make sure to have output shape [batch_size, 1] by keepdim=True
        x, _ = torch.max(x, dim=-1, keepdim=True)
        return x
