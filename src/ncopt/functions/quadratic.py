import torch


class Quadratic(torch.nn.Module):
    def __init__(self, input_dim: int = None, params: tuple = None):
        """Implements the function

            x --> (1/2)*x.T*A*x + b.T*x + c

        Parameters
        ----------
        input_dim : int, optional
            If no params are specified this specifies the dimension of the tensors, by default None
        params : tuple, optional
           3-Tuple of tensors, by default None.
           If specified, should contain the values of A, b and c.
        """
        super().__init__()

        assert params is not None or (
            input_dim is not None
        ), "Specify either a dimension or parameters"
        if params is not None:
            assert (
                len(params) == 3
            ), f"params should contains three elements (A,b,c), but contains only {len(params)}."

        if params is None:
            self.A = torch.nn.Parameter(torch.randn(input_dim, input_dim))
            self.b = torch.nn.Parameter(
                torch.randn(
                    input_dim,
                )
            )
            self.c = torch.nn.Parameter(
                torch.randn(
                    1,
                )
            )
        else:
            self.A = torch.nn.Parameter(params[0])
            self.b = torch.nn.Parameter(params[1])
            self.c = torch.nn.Parameter(params[2])

    def forward(self, x: torch.tensor) -> torch.tensor:
        """

        Parameters
        ----------
        x : torch.tensor
            Should be batched, and have shape [batch_size, input_dim]

        Returns
        -------
        torch.tensor of shape [batch_size, 1]
        """
        out = 0.5 * torch.sum((x @ self.A) * x, dim=1, keepdim=True)
        out = out + (x * self.b).sum(dim=1, keepdim=True)
        out = out + self.c
        return out
