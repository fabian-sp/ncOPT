import torch


class NonsmoothRosenbrock(torch.nn.Module):
    """A 2D nonsmooth Rosenbrock function, given by

    x --> a*|x_1^2 - x_2| + (1-x_1)^2

    """

    def __init__(self, a: float = 8):
        super().__init__()

        self.a = a
        return

    def forward(self, x: torch.tensor) -> torch.tensor:
        """x should be of shape (batch_size, 2)"""
        out = self.a * torch.abs(x[:, 0] ** 2 - x[:, 1])
        out += (1 - x[:, 0]) ** 2
        out = out[:, None]  # need output shape (batch_size, 1)
        return out
