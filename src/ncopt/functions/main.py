from typing import Callable, Optional, Union

import numpy as np
import torch


class ObjectiveOrConstraint(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        dim: Optional[int] = None,
        dim_out: int = 1,
        name: Optional[str] = None,
        prepare_inputs: Optional[Callable] = None,
        is_differentiable: bool = False,
    ):
        super().__init__()

        # TODO: Load from checkpoint if this is a string
        self.model = model
        self.dim = dim
        self.dim_out = dim_out
        self.name = name
        self.prepare_inputs = prepare_inputs
        self.is_differentiable = is_differentiable

        # Go into eval mode
        self.model.eval()
        self.eval()

        return

    def forward(self, x: torch.tensor):
        x = self.prepare_inputs(x) if self.prepare_inputs else x
        out = self.model.forward(x)
        return out

    def single_eval(self, x: Union[torch.tensor, np.ndarray]):
        """Convenience function for (only!) evaluating at a single point.
        This is needed for the Armijo line search in SQPGS.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        with torch.no_grad():
            out = self.forward(x.reshape(1, -1))

        return out.squeeze(dim=0).numpy()
