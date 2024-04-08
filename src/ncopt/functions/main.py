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
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = torch.float32,
        prepare_inputs: Optional[Callable] = None,
        is_differentiable: bool = False,
    ):
        super().__init__()

        # TODO: Load from checkpoint if this is a string
        self.model = model
        self.dim = dim
        self.dim_out = dim_out
        self.name = name
        self.device = device
        self.dtype = dtype
        self.prepare_inputs = prepare_inputs
        self.is_differentiable = is_differentiable

        # If no device is provided, set it to the same as the first model parameter
        # this might fail for distributed models
        # if model has no parameters, we set device to cpu
        if not self.device:
            if sum(p.numel() for p in model.parameters() if p.requires_grad) > 0:
                self.device = next(model.parameters()).device
            else:
                self.device = torch.device("cpu")

        # Go into eval mode
        self.model.eval()
        self.eval()

        return

    def forward(self, x: torch.tensor):
        x = self.prepare_inputs(x) if self.prepare_inputs else x
        out = self.model.forward(x.to(self.device))
        return out

    def single_eval(self, x: Union[torch.tensor, np.ndarray]):
        """Convenience function for (only!) evaluating at a single point.
        This is needed for the Armijo line search in SQPGS.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.dtype)

        with torch.no_grad():
            out = self.forward(x.reshape(1, -1))

        return out.squeeze(dim=0).detach().cpu().numpy()
