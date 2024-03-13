from typing import Callable, Optional

import torch


class ObjectiveOrConstraint(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        name: Optional[str] = None,
        prepare_inputs: Optional[Callable] = None,
        is_differentiable: bool = False,
    ):
        super().__init__()

        # TODO: Load from checkpoint if this is a string
        self.model = model
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
