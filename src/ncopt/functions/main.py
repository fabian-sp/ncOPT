import torch


class ObjectiveOrConstraint(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, name: str = None, prepare_inputs=None):
        super().__init__()

        self.prepare_inputs = prepare_inputs
        self.model = model
        # Optional: load from checkpoint if this is a string

        self.name = name

        # Go into eval mode
        self.model.eval()
        self.eval()

        return

    def forward(self, x: torch.tensor):
        x = self.prepare_inputs(x) if self.prepare_inputs else x
        out = self.model.forward(x)
        return out
