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
    ) -> None:
        """Wrapper for objective and constraint functions.

        The function g(x) is represented by ``model`` which computes g(x) when given x as input.
        Important:
            * The mapping ``model`` should be compatible with batched inputs,
                where the first dimension is the batch dimension.
            * The output of ``model`` must be of shape ``(batch_size, dim_out)``,
                where ``dim_out`` must be one if the function is used as objective.
                In particular, if the output shape is ``(batch_size, )``,
                we don't guarantee that the solver will work.

        Note that we set ``model`` into ``.eval()`` mode during initialization.

        Parameters
        ----------
        model : torch.nn.Module
            The Pytorch module for evaluating the function.
        dim : Optional[int], optional
            Input dimension of the function, by default None.
            Needs to be specified for objective functions for the solver.
        dim_out : int, optional
            Output dimension of the function, by default 1.
            Needs to be specified for each constraint function for the solver.
        name : Optional[str], optional
            A name for the function, by default None.
            Not needed for solver, only for convenience.
        device : Optional[Union[str, torch.device]], optional
            A device for the forward pass, by default None.
            You normally don't need to specify this: we automatically use the device
            where the parameters of the model lie on.
            If you specify the device, please make sure, that the parameters of the model
            lie on the correct device.
        dtype : Optional[torch.dtype], optional
            Format to convert when evaluating a tensor that was converted
            from ``np.ndarray``, by default torch.float32.
        prepare_inputs : Optional[Callable], optional
            Callable to prepare the model inpt, by default None.
            Note that the solver always inputs the full optimization variable (as batch).
            If your function only needs a subset of the vector as input, or
            needs reshaping operations etc, then you can do this via this callable.
            Note that the callable should be compatible with batched tensors.
        is_differentiable : bool, optional
            Whether the function is (continuously) differentiable, by default False.
            If ``True``, then no extra points will be sampled for this function.

        """
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

        # If no device is provided, set it to the same as the model parameters
        # if model has no parameters, we set device to cpu
        if not self.device:
            if sum(p.numel() for p in model.parameters() if p.requires_grad) > 0:
                devices = {p.device for p in model.parameters()}
                if len(devices) == 1:
                    self.device = devices.pop()
                else:
                    raise KeyError(
                        "Model parameters lie on more than one device. Currently not supported."
                    )
            else:
                self.device = torch.device("cpu")

        # Go into eval mode
        self.model.eval()
        self.eval()

        return

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Method for evaluating the function.
        This is a simple wrapper around ``model.forward()``.
        We apply the ``prepare_inputs`` method, and move ``x`` to the correct device.

        Parameters
        ----------
        x : torch.tensor
            Batched input tensor. This will of shape ``(batch_size, dim)``,
            where ``dim`` is the dimension of the optimization variable.
            Note that ``x`` can be cropped or reshaped via ``prepare_inputs`` if necessary.

        Returns
        -------
        torch.tensor
            Output of the function. Of shape ``(batch_size, dim_out)``.
        """
        x = self.prepare_inputs(x) if self.prepare_inputs else x
        out = self.model.forward(x.to(self.device))
        return out

    def single_eval(self, x: Union[torch.tensor, np.ndarray]) -> np.ndarray:
        """Convenience function for (only!) evaluating at a single point.
        This is needed for the Armijo line search in SQP-GS.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.dtype)

        with torch.no_grad():
            out = self.forward(x.reshape(1, -1))

        return out.squeeze(dim=0).detach().cpu().numpy()
