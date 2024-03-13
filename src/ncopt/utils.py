import logging
from typing import Optional, Tuple

import torch
from torch.autograd.functional import jacobian
from torch.func import jacfwd, jacrev, vmap

# %% Computing Jacobians
"""
Important: jacobian and jacrev do the forward pass for each row of the input, that is,
WITHOUT the batch dimension!

E.g. if input has shape (d1, d2), then normal forward pas has input
(b, d1, d2); but jacobian/jacrev will do the forward pass with a (d1,d2) tensor.

This becomes an issue when there are reshape/view modules, because they often will
have different results when there is no batch dimension.

So we fix this by adding dummy dimensions!
"""

""" 

    Functions for computing the Jacobian of model(inputs) wrt. inputs.
    Assume inputs has shape (batch, *d), where d itself could be a tuple. 
    Assume that model output has shape m (with m integer).

    Then the Jacobian of model has shape (m,d). 
    The output of the below functions should have 
        - either shape (b,m,*d),
        - or shape (b,1,m,*d) (see above for explanation).

    For more info see:
        * https://discuss.pytorch.org/t/computing-batch-jacobian-efficiently/80771/11
        * https://pytorch.org/tutorials/intermediate/jacobians_hessians.html
"""


def compute_batch_jacobian_naive(
    model: torch.nn.Module, inputs: torch.Tensor
) -> Tuple[torch.tensor, torch.tensor]:
    """Naive way for computing Jacobian. Used for testing.

    Parameters
    ----------
    model : torch.nn.Module
        The function of which to compute the Jacobian.
    inputs : torch.Tensor
        The inputs for model. First dimension should be batch dimension.

    Returns
    -------
    tuple[torch.tensor, torch.tensor]
        Jacobian and output.
    """
    b = inputs.size(0)
    out = model.forward(inputs)
    # want to have batch dimension --> double brackets
    jac = torch.stack([jacobian(model, inputs[[i]]) for i in range(b)])
    # Now jac has shape [b, 1, out_dim, 1, in_dim]  --> squeeze
    # This only works if output shape is scalar/vector.
    return jac.squeeze(dim=(1, 3)), out


def compute_batch_jacobian_vmap(
    model: torch.nn.Module, inputs: torch.Tensor, forward: bool = False
):
    """Vmap over batch dimension. This has the issue that the inputs are given to
    model.forward() without the first dimension. We counteract by adding a dummy dimension.

    Parameters
    ----------
    model : torch.nn.Module
        The function of which to compute the Jacobian.
    inputs : torch.Tensor
        The inputs for model. First dimension should be batch dimension.
    forward: bool.
        Whether to compute in forward mode (jacrev or jacfwd). By default False.

    Returns
    -------
    tuple[torch.tensor, torch.tensor]
        Jacobian and output.
    """

    # functional version of model; dummy dim because vmap removes batch_dim
    def fmodel(model, inputs):
        out = model(inputs[None, :])
        return out, out

    # argnums specifies which argument to compute jacobian wrt
    # in_dims: dont map over params (None), map over first dim of inputs (0)
    if not forward:
        jac, out = vmap(jacrev(fmodel, argnums=(1), has_aux=True), in_dims=(None, 0))(model, inputs)
    else:
        jac, out = vmap(jacfwd(fmodel, argnums=(1), has_aux=True), in_dims=(None, 0))(model, inputs)

    # now remove dummy dimension again
    return jac.squeeze(dim=1), out.squeeze(dim=1)


def compute_batch_jacobian(
    model: torch.nn.Module, inputs: torch.Tensor, forward: bool = False
) -> Tuple[torch.tensor, torch.tensor]:
    """Not using vmap. This results in the Jacobian being of shape
    [batch_size, dim_out, batch_size, *dim_in]

    Parameters
    ----------
    model : torch.nn.Module
        The function of which to compute the Jacobian.
    inputs : torch.Tensor
        The inputs for model. First dimension should be batch dimension.
    forward: bool.
        Whether to compute in forward mode (jacrev or jacfwd). By default False.

    Returns
    -------
    tuple[torch.tensor, torch.tensor]
        Jacobian and output.
    """

    def fmodel(model, inputs):  # functional version of model
        out = model(inputs)
        return out, out

    # argnums specifies which argument to compute jacobian wrt
    if not forward:
        jac, out = jacrev(fmodel, argnums=(1), has_aux=True)(model, inputs)
    else:
        jac, out = jacfwd(fmodel, argnums=(1), has_aux=True)(model, inputs)

    # Now only take the "diagonal". This only works if output shape is scalar/vector.
    jac = torch.stack([jac[i, :, i, :] for i in range(jac.shape[0])])
    return jac, out


# %%
# copied from https://github.com/aaronpmishkin/experiment_utils/blob/main/src/experiment_utils/utils.py#L298
def get_logger(
    name: str,
    verbose: bool = False,
    debug: bool = False,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Construct a logging.Logger instance with an appropriate configuration.

    Params:
        name: name for the Logger instance.
        verbose: (optional) whether or not the logger should print verbosely
            (ie. at the `INFO` level).
        debug: (optional) whether or not the logger should print in debug mode
            (ie. at the `DEBUG` level).
        log_file: (optional) path to a file where the log should be stored. The
            log is printed to `stdout` when `None`.

     Returns:
        Instance of logging.Logger.
    """

    level = logging.WARNING
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO

    logging.basicConfig(level=level, filename=log_file)
    logger = logging.getLogger(name)
    logging.root.setLevel(level)
    logger.setLevel(level)
    return logger
