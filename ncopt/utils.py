import torch
from torch.autograd.functional import jacobian
from torch.func import vmap, jacrev, jacfwd, functional_call

import logging
from typing import Optional

#%% Computing Jacobians
"""
Important: jacobian and jacrev do the forward pass for each row of the input, that is,
WITHOUT the batch dimension!

E.g. if input has shape (d1, d2), then normal forward pas has input
(b, d1, d2); but jacobian/jacrev will do the forward pass with a (d1,d2) tensor.

This becomes an issue when there are reshape/view modules, because they often will
have different results when there is no batch dimension.

* If the forward method uses rehshape/view, the batch dimension should be
    specified with -1, and not with x.shape[0] or similar!
* For the Jacobian, we get an extra dimension in such cases --> needs to be
     removed later on
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

def compute_batch_jacobian_naive(model: torch.nn.Module, inputs: torch.Tensor):
    """

    Parameters
    ----------
    model : torch.nn.Module
        The function of which to compute the Jacobian.
    inputs : torch.Tensor
        The inputs for model. First dimension should be batch dimension.
    """
    b = inputs.size(0)
    # want to have batch dimension --> double brackets
    return torch.stack([jacobian(model, inputs[i]) for i in range(b)])
 
def compute_batch_jacobian_vmap(model: torch.nn.Module, inputs: torch.Tensor, 
                                forward: bool=False):
    """

    Parameters
    ----------
    model : torch.nn.Module
        The function of which to compute the Jacobian.
    inputs : torch.Tensor
        The inputs for model. First dimension should be batch dimension.
    forward: bool. 
        Whether to compute in forward mode (jacrev or jacfwd). By default False.
    """
    params = dict(model.named_parameters())

    def fmodel(params, inputs): #functional version of model
        return functional_call(model, params, inputs)

    # argnums specifies which argument to compute jacobian wrt
    # in_dims: dont map over params (None), map over first dim of inputs (0)
    if not forward:
        return vmap(jacrev(fmodel, argnums=(1)), in_dims=(None,0))(params, inputs)
    else:
        return vmap(jacfwd(fmodel, argnums=(1)), in_dims=(None,0))(params, inputs)

#%%
# copied from https://github.com/aaronpmishkin/experiment_utils/blob/main/src/experiment_utils/utils.py#L298
def get_logger(
    name: str,
    verbose: bool=False,
    debug: bool=False,
    log_file: Optional[str]=None,
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
