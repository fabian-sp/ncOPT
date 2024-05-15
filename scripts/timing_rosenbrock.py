"""
author: Fabian Schaipp
"""

import timeit

import numpy as np
import torch

from ncopt.functions import ObjectiveOrConstraint
from ncopt.functions.max_linear import MaxOfLinear
from ncopt.functions.rosenbrock import NonsmoothRosenbrock
from ncopt.sqpgs import SQPGS

# %%
f = ObjectiveOrConstraint(NonsmoothRosenbrock(a=8.0), dim=2)
g = MaxOfLinear(
    params=(torch.diag(torch.tensor([torch.sqrt(torch.tensor(2.0)), 2.0])), -torch.ones(2))
)


# inequality constraints (list of functions)
gI = [ObjectiveOrConstraint(g, dim_out=1)]

xstar = np.array([1 / np.sqrt(2), 0.5])

np.random.seed(31)
x0 = np.random.randn(2)

# %% Timing one scalar inequality constraint

# equality constraints (list of scalar functions)
gE = []
problem = SQPGS(f, gI, gE, x0, tol=1e-20, max_iter=100, verbose=False)

timeit.timeit("x_k = problem.solve()", number=20)

# result (start of refactoring):
# 168 ms ± 2.96 ms per loop (mean ± std. dev. of 7 runs, 20 loops each)

# %% Timing equality constraint

g1 = ObjectiveOrConstraint(torch.nn.Linear(2, 2), dim_out=2)
g1.model.weight.data = torch.eye(2)
g1.model.bias.data = torch.zeros(2)

# equality constraints (list of scalar functions)
gE = [g1]
problem = SQPGS(f, gI, gE, x0, tol=1e-20, max_iter=100, verbose=False)

np.random.seed(31)
x0 = np.random.randn(2)

timeit.timeit("x_k = problem.solve()", number=10)

# result (start of refactoring):
# 291 ms ± 3.53 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
