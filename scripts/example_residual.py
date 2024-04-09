"""
@author: Fabian Schaipp

Implements Example 5.3 in

    Frank E. Curtis and Michael L. Overton, A sequential quadratic programming
    algorithm for nonconvex, nonsmooth constrained optimization,
    SIAM Journal on Optimization 2012 22:2, 474-500, https://doi.org/10.1137/090780201.

Useful script for testing and performance optimization.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.fft import dct

from ncopt.functions import ObjectiveOrConstraint
from ncopt.functions.norm_residual import NormResidual
from ncopt.sqpgs import SQPGS

# %%
np.random.seed(1234)

d = 90  # problem dimension
m = 10  # number of samples
q = 1.0  # residual norm order


device = "cuda" if torch.cuda.is_available() else "cpu"

# %% Objective function: ||x||_q

Id, zeros = torch.eye(d, d), torch.zeros(d)
obj = NormResidual(params=(Id, zeros), q=q, offset=0.0)

obj.to(device)

# %% Constraint: ||Rx-y|| <= delta

num_zeros = int(0.9 * d)
oracle = np.concatenate((np.zeros(num_zeros), np.random.randn(d - num_zeros)))
np.random.shuffle(oracle)

# first m rows of discrete dxd cosine transformation matrix
R = torch.from_numpy(dct(np.eye(d), axis=0)[:m, :]).type(torch.float32)
y = (R @ oracle).type(torch.float32)
delta = 1.0

const = NormResidual(params=(R, y), q=2, offset=delta)
const.to(device)

assert np.allclose(
    const.forward(torch.from_numpy(oracle).type(torch.float32).reshape(1, -1)).item(), -delta
)

# %% Set up problem

f = ObjectiveOrConstraint(obj, dim=d)

gI = [ObjectiveOrConstraint(const, dim_out=1)]
gE = []

problem = SQPGS(f, gI, gE, x0=None, tol=1e-6, max_iter=200, verbose=True)

# %% Solve

np.random.seed(123)
torch.manual_seed(123)
x = problem.solve()

# %% Plotting

fig, ax = problem.plot_timings()
fig, ax = problem.plot_metrics()

fig, ax = plt.subplots()
# %%