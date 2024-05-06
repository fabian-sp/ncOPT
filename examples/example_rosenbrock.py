"""
Implements Example 5.1 in

    Frank E. Curtis and Michael L. Overton, A sequential quadratic programming
    algorithm for nonconvex, nonsmooth constrained optimization,
    SIAM Journal on Optimization 2012 22:2, 474-500, https://doi.org/10.1137/090780201.

"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from ncopt.functions import ObjectiveOrConstraint
from ncopt.functions.max_linear import MaxOfLinear
from ncopt.functions.rosenbrock import NonsmoothRosenbrock
from ncopt.sqpgs import SQPGS

# %% Setup

f = ObjectiveOrConstraint(NonsmoothRosenbrock(a=8.0), dim=2)
g = MaxOfLinear(
    params=(torch.diag(torch.tensor([torch.sqrt(torch.tensor(2.0)), 2.0])), -torch.ones(2))
)

# inequality constraints (list of functions)
gI = [ObjectiveOrConstraint(g, dim_out=1)]

# equality constraints
gE = []

# Optional equality constraint (for testing and education purpose)
# g2 = ObjectiveOrConstraint(torch.nn.Linear(2, 2), dim_out=2)
# g2.model.weight.data = torch.eye(2)
# g2.model.bias.data = torch.zeros(2)
# gE = [g2]

xstar = np.array([1 / np.sqrt(2), 0.5])  # solution

# %% How to use the solver

problem = SQPGS(f, gI, gE, x0=None, tol=1e-10, max_iter=100, verbose=True)
x = problem.solve()

print("Distance to solution:", f"{np.linalg.norm(x - xstar):.9f}")

# %% Solve from multiple starting points and plot

_x, _y = np.linspace(-2, 2, 100), np.linspace(-2, 2, 100)
X, Y = np.meshgrid(_x, _y)
Z = np.zeros_like(X)

for j in np.arange(X.shape[0]):
    for i in np.arange(X.shape[1]):
        Z[i, j] = f.single_eval(np.array([X[i, j], Y[i, j]]))

# %%
np.random.seed(1)
torch.manual_seed(1)

fig, ax = plt.subplots(figsize=(5, 4))

# Plot contour and solution
ax.contourf(X, Y, Z, cmap="gist_heat", levels=20, alpha=0.7, antialiased=True, lw=0, zorder=0)
ax.scatter(xstar[0], xstar[1], marker="*", s=200, c="gold", zorder=1)

for i in range(7):
    x0 = np.random.randn(2)
    problem = SQPGS(f, gI, gE, x0, tol=1e-10, max_iter=100, verbose=False, store_history=True)
    x_k = problem.solve()
    print(x_k)

    x_hist = problem.x_hist
    ax.plot(x_hist[:, 0], x_hist[:, 1], c="silver", lw=1, ls="--", alpha=0.5, zorder=2)
    ax.scatter(x_k[0], x_k[1], marker="+", s=50, c="k", zorder=3)
    ax.scatter(x0[0], x0[1], marker="o", s=30, c="silver", zorder=3)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

fig.suptitle("Trajectory for multiple starting points")

legend_elements = [
    Line2D([0], [0], marker="*", lw=0, color="gold", label="Solution", markersize=15),
    Line2D([0], [0], marker="o", lw=0, color="silver", label="Starting point", markersize=8),
    Line2D([0], [0], marker="+", lw=0, color="k", label="Final iterate", markersize=8),
]
ax.legend(handles=legend_elements, ncol=3, fontsize=8)

fig.tight_layout()
fig.savefig("../data/img/rosenbrock.png")
