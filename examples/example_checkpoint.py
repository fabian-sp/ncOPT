"""
Illustrates how to use a pretrained neural network as constraint function.

We load a checkpoint, that has been trained with the script in scripts/train_max_fun.py.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from ncopt.functions import ObjectiveOrConstraint
from ncopt.functions.max_linear import MaxOfLinear
from ncopt.functions.quadratic import Quadratic
from ncopt.sqpgs import SQPGS

# %% Load the checkpoint
checkpoint_dir = "../data/checkpoints/max2d.pt"

model = MaxOfLinear(input_dim=2, output_dim=2)

checkpoint = torch.load(checkpoint_dir)
model.load_state_dict(checkpoint["model_state_dict"])

print("Weights:", model.linear.weight)
print("Bias:", model.linear.bias)


# %% Problem setup

# Define the constraint
g = ObjectiveOrConstraint(model, dim_out=1)

# Define the objective: f(x) = 0.5*||x-(1,1)||^2
params = (torch.eye(2), -torch.ones(2), torch.tensor(1.0))
f = ObjectiveOrConstraint(Quadratic(params=params), dim=2, is_differentiable=True)

problem = SQPGS(f, [g], [], x0=None, tol=1e-10, max_iter=500, verbose=True)
x = problem.solve()

print("Final iterate: ", x)

# %% Plot the feasible region

_x, _y = np.linspace(-2, 2, 100), np.linspace(-2, 2, 100)
X, Y = np.meshgrid(_x, _y)
Z = np.zeros_like(X)

for j in np.arange(X.shape[0]):
    for i in np.arange(X.shape[1]):
        Z[i, j] = g.single_eval(np.array([X[i, j], Y[i, j]]))

Z[Z > 0] = np.nan  # only show feasible set

fig, ax = plt.subplots(figsize=(4, 4))

# Plot contour of feasible set
ax.contourf(X, Y, Z, cmap="Blues", levels=np.linspace(-4, 0, 20), antialiased=True, lw=0, zorder=0)

# plot circle
ax.scatter(1, 1, marker="o", s=50, c="k")
dist_to_ones = np.linalg.norm(x - np.ones(2))
c = plt.Circle((1, 1), radius=dist_to_ones, facecolor=None, fill=False, edgecolor="grey")
ax.add_patch(c)
# plot final iterate
ax.scatter(x[0], x[1], marker="o", s=50, c="silver", label="Final iterate")


ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.legend(loc="upper left")
ax.set_title("Minimize distance to black dot")

fig.tight_layout()
fig.savefig("../data/img/checkpoint.png")
