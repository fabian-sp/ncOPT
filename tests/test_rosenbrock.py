"""
author: Fabian Schaipp
"""

import numpy as np
import torch

from ncopt.functions import ObjectiveOrConstraint
from ncopt.functions.max_linear import MaxOfLinear
from ncopt.functions.rosenbrock import NonsmoothRosenbrock
from ncopt.sqpgs.main import SQPGS

f = ObjectiveOrConstraint(NonsmoothRosenbrock(a=8.0), dim=2)
g = MaxOfLinear(
    params=(torch.diag(torch.tensor([torch.sqrt(torch.tensor(2.0)), 2.0])), -torch.ones(2))
)


def test_rosenbrock_from_zero():
    torch.manual_seed(1)
    gI = [ObjectiveOrConstraint(g, dim_out=1)]
    gE = []
    xstar = np.array([1 / np.sqrt(2), 0.5])
    problem = SQPGS(f, gI, gE, tol=1e-8, max_iter=200, verbose=False)
    x_k = problem.solve()
    np.testing.assert_array_almost_equal(x_k, xstar, decimal=4)


def test_rosenbrock_from_rand():
    torch.manual_seed(1)
    gI = [ObjectiveOrConstraint(g, dim_out=1)]
    gE = []
    xstar = np.array([1 / np.sqrt(2), 0.5])
    rng = np.random.default_rng(0)
    x0 = rng.random(2)
    problem = SQPGS(f, gI, gE, x0=x0, tol=1e-8, max_iter=200, verbose=False)
    x_k = problem.solve()
    np.testing.assert_array_almost_equal(x_k, xstar, decimal=4)


def test_rosenbrock_with_eq():
    torch.manual_seed(12)
    g1 = ObjectiveOrConstraint(torch.nn.Linear(2, 2), dim_out=2)
    g1.model.weight.data = torch.eye(2)
    g1.model.bias.data = -torch.ones(2)
    gI = []
    gE = [g1]
    xstar = np.ones(2)
    problem = SQPGS(f, gI, gE, tol=1e-8, max_iter=200, verbose=False)
    x_k = problem.solve()
    np.testing.assert_array_almost_equal(x_k, xstar, decimal=4)
