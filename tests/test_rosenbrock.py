"""
author: Fabian Schaipp
"""

import numpy as np

from ncopt.funs import f_rosenbrock, g_linear, g_max
from ncopt.sqpgs.main import SQPGS

f = f_rosenbrock(w=8.0)
g = g_max()

np.random.seed(12345)


def test_rosenbrock_from_zero():
    gI = [g]
    gE = []
    xstar = np.array([1 / np.sqrt(2), 0.5])
    problem = SQPGS(f, gI, gE, tol=1e-8, max_iter=200, verbose=False)
    x_k = problem.solve()
    np.testing.assert_array_almost_equal(x_k, xstar, decimal=4)


def test_rosenbrock_from_rand():
    gI = [g]
    gE = []
    xstar = np.array([1 / np.sqrt(2), 0.5])
    rng = np.random.default_rng(0)
    x0 = rng.random(2)
    problem = SQPGS(f, gI, gE, x0=x0, tol=1e-8, max_iter=200, verbose=False)
    x_k = problem.solve()
    np.testing.assert_array_almost_equal(x_k, xstar, decimal=4)


def test_rosenbrock_with_eq():
    g1 = g_linear(A=np.eye(2), b=np.ones(2))
    gI = []
    gE = [g1]
    xstar = np.ones(2)
    problem = SQPGS(f, gI, gE, tol=1e-8, max_iter=200, verbose=False)
    x_k = problem.solve()
    np.testing.assert_array_almost_equal(x_k, xstar, decimal=4)
