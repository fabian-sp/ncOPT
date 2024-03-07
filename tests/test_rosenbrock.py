"""
author: Fabian Schaipp
"""

import numpy as np

from ncopt.funs import f_rosenbrock, g_linear, g_max
from ncopt.sqpgs.main import SQPGS

f = f_rosenbrock()
g = g_max()


def test_rosenbrock_from_zero():
    gI = [g]
    gE = []
    xstar = np.array([1 / np.sqrt(2), 0.5])
    problem = SQPGS(f, gI, gE, tol=1e-8, max_iter=200, verbose=False)
    x_k = problem.solve()
    np.testing.assert_array_almost_equal(x_k, xstar, decimal=4)

    return


def test_rosenbrock_from_rand():
    gI = [g]
    gE = []
    xstar = np.array([1 / np.sqrt(2), 0.5])
    x0 = np.random.rand(2)
    problem = SQPGS(f, gI, gE, x0=x0, tol=1e-8, max_iter=200, verbose=False)
    x_k = problem.solve()
    np.testing.assert_array_almost_equal(x_k, xstar, decimal=4)

    return


def test_rosenbrock_with_eq():
    g1 = g_linear(A=np.eye(2), b=np.ones(2))
    gI = []
    gE = [g1]
    xstar = np.ones(2)
    problem = SQPGS(f, gI, gE, tol=1e-8, max_iter=200, verbose=False)
    x_k = problem.solve()
    np.testing.assert_array_almost_equal(x_k, xstar, decimal=4)

    return
