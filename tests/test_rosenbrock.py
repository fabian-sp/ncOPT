"""
author: Fabian Schaipp
"""
import numpy as np

from ncopt.funs import f_rosenbrock, g_linear, g_max
from ncopt.sqpgs import SQP_GS

f = f_rosenbrock()
g = g_max()


def test_rosenbrock_from_zero():
    gI = [g]
    gE = []
    xstar = np.array([1 / np.sqrt(2), 0.5])
    x_k, x_hist, SP = SQP_GS(f, gI, gE, tol=1e-8, max_iter=200, verbose=False)
    np.testing.assert_array_almost_equal(x_k, xstar, decimal=4)

    return


def test_rosenbrock_from_rand():
    gI = [g]
    gE = []
    xstar = np.array([1 / np.sqrt(2), 0.5])
    x0 = np.random.rand(2)
    x_k, x_hist, SP = SQP_GS(f, gI, gE, x0, tol=1e-8, max_iter=200, verbose=False)
    np.testing.assert_array_almost_equal(x_k, xstar, decimal=4)

    return


def test_rosenbrock_with_eq():
    g1 = g_linear(A=np.eye(2), b=np.ones(2))
    gI = []
    gE = [g1]
    xstar = np.ones(2)
    x0 = np.zeros(2)
    x_k, x_hist, SP = SQP_GS(f, gI, gE, x0, tol=1e-8, max_iter=200, verbose=False)
    np.testing.assert_array_almost_equal(x_k, xstar, decimal=4)

    return
