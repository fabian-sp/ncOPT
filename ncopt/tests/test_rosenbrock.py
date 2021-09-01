"""
author: Fabian Schaipp
"""

import numpy as np
import os


os.chdir('../..')

from ncopt.sqpgs import SQP_GS
from ncopt.funs import f_rosenbrock, g_max, g_upper_bound

f = f_rosenbrock()
g = g_max()
g1 = g_upper_bound(lhs=5)

def test_rosenbrock_from_zero():
    gI = [g]
    gE = []
    xstar = np.array([1/np.sqrt(2), 0.5])
    x_k, x_hist, SP = SQP_GS(f, gI, gE, tol = 1e-8, max_iter = 200, verbose = False)
    np.testing.assert_array_almost_equal(x_k, xstar, decimal = 4)

    return

def test_rosenbrock_from_rand():
    gI = [g]
    gE = []
    xstar = np.array([1/np.sqrt(2), 0.5])
    x0 = np.random.rand(2)
    x_k, x_hist, SP = SQP_GS(f, gI, gE, x0, tol = 1e-8, max_iter = 200, verbose = False)
    np.testing.assert_array_almost_equal(x_k, xstar, decimal = 4)

    return