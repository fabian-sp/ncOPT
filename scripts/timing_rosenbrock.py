"""
author: Fabian Schaipp
"""
import timeit

import numpy as np

from ncopt.funs import f_rosenbrock, g_linear, g_max
from ncopt.sqpgs import SQPGS

# %%
f = f_rosenbrock()
g = g_max()

# inequality constraints (list of functions)
gI = [g]

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

A = np.eye(2)
b = np.zeros(2)
g1 = g_linear(A, b)

# equality constraints (list of scalar functions)
gE = [g1]
problem = SQPGS(f, gI, gE, x0, tol=1e-20, max_iter=100, verbose=False)

np.random.seed(31)
x0 = np.random.randn(2)

timeit.timeit("x_k = problem.solve()", number=10)

# result (start of refactoring):
# 291 ms ± 3.53 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
