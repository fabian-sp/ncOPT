"""
author: Fabian Schaipp
"""

import numpy as np


class f_rosenbrock:
    """
    Nonsmooth Rosenbrock function (see 5.1 in Curtis, Overton
    "SQP FOR NONSMOOTH CONSTRAINED OPTIMIZATION")

    x -> w|x_1^2 - x_2| + (1 - x_1)^2
    """

    def __init__(self, w=8):
        self.name = "rosenbrock"
        self.dim = 2
        self.dimOut = 1
        self.w = w

    def eval(self, x):
        return self.w * np.abs(x[0] ** 2 - x[1]) + (1 - x[0]) ** 2

    def differentiable(self, x):
        return np.abs(x[0] ** 2 - x[1]) > 1e-10

    def grad(self, x):
        a = np.array([-2 + x[0], 0])
        sign = np.sign(x[0] ** 2 - x[1])

        if sign == 1:
            b = np.array([2 * x[0], -1])
        elif sign == -1:
            b = np.array([-2 * x[0], 1])
        else:
            b = np.array([-2 * x[0], 1])

        # b = np.sign(x[0]**2 -x[1]) * np.array([2*x[0], -1])
        return a + b


class g_max:
    """
    maximum function (see 5.1 in Curtis, Overton "SQP FOR NONSMOOTH CONSTRAINED OPTIMIZATION")

    x -> max(c1*x_1, c2*x_2) - 1
    """

    def __init__(self, c1=np.sqrt(2), c2=2.0):
        self.name = "max"
        self.c1 = c1
        self.c2 = c2
        self.dimOut = 1
        return

    def eval(self, x):
        return np.maximum(self.c1 * x[0], self.c2 * x[1]) - 1

    def differentiable(self, x):
        return np.abs(self.c1 * x[0] - self.c2 * x[1]) > 1e-10

    def grad(self, x):
        sign = np.sign(self.c1 * x[0] - self.c2 * x[1])
        if sign == 1:
            g = np.array([self.c1, 0])
        elif sign == -1:
            g = np.array([0, self.c2])
        else:
            g = np.array([0, self.c2])
        return g


class g_linear:
    """
    linear constraint:

    x -> Ax - b
    """

    def __init__(self, A, b):
        self.name = "linear"
        self.A = A
        self.b = b
        self.dim = A.shape[0]
        self.dimOut = A.shape[1]
        return

    def eval(self, x):
        return self.A @ x - self.b

    def differentiable(self, x):
        return True

    def grad(self, x):
        return self.A
