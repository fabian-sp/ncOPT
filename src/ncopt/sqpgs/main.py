"""
@author: Fabian Schaipp

Implements the SQP-GS algorithm from

    Frank E. Curtis and Michael L. Overton, A sequential quadratic programming
    algorithm for nonconvex, nonsmooth constrained optimization,
    SIAM Journal on Optimization 2012 22:2, 474-500, https://doi.org/10.1137/090780201.

The notation of the code tries to follow closely the notation of the paper.
"""

import copy
import time
from typing import Optional

import cvxpy as cp
import numpy as np

from ncopt.sqpgs.defaults import DEFAULT_ARG, DEFAULT_OPTION
from ncopt.utils import get_logger


class SQPGS:
    def __init__(
        self,
        f,
        gI,
        gE,
        x0: Optional[np.array] = None,
        tol: float = DEFAULT_ARG.tol,
        max_iter: int = DEFAULT_ARG.max_iter,
        verbose: bool = DEFAULT_ARG.verbose,
        options: dict = {},
        assert_tol: float = DEFAULT_ARG.assert_tol,
        store_history: bool = DEFAULT_ARG.store_history,
        log_every: int = DEFAULT_ARG.log_every,
    ) -> None:
        if tol < 0:
            raise ValueError(f"Tolerance must be non-negative, but was specified as {tol}.")
        if max_iter < 0:
            raise ValueError(
                f"Maximum number of iterations must be non-negative, \
                but was specified as {max_iter}."
            )
        if assert_tol < 0:
            raise ValueError(
                f"Assertion tolerance must be non-negative, but \
                was specified as {assert_tol}."
            )

        self.f = f
        self.gI = gI
        self.gE = gE
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.options = options
        self.assert_tol = assert_tol
        self.store_history = store_history
        self.log_every = log_every

        # Set options/hyperparameters
        # (defaults are chose according to recommendations in paper)
        self.options = copy.deepcopy(DEFAULT_OPTION.copy())
        self.options.update(options)

        ###############################################################
        # Extract dimensions

        # extract dimensions of constraints
        self.dim = self.f.dim
        self.dimI = np.array([g.dimOut for g in self.gI], dtype=int)
        self.dimE = np.array([g.dimOut for g in self.gE], dtype=int)

        self.nI_ = len(self.gI)  # number of inequality function objects
        self.nE_ = len(self.gE)  # number of equality function objects

        self.nI = sum(self.dimI)  # number of inequality costraints
        self.nE = sum(self.dimE)  # number of equality costraints

        ###############################################################
        # Initialize

        self.status = "not optimal"
        self.logger = get_logger(name="ncopt", verbose=self.verbose)

        # starting point
        if x0 is None:
            self.x_k = np.zeros(self.dim)
        else:
            self.x_k = x0.copy()

    def solve(self):
        ###############################################################
        # Set all hyperparameters

        eps = self.options["eps"]  # sampling radius
        rho = self.options["rho"]
        theta = self.options["theta"]

        eta = self.options["eta"]
        gamma = self.options["gamma"]
        beta_eps = self.options["beta_eps"]
        beta_rho = self.options["beta_rho"]
        beta_theta = self.options["beta_theta"]
        nu = self.options["nu"]
        xi_s = self.options["xi_s"]
        xi_y = self.options["xi_y"]
        xi_sy = self.options["xi_sy"]
        iter_H = self.options["iter_H"]

        p0 = self.options["num_points_obj"]  # sample points for objective
        pI_ = self.options["num_points_gI"] * np.ones(
            self.nI_, dtype=int
        )  # sample points for ineq constraint
        pE_ = self.options["num_points_gE"] * np.ones(
            self.nE_, dtype=int
        )  # sample points for eq constraint

        pI = np.repeat(pI_, self.dimI)
        pE = np.repeat(pE_, self.dimE)
        ###############################################################

        self.SP = SubproblemSQPGS(self.dim, p0, pI, pE, self.assert_tol)

        E_k = np.inf  # for stopping criterion
        x_kmin1 = None  # last iterate
        g_kmin1 = None  #

        # Hessian matrix
        H = np.eye(self.dim)
        s_hist = np.zeros((self.dim, iter_H))
        y_hist = np.zeros((self.dim, iter_H))

        do_step = False

        x_hist = [self.x_k] if self.store_history else None
        timings = {"total": [], "sp_update": [], "sp_solve": []}
        metrics = {
            "objective": [],
            "constraint_violation": [],
            "accuracy": [],
            "sampling_radius": [],
        }
        ##############################################
        # START OF LOOP
        ##############################################
        for iter_k in range(self.max_iter):
            t0 = time.perf_counter()
            if E_k <= self.tol:
                self.status = "optimal"
                break

            ##############################################
            # SAMPLING
            ##############################################
            B_f = sample_points(self.x_k, eps, p0)
            B_f = np.vstack((self.x_k, B_f))

            B_gI = list()
            for j in np.arange(self.nI_):
                B_j = sample_points(self.x_k, eps, pI_[j])
                B_j = np.vstack((self.x_k, B_j))
                B_gI.append(B_j)

            B_gE = list()
            for j in np.arange(self.nE_):
                B_j = sample_points(self.x_k, eps, pE_[j])
                B_j = np.vstack((self.x_k, B_j))
                B_gE.append(B_j)

            ####################################
            # COMPUTE GRADIENTS AND EVALUATE
            ###################################
            D_f = compute_gradients(self.f, B_f)[0]  # returns list, always has one element

            D_gI = list()
            for j in np.arange(self.nI_):
                D_gI += compute_gradients(self.gI[j], B_gI[j])

            D_gE = list()
            for j in np.arange(self.nE_):
                D_gE += compute_gradients(self.gE[j], B_gE[j])

            f_k = self.f.eval(self.x_k)
            # hstack cannot handle empty lists!
            if self.nI_ > 0:
                gI_k = np.hstack([self.gI[j].eval(self.x_k) for j in range(self.nI_)])
            else:
                gI_k = np.array([])

            if self.nE_ > 0:
                gE_k = np.hstack([self.gE[j].eval(self.x_k) for j in range(self.nE_)])
            else:
                gE_k = np.array([])

            ##############################################
            # SUBPROBLEM
            ##############################################

            self.SP.solve(H, rho, D_f, D_gI, D_gE, f_k, gI_k, gE_k)

            timings["sp_update"].append(self.SP.setup_time)
            timings["sp_solve"].append(self.SP.solve_time)

            d_k = self.SP.d.value.copy()
            # compute g_k from paper
            g_k = (
                self.SP.lambda_f @ D_f
                + np.sum([self.SP.lambda_gI[j] @ D_gI[j] for j in range(self.nI)], axis=0)
                + np.sum([self.SP.lambda_gE[j] @ D_gE[j] for j in range(self.nE)], axis=0)
            )

            # evaluate v(x) at x=x_k
            v_k = np.maximum(gI_k, 0).sum() + np.sum(np.abs(gE_k))
            phi_k = rho * f_k + v_k
            delta_q = phi_k - q_rho(d_k, rho, H, f_k, gI_k, gE_k, D_f, D_gI, D_gE)

            assert (
                delta_q >= -self.assert_tol
            ), f"Value is supposed to be non-negative, but is {delta_q}."
            assert (
                np.abs(self.SP.lambda_f.sum() - rho) <= self.assert_tol
            ), f"Value is supposed to be negative, but is {np.abs(self.SP.lambda_f.sum() - rho)}."

            # Logging, start after first iteration
            if iter_k % self.log_every == 1:
                violI_k = np.maximum(gI_k, 0)
                violE_k = np.abs(gE_k)
                viol_k = np.max(np.hstack((violI_k, violE_k)))
                self.logger.info(
                    f"Iter {iter_k}, objective {f_k:.3E}, constraint violation {viol_k:.3E}, \
                    accuracy {E_k:.3E}, \
                    avg runtime/iter {(1e3) * np.mean(timings['total']):.3E} ms."
                )
                metrics["objective"].append(f_k)
                metrics["constraint_violation"].append(viol_k)
                metrics["accuracy"].append(E_k)
                metrics["sampling_radius"].append(eps)

            new_E_k = stop_criterion(
                self.gI, self.gE, g_k, self.SP, gI_k, gE_k, B_gI, B_gE, self.nI_, self.nE_, pI, pE
            )
            E_k = min(E_k, new_E_k)

            ##############################################
            # STEP
            ##############################################

            do_step = delta_q > nu * eps**2  # Flag whether step is taken or not
            if do_step:
                alpha = 1.0
                phi_new = phi_rho(self.x_k + alpha * d_k, self.f, self.gI, self.gE, rho)

                # Armijo step size rule
                while phi_new > phi_k - eta * alpha * delta_q:
                    alpha *= gamma
                    phi_new = phi_rho(self.x_k + alpha * d_k, self.f, self.gI, self.gE, rho)

                # update Hessian
                if x_kmin1 is not None:
                    s_k = self.x_k - x_kmin1
                    s_hist = np.roll(s_hist, 1, axis=1)
                    s_hist[:, 0] = s_k

                    y_k = g_k - g_kmin1
                    y_hist = np.roll(y_hist, 1, axis=1)
                    y_hist[:, 0] = y_k

                    hH = np.eye(self.dim)
                    for l in np.arange(iter_H):
                        sl = s_hist[:, l]
                        yl = y_hist[:, l]

                        cond1 = (np.linalg.norm(sl) <= xi_s * eps) and (
                            np.linalg.norm(yl) <= xi_y * eps
                        )
                        cond2 = np.inner(sl, yl) >= xi_sy * eps**2
                        cond = cond1 and cond2

                        if cond:
                            Hs = hH @ sl
                            hH = (
                                hH
                                - np.outer(Hs, Hs) / (sl @ Hs + 1e-16)
                                + np.outer(yl, yl) / (yl @ sl + 1e-16)
                            )

                    H = hH.copy()

                ####################################
                # ACTUAL STEP
                ###################################
                x_kmin1 = self.x_k.copy()
                g_kmin1 = g_k.copy()

                self.x_k = self.x_k + alpha * d_k

            ##############################################
            # NO STEP
            ##############################################
            else:
                if v_k <= theta:
                    theta *= beta_theta
                else:
                    rho *= beta_rho

                eps *= beta_eps

            if self.store_history:
                x_hist.append(self.x_k)
            t1 = time.perf_counter()
            timings["total"].append(t1 - t0)

        ##############################################
        # END OF LOOP
        ##############################################
        self.x_hist = np.vstack(x_hist) if self.store_history else None
        self.info = {"timings": timings, "metrics": metrics}
        if E_k > self.tol:
            self.status = "max iterations reached"

        _final_msg = f"SQP-GS has terminated with status: {self.status}, final accuracy {E_k}."
        self.logger.info(_final_msg)
        print(_final_msg)  # we always want to display this.

        return self.x_k


def sample_points(x, eps, N):
    """
    sample N points uniformly distributed in eps-ball around x
    """
    dim = len(x)
    U = np.random.randn(N, dim)
    norm_U = np.linalg.norm(U, axis=1)
    R = np.random.rand(N) ** (1 / dim)
    Z = eps * (R / norm_U)[:, np.newaxis] * U

    return x + Z


def q_rho(d, rho, H, f_k, gI_k, gE_k, D_f, D_gI, D_gE):
    term1 = rho * (f_k + np.max(D_f @ d))

    term2 = 0
    for j in np.arange(len(D_gI)):
        term2 += np.maximum(gI_k[j] + D_gI[j] @ d, 0).max()

    term3 = 0
    for l in np.arange(len(D_gE)):
        term3 += np.abs(gE_k[l] + D_gE[l] @ d).max()

    term4 = 0.5 * d.T @ H @ d
    return term1 + term2 + term3 + term4


def phi_rho(x, f, gI, gE, rho):
    term1 = rho * f.eval(x)

    # inequalities
    if len(gI) > 0:
        term2 = np.sum(np.hstack([np.maximum(gI[j].eval(x), 0) for j in range(len(gI))]))
    else:
        term2 = 0
    # equalities: max(x,0) + max(-x,0) = abs(x)
    if len(gE) > 0:
        term3 = np.sum(np.hstack([np.abs(gE[l].eval(x)) for l in range(len(gE))]))
    else:
        term3 = 0

    return term1 + term2 + term3


def stop_criterion(gI, gE, g_k, SP, gI_k, gE_k, B_gI, B_gE, nI_, nE_, pI, pE):
    """
    computes E_k in the paper
    """
    val1 = np.linalg.norm(g_k, np.inf)

    # as gI or gE could be empty, we need a max value for empty arrays --> initial argument
    val2 = np.max(gI_k, initial=-np.inf)
    val3 = np.max(np.abs(gE_k), initial=-np.inf)

    gI_vals = list()
    for j in np.arange(nI_):
        gI_vals += eval_ineq(gI[j], B_gI[j])

    val4 = -np.inf
    for j in np.arange(len(gI_vals)):
        val4 = np.maximum(val4, np.max(SP.lambda_gI[j] * gI_vals[j]))

    gE_vals = list()
    for j in np.arange(nE_):
        gE_vals += eval_ineq(gE[j], B_gE[j])

    val5 = -np.inf
    for j in np.arange(len(gE_vals)):
        val5 = np.maximum(val5, np.max(SP.lambda_gE[j] * gE_vals[j]))

    return np.max(np.array([val1, val2, val3, val4, val5]))


def eval_ineq(fun, X):
    """
    evaluate function at multiple inputs
    needed in stop_criterion

    Returns
    -------
    list of array, number of entries = fun.dimOut
    """
    (N, _) = X.shape
    D = np.zeros((N, fun.dimOut))
    for i in np.arange(N):
        D[
            i,
            :,
        ] = fun.eval(X[i, :])

    return [D[:, j] for j in range(fun.dimOut)]


def compute_gradients(fun, X):
    """
    computes gradients of function object f at all rows of array X

    Returns
    -------
    list of 2d-matrices, length of fun.dimOut
    """
    (N, dim) = X.shape

    # fun.grad returns Jacobian, i.e. dimOut x dim
    D = np.zeros((N, fun.dimOut, dim))
    for i in np.arange(N):
        D[i, :, :] = fun.grad(X[i, :])

    return [D[:, j, :] for j in np.arange(fun.dimOut)]


# %%


class SubproblemSQPGS:
    def __init__(
        self, dim: int, p0: np.ndarray, pI: np.ndarray, pE: np.ndarray, assert_tol: float
    ) -> None:
        """
        dim : solution space dimension
        p0 : number of sample points for f (excluding x_k itself)
        pI : array, number of sample points for inequality constraint (excluding x_k itself)
        pE : array, number of sample points for equality constraint (excluding x_k itself)
        """

        self.dim = dim
        self.p0 = p0
        self.pI = pI
        self.pE = pE

        self.assert_tol = assert_tol

        self.d = cp.Variable(self.dim)
        self._problem = None

    @property
    def nI(self) -> int:
        return len(self.pI)

    @property
    def nE(self) -> int:
        return len(self.pE)

    @property
    def has_ineq_constraints(self) -> bool:
        return self.nI > 0

    @property
    def has_eq_constraints(self) -> bool:
        return self.nE > 0

    @property
    def problem(self) -> cp.Problem:
        assert self._problem is not None, "Problem not yet initialized."
        return self._problem

    @property
    def status(self) -> str:
        return self.problem.status

    @property
    def objective_val(self) -> float:
        return self.problem.value

    @property
    def setup_time(self) -> float:
        return self.problem.solver_stats.setup_time

    @property
    def solve_time(self) -> float:
        return self.problem.solver_stats.solve_time

    def solve(
        self,
        H: np.ndarray,
        rho: float,
        D_f: np.ndarray,
        D_gI: list[np.ndarray],
        D_gE: list[np.ndarray],
        f_k: float,
        gI_k: np.ndarray,
        gE_k: np.ndarray,
    ) -> None:
        """
        This solves the quadratic program

        Parameters

        H : array
            Hessian approximation
        rho : float
            parameter
        D_f : array
            gradient of f at the sampled points
        D_gI : list
            j-th element is the gradient array of c^j at the sampled points.
        D_gE : list
            j-th element is the gradient array of h^j at the sampled points.
        f_k : float
            evaluation of f at x_k.
        gI_k : array
            evaluation of inequality constraints at x_k.
        gE_k : array
            evaluation of equality constraints at x_k.

        Updates
        self.d: Variable
            search direction

        self.lambda_f: array
            KKT multipier for objective.

        self.lambda_gE: list
            KKT multipier for equality constraints.

        self.lambda_gI: list
            KKT multipier for inequality constraints.

        """

        d = self.d
        z = cp.Variable()
        if self.has_ineq_constraints:
            r_I = cp.Variable(gI_k.size, nonneg=True)
        if self.has_eq_constraints:
            r_E = cp.Variable(gE_k.size, nonneg=True)

        objective = rho * z + (1 / 2) * cp.quad_form(d, H)

        obj_constraint = f_k + D_f @ d <= z
        constraints = [obj_constraint]

        if self.has_ineq_constraints:
            ineq_constraints = [gI_k[j] + D_gI[j] @ d <= r_I[j] for j in range(self.nI)]
            constraints += ineq_constraints
            objective = objective + cp.sum(r_I)

        if self.has_eq_constraints:
            eq_constraints_plus = [gE_k[j] + D_gE[j] @ d <= r_E[j] for j in range(self.nE)]
            eq_constraints_neg = [gE_k[j] + D_gE[j] @ d >= r_E[j] for j in range(self.nE)]
            constraints += eq_constraints_plus + eq_constraints_neg
            objective = objective + cp.sum(r_E)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.CLARABEL)

        assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        self._problem = problem

        # Extract dual variables
        duals = problem.solution.dual_vars
        self.lambda_f = duals[obj_constraint.id]

        if self.has_ineq_constraints:
            self.lambda_gI = [duals[c.id] for c in ineq_constraints]
        else:
            self.lambda_gI = []

        if self.has_eq_constraints:
            self.lambda_gE = [
                duals[c_plus.id] - duals[c_neg.id]
                for c_plus, c_neg in zip(eq_constraints_plus, eq_constraints_neg)
            ]
        else:
            self.lambda_gE = []
