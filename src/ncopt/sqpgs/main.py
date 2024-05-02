"""
Implements the SQP-GS algorithm from

    Frank E. Curtis and Michael L. Overton, A sequential quadratic programming
    algorithm for nonconvex, nonsmooth constrained optimization,
    SIAM Journal on Optimization 2012 22:2, 474-500, https://doi.org/10.1137/090780201.

The notation of the code tries to follow closely the notation of the paper.
"""

import copy
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ncopt.functions import ObjectiveOrConstraint
from ncopt.plot_utils import plot_metrics, plot_timings
from ncopt.sqpgs.cvxpy_subproblem import CVXPYSubproblemSQPGS
from ncopt.sqpgs.defaults import DEFAULT_ARG, DEFAULT_OPTION
from ncopt.sqpgs.osqp_subproblem import OSQPSubproblemSQPGS
from ncopt.utils import compute_batch_jacobian_vmap, get_logger


class SQPGS:
    def __init__(
        self,
        f: ObjectiveOrConstraint,
        gI: List[ObjectiveOrConstraint],
        gE: List[ObjectiveOrConstraint],
        x0: Optional[np.array] = None,
        tol: float = DEFAULT_ARG.tol,
        max_iter: int = DEFAULT_ARG.max_iter,
        verbose: bool = DEFAULT_ARG.verbose,
        options: dict = {},
        assert_tol: float = DEFAULT_ARG.assert_tol,
        store_history: bool = DEFAULT_ARG.store_history,
        log_every: int = DEFAULT_ARG.log_every,
    ) -> None:
        """The problem object for using the SQP-GS method.

        Parameters
        ----------
        f : ObjectiveOrConstraint
            The objective funtion. Argument ``dim`` must be specified.
            Output must be of shape ``(batch_size, 1)``.
        gI : List[ObjectiveOrConstraint]
            List of inequality constraint functions.
            Argument ``dim_out`` must be specified for each.
            Output must be of shape ``(batch_size, dim_out)``.
            Pass empty list for no inequality constraints.
        gE : List[ObjectiveOrConstraint]
            List of equality constraint functions.
            Argument ``dim_out`` must be specified for each.
            Output must be of shape ``(batch_size, dim_out)``.
            Pass empty list for no equality constraints.
        x0 : Optional[np.array], optional
            Starting point, by default vector of zeros.
        tol : float, optional
            Tolerance for stopping, measured by E_k in [Curtis and Overton, 2012].
            By default DEFAULT_ARG.tol.
        max_iter : int, optional
            Maximum number of iterations, by default DEFAULT_ARG.max_iter.
        verbose : bool, optional
            Whether to print status updates, by default DEFAULT_ARG.verbose.
        options : dict, optional
            Dictionary with options, by default {}.
            All specified entries will overwrite the default value.
            See ``./defaults.py`` for possible key names and default values.
        assert_tol : float, optional
            Assertion tolerance used in mathematical checks, by default DEFAULT_ARG.assert_tol.
            You can avoid errors raised by assertions by increasing this number.
            Note that in that case, the algorihtm might be not producing correct output.
        store_history : bool, optional
            Whether to store the history of iterates in every iteration, by default False.
            Only use this for small examples, as it might produce memory overflow.
        log_every : int, optional
            Frequency of status updates (if ``verbose=True``) by default DEFAULT_ARG.log_every.

        """
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
        if not hasattr(f, "dim"):
            raise KeyError(
                "Input dimension needs to be specified for the objective function \
                           Make sure to pass `dim` when initializing the \
                            `ObjectiveOrConstraint` object."
            )

        self.dim = self.f.dim
        self.dimI = np.array([g.dim_out for g in self.gI], dtype=int)
        self.dimE = np.array([g.dim_out for g in self.gE], dtype=int)

        self.nI_ = len(self.gI)  # number of inequality function objects
        self.nE_ = len(self.gE)  # number of equality function objects

        self.nI = sum(self.dimI)  # number of inequality costraints
        self.nE = sum(self.dimE)  # number of equality costraints

        # Construct number of sample point arrays
        self.p0, self.pI_, self.pE_ = self._init_sample_points()
        ###############################################################
        # Initialize

        self.status = "not optimal"
        self.logger = get_logger(name="ncopt", verbose=self.verbose)

        # starting point
        if x0 is None:
            self.x_k = np.zeros(self.dim)
        else:
            self.x_k = x0.copy()

    def _init_sample_points(self):
        # sample points for objective
        p0 = self.options["num_points_obj"] if not self.f.is_differentiable else 0

        pI_ = self.options["num_points_gI"] * np.ones(
            self.nI_, dtype=int
        )  # sample points for ineq constraint
        pE_ = self.options["num_points_gE"] * np.ones(
            self.nE_, dtype=int
        )  # sample points for eq constraint

        # if function is differentiable, set to zero here --> only sample at x_k
        _is_differentiable_I = [g.is_differentiable for g in self.gI]
        _is_differentiable_E = [g.is_differentiable for g in self.gE]

        pI_[_is_differentiable_I] = 0
        pE_[_is_differentiable_E] = 0

        return p0, pI_, pE_

    def plot_timings(self, ax=None):
        """Plotting the timings of each algorithm step, over the course of iterations.

        Parameters
        ----------
        ax : matplotlib.axes._axes.Axes, optional
            Axis to plot on, by default None.
            If not specified, a new figure and axis will be created.

        Returns
        -------
        fig: matplotlib.figure.Figure
            Figure object of the plot.
        ax: matplotlib.axes._axes.Axes
            Axis object of the plot
        """
        fig, ax = plot_timings(self.info["timings"], ax=ax)
        return fig, ax

    def plot_metrics(self, ax=None):
        """Plotting several metrics of each logged step, over the course of iterations.

        Parameters
        ----------
        ax : matplotlib.axes._axes.Axes, optional
            Axis to plot on, by default None.
            If not specified, a new figure and axis will be created.

        Returns
        -------
        fig: matplotlib.figure.Figure
            Figure object of the plot.
        ax: matplotlib.axes._axes.Axes
            Axis object of the plot
        """
        fig, ax = plot_metrics(self.info["metrics"], self.log_every, ax=ax)
        return fig, ax

    def solve(self) -> np.ndarray:
        """Method for solving the problem.
        All metrics and timings will be stored in ``self.info``.

        Note that repeatedly calling this method will use the last iterate as new starting point.

        Returns
        -------
        np.ndarray
            The final iterate (also stored in ``self.x_k``).
        """
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

        ###############################################################
        # helper arrays for subproblem
        p0 = self.p0
        pI = np.repeat(self.pI_, self.dimI)
        pE = np.repeat(self.pE_, self.dimE)

        if self.options["qp_solver"] == "osqp":
            self.SP = OSQPSubproblemSQPGS(self.dim, p0, pI, pE, self.assert_tol)
        else:
            self.SP = CVXPYSubproblemSQPGS(
                self.dim, p0, pI, pE, self.assert_tol, self.options["qp_solver"]
            )

        E_k = np.inf  # for stopping criterion
        x_kmin1 = None  # last iterate
        g_kmin1 = None  #

        # Hessian matrix
        H = np.eye(self.dim)
        s_hist = np.zeros((self.dim, iter_H))
        y_hist = np.zeros((self.dim, iter_H))

        do_step = False

        x_hist = [self.x_k] if self.store_history else None
        timings = {
            "total": [],
            "sample_and_grad": [],
            "subproblem": [],
            "step": [],
            "other": [],
        }
        metrics = {
            "objective": [],
            "constraint_violation": [],
            "accuracy": [],
            "sampling_radius": [],
        }
        ##############################################
        # Start of loop
        ##############################################
        for iter_k in range(self.max_iter):
            t0 = time.perf_counter()
            if E_k <= self.tol:
                self.status = "optimal"
                break

            ##############################################
            # Sampling
            ##############################################
            B_f = sample_points(self.x_k, eps, p0, stack_x=True)

            B_gI = list()
            for j in np.arange(self.nI_):
                B_j = sample_points(self.x_k, eps, pI[j], stack_x=True)
                B_gI.append(B_j)

            B_gE = list()
            for j in np.arange(self.nE_):
                B_j = sample_points(self.x_k, eps, pE[j], stack_x=True)
                B_gE.append(B_j)

            ####################################
            # Compute gradients and evaluate
            ###################################
            D_f, V_f = compute_value_and_jacobian(self.f, B_f, as_numpy=True, split_jac=False)
            assert V_f.shape[1] == 1, "Objective must be a scalar function."
            f_k = V_f[0, 0]  # convert to float
            # squeeze output dimension
            D_f = D_f.squeeze(axis=1)

            D_gI, V_gI = list(), list()
            for j in np.arange(self.nI_):
                this_jac, this_out = compute_value_and_jacobian(
                    self.gI[j], B_gI[j], as_numpy=True, split_jac=True
                )
                D_gI += this_jac
                V_gI.append(this_out)

            D_gE, V_gE = list(), list()
            for j in np.arange(self.nE_):
                this_jac, this_out = compute_value_and_jacobian(
                    self.gE[j], B_gE[j], as_numpy=True, split_jac=True
                )
                D_gE += this_jac
                V_gE.append(this_out)

            # Get value at x_k
            # hstack cannot handle empty lists!
            if self.nI_ > 0:
                gI_k = np.hstack([v[0] for v in V_gI])
            else:
                gI_k = np.array([])

            if self.nE_ > 0:
                gE_k = np.hstack([v[0] for v in V_gE])
            else:
                gE_k = np.array([])

            ##############################################
            # Subproblem solve
            ##############################################
            _reg_H = self.options["reg_H"]
            t1 = time.perf_counter()
            if isinstance(self.SP, OSQPSubproblemSQPGS):
                self.SP.solve(H + _reg_H * np.eye(self.dim), rho, D_f, D_gI, D_gE, f_k, gI_k, gE_k)
            else:
                self.SP.solve(
                    np.linalg.cholesky(H + _reg_H * np.eye(self.dim)),
                    rho,
                    D_f,
                    D_gI,
                    D_gE,
                    f_k,
                    gI_k,
                    gE_k,
                )

            d_k = self.SP.d.copy()
            t2 = time.perf_counter()

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
                    f"Iter {iter_k}, objective {f_k:.3E}, constraint violation {viol_k:.3E}, "
                    + f"accuracy {E_k:.3E}, "
                    + f"avg runtime/iter {(1e3) * np.mean(timings['total']):.3E} ms."
                )
                metrics["objective"].append(f_k)
                metrics["constraint_violation"].append(viol_k)
                metrics["accuracy"].append(E_k)
                metrics["sampling_radius"].append(eps)

            new_E_k = stop_criterion(
                self.gI, self.gE, g_k, self.SP, gI_k, gE_k, V_gI, V_gE, self.nI_, self.nE_
            )
            E_k = min(E_k, new_E_k)

            ##############################################
            # Step
            ##############################################
            t3 = time.perf_counter()
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

                    H = np.eye(self.dim)
                    for l in np.arange(iter_H):
                        sl = s_hist[:, l]
                        yl = y_hist[:, l]

                        cond1 = (np.linalg.norm(sl) <= xi_s * eps) and (
                            np.linalg.norm(yl) <= xi_y * eps
                        )
                        cond2 = np.inner(sl, yl) >= xi_sy * eps**2
                        cond = cond1 and cond2

                        if cond:
                            Hs = H @ sl
                            H = (
                                H
                                - np.outer(Hs, Hs) / (sl @ Hs + 1e-16)
                                + np.outer(yl, yl) / (yl @ sl + 1e-16)
                            )

                ####################################
                # Actual step
                ###################################
                x_kmin1 = self.x_k.copy()
                g_kmin1 = g_k.copy()

                self.x_k = self.x_k + alpha * d_k

            ##############################################
            # No step
            ##############################################
            else:
                if v_k <= theta:
                    theta *= beta_theta
                else:
                    rho *= beta_rho

                eps *= beta_eps

            if self.store_history:
                x_hist.append(self.x_k)

            t4 = time.perf_counter()
            timings["total"].append(t4 - t0)
            timings["sample_and_grad"].append(t1 - t0)
            timings["subproblem"].append(t2 - t1)
            timings["other"].append(t3 - t2)
            timings["step"].append(t4 - t3)

        ##############################################
        # End of loop
        ##############################################
        self.x_hist = np.vstack(x_hist) if self.store_history else None
        self.info = {"timings": timings, "metrics": metrics}
        if E_k > self.tol:
            self.status = "max iterations reached"

        _final_msg = f"SQP-GS has terminated with status: {self.status}, final accuracy {E_k}."
        self.logger.info(_final_msg)
        print(_final_msg)  # we always want to display this.

        return self.x_k


def sample_points(x: np.ndarray, eps: float, n_points: int, stack_x: bool = True) -> torch.Tensor:
    """Sample ``n_points`` uniformly from the ``eps``-ball around ``x``.

    Parameters
    ----------
    x : np.ndarray
        The centre of the ball.
    eps : float
        Sampling radius.
    n_points : int
        Number of sampled points.
    stack_x : bool
        Whether to stack ``x`` itself at the top. Default true.
    Returns
    -------
    torch.Tensor
        Shape (n_points, len(x)).
    """
    dim = len(x)
    if n_points == 0:
        # return only x
        X = torch.zeros(1, dim)
    else:
        U = torch.randn(n_points, dim)
        norm_U = torch.linalg.norm(U, axis=1)
        R = torch.rand(n_points) ** (1 / dim)
        X = eps * (R / norm_U).reshape(-1, 1) * U

        if stack_x:
            X = torch.vstack((torch.zeros(1, dim), X))

    X += torch.from_numpy(x).reshape(1, -1)
    return X


def compute_value_and_jacobian(
    f: ObjectiveOrConstraint, X: torch.Tensor, as_numpy: bool = True, split_jac: bool = True
) -> Tuple[Union[torch.tensor, np.ndarray], Union[torch.tensor, np.ndarray]]:
    """Evaluates function value and Jacobian for a batched input ``X``.


    Parameters
    ----------
    f : ObjectiveOrConstraint
        The function to evaluate. Should map from ``dim`` to R^m (with m integer)
    X : torch.Tensor
        Input points. Should be of shape ``(batch_size, dim)``.
    as_numpy : bool, optional
        Whether to convert Jacobian into numpy array, by default True
    split_jac : bool, optional
        Whether to split Jacobian into a list, by default True.
        The splitting happens wrt the output dimension.

    Returns
    -------
    Tuple[Union[torch.tensor, np.ndarray], Union[torch.tensor, float]]
        Jacobian of shape (batch_size, dim), and output values of shape (batch_size, dim_out).
    """
    # Observation:
    # The jacobian is still computed correctly for compute_batch_jacobian_vmap,
    # even if the model would output a tensor of shape (b,) instead of (b,1)
    # We could make the solver more flexible to handle this case, but then might
    # lose the ability to switch out compute_batch_jacobian_vmap

    jac, out = compute_batch_jacobian_vmap(f, X)

    if as_numpy:
        jac = jac.detach().cpu().numpy()
        out = out.detach().cpu().numpy()

    if split_jac:
        jac = [jac[:, j, :] for j in range(jac.shape[1])]

    return jac, out


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
    term1 = rho * f.single_eval(x).squeeze()  # want a float here

    # inequalities
    if len(gI) > 0:
        term2 = np.sum(np.hstack([np.maximum(gI[j].single_eval(x), 0) for j in range(len(gI))]))
    else:
        term2 = 0
    # equalities: max(x,0) + max(-x,0) = abs(x)
    if len(gE) > 0:
        term3 = np.sum(np.hstack([np.abs(gE[l].single_eval(x)) for l in range(len(gE))]))
    else:
        term3 = 0

    return term1 + term2 + term3


def stop_criterion(gI, gE, g_k, SP, gI_k, gE_k, V_gI, V_gE, nI_, nE_):
    """
    computes E_k in the paper
    """
    val1 = np.linalg.norm(g_k, np.inf)

    # as gI or gE could be empty, we need a max value for empty arrays --> initial argument
    val2 = np.max(gI_k, initial=-np.inf)
    val3 = np.max(np.abs(gE_k), initial=-np.inf)

    gI_vals = list()
    for j in np.arange(nI_):
        V = V_gI[j]
        gI_vals += [V[:, i] for i in range(gI[j].dim_out)]

    val4 = -np.inf
    for j in np.arange(len(gI_vals)):
        val4 = np.maximum(val4, np.max(SP.lambda_gI[j] * gI_vals[j]))

    gE_vals = list()
    for j in np.arange(nE_):
        V = V_gE[j]
        gE_vals += [V[:, i] for i in range(gE[j].dim_out)]

    val5 = -np.inf
    for j in np.arange(len(gE_vals)):
        val5 = np.maximum(val5, np.max(SP.lambda_gE[j] * gE_vals[j]))

    return np.max(np.array([val1, val2, val3, val4, val5]))
