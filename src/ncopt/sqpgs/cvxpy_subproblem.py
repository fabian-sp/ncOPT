from typing import List

import cvxpy as cp
import numpy as np

from ncopt.sqpgs.defaults import DEFAULT_OPTION

CVXPY_SOLVER_DICT = {
    "osqp-cvxpy": cp.OSQP,
    "clarabel": cp.CLARABEL,
    "cvxopt": cp.CVXOPT,
    "mosek": cp.MOSEK,
    "gurobi": cp.GUROBI,
}


class CVXPYSubproblemSQPGS:
    def __init__(
        self,
        dim: int,
        p0: np.ndarray,
        pI: np.ndarray,
        pE: np.ndarray,
        assert_tol: float,
        solver: str = DEFAULT_OPTION.qp_solver,
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
        self._qp_solver = CVXPY_SOLVER_DICT.get(solver, cp.OSQP)

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
        L: np.ndarray,
        rho: float,
        D_f: np.ndarray,
        D_gI: List[np.ndarray],
        D_gE: List[np.ndarray],
        f_k: float,
        gI_k: np.ndarray,
        gE_k: np.ndarray,
    ) -> None:
        """
        This solves the quadratic program

        Parameters

        L : array
            Cholesky factor of Hessian approximation
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

        d = cp.Variable(self.dim)
        z = cp.Variable()
        if self.has_ineq_constraints:
            r_I = cp.Variable(gI_k.size, nonneg=True)
        if self.has_eq_constraints:
            r_E = cp.Variable(gE_k.size, nonneg=True)

        objective = rho * z + (1 / 2) * cp.sum_squares(L.T @ d)

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
        problem.solve(solver=self._qp_solver, verbose=False)

        assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        self._problem = problem

        # Extract primal solution
        self.d = d.value.copy()

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
