from typing import List

import numpy as np
import osqp
from scipy import sparse

# see: https://osqp.org/docs/interfaces/status_values.html
OSQP_ALLOWED_STATUS = [
    "solved",
    "solved inaccurate",
    "maximum iterations reached",
    "run time limit reached",
]


class OSQPSubproblemSQPGS:
    def __init__(
        self,
        dim: int,
        p0: np.ndarray,
        pI: np.ndarray,
        pE: np.ndarray,
        assert_tol: float,
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

        self._problem = None
        self.P, self.q, self.inG, self.inh, self.nonnegG, self.nonnegh = self._initialize()

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
    def problem(self) -> osqp.interface.OSQP:
        assert self._problem is not None, "Problem not yet initialized."
        return self._problem

    @property
    def status(self) -> str:
        return self.problem.status

    def _initialize(self):
        """
        The quadratic subrpoblem we solve in every iteration is of the form:

        min_y 1/2 * y.T @ P @ y + q.T @ y subject to G @ y <= h

        variable structure: y = (d,z,rI,rE) with
        d = search direction
        z = helper variable for objective
        rI = helper variable for inequality constraints
        rI = helper variable for equality constraints

        This function initializes the variables P,q,G,h.
        The entries which change in every iteration are then updated in self.update()

        G and h consist of two parts:
            1) inG, inh: the inequalities from the paper
            2) nonnegG, nonnegh: nonnegativity bounds rI >= 0, rE >= 0
        """

        dimQP = self.dim + 1 + self.nI + self.nE

        P = np.zeros((dimQP, dimQP))
        q = np.zeros(dimQP)

        inG = np.zeros((1 + self.p0 + np.sum(1 + self.pI) + 2 * np.sum(1 + self.pE), dimQP))
        inh = np.zeros(1 + self.p0 + np.sum(1 + self.pI) + 2 * np.sum(1 + self.pE))

        # structure of inG (p0+1, sum(1+pI), sum(1+pE), sum(1+pE))
        inG[: self.p0 + 1, self.dim] = -1

        for j in range(self.nI):
            inG[
                self.p0 + 1 + (1 + self.pI)[:j].sum() : self.p0
                + 1
                + (1 + self.pI)[:j].sum()
                + self.pI[j]
                + 1,
                self.dim + 1 + j,
            ] = -1

        for j in range(self.nE):
            inG[
                self.p0 + 1 + (1 + self.pI).sum() + (1 + self.pE)[:j].sum() : self.p0
                + 1
                + (1 + self.pI).sum()
                + (1 + self.pE)[:j].sum()
                + self.pE[j]
                + 1,
                self.dim + 1 + self.nI + j,
            ] = -1
            inG[
                self.p0
                + 1
                + (1 + self.pI).sum()
                + (1 + self.pE).sum()
                + (1 + self.pE)[:j].sum() : self.p0
                + 1
                + (1 + self.pI).sum()
                + (1 + self.pE).sum()
                + (1 + self.pE)[:j].sum()
                + self.pE[j]
                + 1,
                self.dim + 1 + self.nI + j,
            ] = +1

        # we have nI+nE r-variables
        nonnegG = np.hstack(
            (np.zeros((self.nI + self.nE, self.dim + 1)), -np.eye(self.nI + self.nE))
        )
        nonnegh = np.zeros(self.nI + self.nE)

        return P, q, inG, inh, nonnegG, nonnegh

    def _update(self, H, rho, D_f, D_gI, D_gE, f_k, gI_k, gE_k):
        self.P[: self.dim, : self.dim] = H
        self.q = np.hstack((np.zeros(self.dim), rho, np.ones(self.nI), np.ones(self.nE)))

        self.inG[: self.p0 + 1, : self.dim] = D_f
        self.inh[: self.p0 + 1] = -f_k

        for j in range(self.nI):
            self.inG[
                self.p0 + 1 + (1 + self.pI)[:j].sum() : self.p0
                + 1
                + (1 + self.pI)[:j].sum()
                + self.pI[j]
                + 1,
                : self.dim,
            ] = D_gI[j]
            self.inh[
                self.p0 + 1 + (1 + self.pI)[:j].sum() : self.p0
                + 1
                + (1 + self.pI)[:j].sum()
                + self.pI[j]
                + 1
            ] = -gI_k[j]

        for j in range(self.nE):
            self.inG[
                self.p0 + 1 + (1 + self.pI).sum() + (1 + self.pE)[:j].sum() : self.p0
                + 1
                + (1 + self.pI).sum()
                + (1 + self.pE)[:j].sum()
                + self.pE[j]
                + 1,
                : self.dim,
            ] = D_gE[j]
            self.inG[
                self.p0
                + 1
                + (1 + self.pI).sum()
                + (1 + self.pE).sum()
                + (1 + self.pE)[:j].sum() : self.p0
                + 1
                + (1 + self.pI).sum()
                + (1 + self.pE).sum()
                + (1 + self.pE)[:j].sum()
                + self.pE[j]
                + 1,
                : self.dim,
            ] = -D_gE[j]

            self.inh[
                self.p0 + 1 + (1 + self.pI).sum() + (1 + self.pE)[:j].sum() : self.p0
                + 1
                + (1 + self.pI).sum()
                + (1 + self.pE)[:j].sum()
                + self.pE[j]
                + 1
            ] = -gE_k[j]
            self.inh[
                self.p0
                + 1
                + (1 + self.pI).sum()
                + (1 + self.pE).sum()
                + (1 + self.pE)[:j].sum() : self.p0
                + 1
                + (1 + self.pI).sum()
                + (1 + self.pE).sum()
                + (1 + self.pE)[:j].sum()
                + self.pE[j]
                + 1
            ] = gE_k[j]

        return

    def solve(
        self,
        H: np.ndarray,
        rho: float,
        D_f: np.ndarray,
        D_gI: List[np.ndarray],
        D_gE: List[np.ndarray],
        f_k: float,
        gI_k: np.ndarray,
        gE_k: np.ndarray,
    ) -> None:
        """
        This solves the quadratic program.

        Updates
        self.d: array
            search direction

        self.lambda_f: array
            KKT multipier for objective.

        self.lambda_gE: list
            KKT multipier for equality constraints.

        self.lambda_gI: list
            KKT multipier for inequality constraints.

        """

        # update params
        self._update(H, rho, D_f, D_gI, D_gE, f_k, gI_k, gE_k)

        A = np.vstack((self.inG, self.nonnegG))
        u = np.hstack((self.inh, self.nonnegh))
        l = np.ones_like(u) * (-np.inf)

        problem = osqp.OSQP()

        problem.setup(
            P=sparse.csc_matrix(self.P),
            q=self.q,
            A=sparse.csc_matrix(A),
            l=l,
            u=u,
            eps_abs=1e-05,
            eps_rel=1e-05,
            verbose=False,
            polish=1,
        )

        res = problem.solve()

        assert res.info.status in OSQP_ALLOWED_STATUS, f"OSQP results in status {res.info.status}."
        self._problem = problem

        primal_solution = res.x

        self.d = primal_solution[: self.dim]
        self.z = primal_solution[self.dim]

        self.rI = primal_solution[self.dim + 1 : self.dim + 1 + self.nI]
        self.rE = primal_solution[self.dim + 1 + self.nI :]

        # extract dual variables = KKT multipliers
        dual_solution = res.y
        lambda_f = dual_solution[: self.p0 + 1]

        lambda_gI = list()
        for j in np.arange(self.nI):
            start_ix = self.p0 + 1 + (1 + self.pI)[:j].sum()
            lambda_gI.append(dual_solution[start_ix : start_ix + 1 + self.pI[j]])

        lambda_gE = list()
        for j in np.arange(self.nE):
            start_ix = self.p0 + 1 + (1 + self.pI).sum() + (1 + self.pE)[:j].sum()

            # from ineq with +
            vec1 = dual_solution[start_ix : start_ix + 1 + self.pE[j]]

            # from ineq with -
            vec2 = dual_solution[
                start_ix + (1 + self.pE).sum() : start_ix + (1 + self.pE).sum() + 1 + self.pE[j]
            ]

            # see Direction.m line 620
            lambda_gE.append(vec1 - vec2)

        self.lambda_f = lambda_f.copy()
        self.lambda_gI = lambda_gI.copy()
        self.lambda_gE = lambda_gE.copy()

        return
