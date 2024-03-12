import cvxpy as cp
import numpy as np
import pytest

from ncopt.sqpgs.main import SubproblemSQPGS


@pytest.fixture
def subproblem_ineq() -> SubproblemSQPGS:
    dim = 2
    p0 = 2
    pI = np.array([3])
    pE = np.array([], dtype=int)
    assert_tol = 1e-5
    subproblem = SubproblemSQPGS(dim, p0, pI, pE, assert_tol)
    return subproblem


def test_subproblem_ineq(subproblem_ineq: SubproblemSQPGS):
    D_f = np.array([[-2.0, 1.0], [-2.04236205, -1.0], [-1.92172864, -1.0]])
    D_gI = [np.array([[0.0, 2.0], [0.0, 2.0], [1.41421356, 0.0], [1.41421356, 0.0]])]
    subproblem_ineq.solve(
        H=np.eye(2, dtype=float),
        rho=0.1,
        D_f=D_f,
        D_gI=D_gI,
        D_gE=[],
        f_k=1.0,
        gI_k=np.array([-1.0]),
        gE_k=np.array([], dtype=float),
    )
    assert subproblem_ineq.status == cp.OPTIMAL
    assert np.isclose(subproblem_ineq.objective_val, 0.080804459)


@pytest.fixture
def subproblem_eq() -> SubproblemSQPGS:
    dim = 2
    p0 = 2
    pI = np.array([], dtype=int)
    pE = np.array([4, 4])
    assert_tol = 1e-5
    subproblem = SubproblemSQPGS(dim, p0, pI, pE, assert_tol)
    return subproblem


def test_subproblem_eq(subproblem_eq: SubproblemSQPGS):
    D_gE = [
        np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
    ]
    subproblem_eq.solve(
        H=np.eye(2, dtype=float),
        rho=0.1,
        D_f=np.array([-2.0, 1.0]),
        D_gI=[],
        D_gE=D_gE,
        f_k=1.0,
        gI_k=np.array([], dtype=float),
        gE_k=np.array([-1.0, -1.0]),
    )
    assert subproblem_eq.status == cp.OPTIMAL
    assert np.isclose(subproblem_eq.objective_val, 1.0)
