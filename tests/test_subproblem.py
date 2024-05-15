import cvxpy as cp
import numpy as np
import pytest

from ncopt.sqpgs.cvxpy_subproblem import CVXPYSubproblemSQPGS
from ncopt.sqpgs.osqp_subproblem import OSQPSubproblemSQPGS


@pytest.fixture
def subproblem_ineq() -> CVXPYSubproblemSQPGS:
    dim = 2
    p0 = 2
    pI = np.array([3])
    pE = np.array([], dtype=int)
    assert_tol = 1e-5
    subproblem = CVXPYSubproblemSQPGS(dim, p0, pI, pE, assert_tol)
    return subproblem


def test_subproblem_ineq(subproblem_ineq: CVXPYSubproblemSQPGS):
    D_f = np.array([[-2.0, 1.0], [-2.04236205, -1.0], [-1.92172864, -1.0]])
    D_gI = [np.array([[0.0, 2.0], [0.0, 2.0], [1.41421356, 0.0], [1.41421356, 0.0]])]
    subproblem_ineq.solve(
        L=np.eye(2, dtype=float),
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
def subproblem_eq() -> CVXPYSubproblemSQPGS:
    dim = 2
    p0 = 2
    pI = np.array([], dtype=int)
    pE = np.array([4, 4])
    assert_tol = 1e-5
    subproblem = CVXPYSubproblemSQPGS(dim, p0, pI, pE, assert_tol)
    return subproblem


def test_subproblem_eq(subproblem_eq: CVXPYSubproblemSQPGS):
    D_gE = [
        np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
    ]
    subproblem_eq.solve(
        L=np.eye(2, dtype=float),
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


def test_subproblem_consistency_ineq():
    dim = 2
    p0 = 2
    pI = np.array([3])
    pE = np.array([], dtype=int)
    assert_tol = 1e-5
    subproblem = CVXPYSubproblemSQPGS(dim, p0, pI, pE, assert_tol)
    subproblem2 = OSQPSubproblemSQPGS(dim, p0, pI, pE, assert_tol)
    D_f = np.array([[-2.0, 1.0], [-2.04236205, -1.0], [-1.92172864, -1.0]])
    D_gI = [np.array([[0.0, 2.0], [0.0, 2.0], [1.41421356, 0.0], [1.41421356, 0.0]])]

    subproblem.solve(
        L=np.eye(dim, dtype=float),
        rho=0.1,
        D_f=D_f,
        D_gI=D_gI,
        D_gE=[],
        f_k=1.0,
        gI_k=np.array([-1.0]),
        gE_k=np.array([], dtype=float),
    )

    subproblem2.solve(
        H=np.eye(dim, dtype=float),
        rho=0.1,
        D_f=D_f,
        D_gI=D_gI,
        D_gE=[],
        f_k=1.0,
        gI_k=np.array([-1.0]),
        gE_k=np.array([], dtype=float),
    )

    assert np.allclose(subproblem.d, subproblem2.d)
    assert np.allclose(subproblem.lambda_f, subproblem2.lambda_f)
    assert np.allclose(subproblem.lambda_gI, subproblem2.lambda_gI)
    assert np.allclose(subproblem.lambda_gE, subproblem2.lambda_gE)


def test_subproblem_consistency_ineq_eq():
    dim = 4
    p0 = 2
    pI = np.array([1])
    pE = np.array([1])
    assert_tol = 1e-5
    subproblem = CVXPYSubproblemSQPGS(dim, p0, pI, pE, assert_tol)
    subproblem2 = OSQPSubproblemSQPGS(dim, p0, pI, pE, assert_tol)
    D_f = np.array(
        [
            [1.83529234, 2.51893663, -0.87507966, 0.53305111],
            [0.7042857, -0.19426588, 1.26820232, 0.59255224],
            [-0.87356341, -0.24994689, -0.82073493, -1.18734854],
        ]
    )

    D_gI = [
        np.array(
            [
                [-1.13249659, 0.67854141, -0.0316317, -1.37963152],
                [-1.64858759, 0.65296873, -0.72343526, 0.60976315],
            ]
        )
    ]

    D_gE = [
        np.array(
            [
                [1.05532136, -0.12589961, 0.49469938, 0.22879848],
                [2.10668334, -0.00816628, -0.43333072, 0.22656999],
            ]
        )
    ]

    subproblem.solve(
        L=np.eye(dim, dtype=float),
        rho=0.1,
        D_f=D_f,
        D_gI=D_gI,
        D_gE=D_gE,
        f_k=1.0,
        gI_k=np.array([-1.0]),
        gE_k=np.array([-1.0]),
    )

    subproblem2.solve(
        H=np.eye(dim, dtype=float),
        rho=0.1,
        D_f=D_f,
        D_gI=D_gI,
        D_gE=D_gE,
        f_k=1.0,
        gI_k=np.array([-1.0]),
        gE_k=np.array([-1.0]),
    )

    assert np.allclose(subproblem.d, subproblem2.d)
    assert np.allclose(subproblem.lambda_f, subproblem2.lambda_f)
    assert np.allclose(subproblem.lambda_gI, subproblem2.lambda_gI)
    assert np.allclose(subproblem.lambda_gE, subproblem2.lambda_gE)
