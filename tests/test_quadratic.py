import torch

from ncopt.functions import ObjectiveOrConstraint
from ncopt.functions.quadratic import Quadratic
from ncopt.sqpgs.main import SQPGS

d = 10


def test_quadratic_problem():
    torch.manual_seed(2)
    params = (torch.eye(d), torch.zeros(d), torch.tensor(0.0))
    model = Quadratic(params=params)
    f = ObjectiveOrConstraint(model, dim=d, is_differentiable=True)
    g = ObjectiveOrConstraint(torch.nn.Linear(d, 2), dim_out=2, is_differentiable=True)
    # make zero infeasible
    g.model.bias.data = g.forward(torch.zeros(1, d)).squeeze() + 1.0

    gI = [g]
    gE = []

    options = {"qp_solver": "osqp-cvxpy"}
    problem = SQPGS(f, gI, gE, tol=1e-8, max_iter=200, verbose=True, options=options)
    _ = problem.solve()
