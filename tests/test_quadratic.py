import numpy as np
import torch

from ncopt.functions import ObjectiveOrConstraint
from ncopt.functions.quadratic import Quadratic
from ncopt.sqpgs.main import SQPGS

d = 3


def test_quadratic_problem():
    torch.manual_seed(2)
    params = (torch.eye(d), torch.zeros(d), torch.tensor(0.0))
    model = Quadratic(params=params)

    f = ObjectiveOrConstraint(model, dim=d, is_differentiable=True)
    g = ObjectiveOrConstraint(torch.nn.Linear(d, d), dim_out=d, is_differentiable=True)
    # make zero infeasible
    g.model.weight.data = -torch.eye(d)
    g.model.bias.data = torch.ones(d)

    gI = [g]
    gE = []

    options = {}
    x0 = (-1) * torch.ones(d).numpy()
    problem = SQPGS(
        f, gI, gE, x0=x0, tol=1e-5, max_iter=50, verbose=False, options=options, log_every=10
    )
    sol = problem.solve()

    assert np.allclose(sol, np.ones(d))
