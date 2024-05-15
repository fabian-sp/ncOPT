class Dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


_defaults = {
    "tol": 1e-8,
    "max_iter": 100,
    "verbose": True,
    "assert_tol": 1e-5,
    "store_history": False,
    "log_every": 10,
}

_option_defaults = {
    "eps": 1e-1,
    "rho": 1e-1,
    "theta": 1e-1,
    "eta": 1e-8,
    "gamma": 0.5,
    "beta_eps": 0.5,
    "beta_rho": 0.5,
    "beta_theta": 0.8,
    "nu": 10,
    "xi_s": 1e3,
    "xi_y": 1e3,
    "xi_sy": 1e-6,
    "iter_H": 10,
    "num_points_obj": 4,
    "num_points_gI": 4,
    "num_points_gE": 4,
    "qp_solver": "osqp",
    "reg_H": 1e-03,
}

DEFAULT_ARG = Dotdict(_defaults)
DEFAULT_OPTION = Dotdict(_option_defaults)
