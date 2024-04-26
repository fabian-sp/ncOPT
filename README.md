# ncOPT

## Short Description
This repository is for solving **constrained optimization problems** where objective and/or constraint functions are (arbitrary) **Pytorch modules**. It is mainly intended for optimization with pre-trained networks, but might be useful also in other contexts.

The algorithms in this package can solve problems of the form

```
    min  f(x)
    s.t. g(x) <= 0
         h(x) = 0
```

where `f`, `g` and `h` are locally Lipschitz functions.

The package supports:

* models/functions on GPU
* batched evaluation and Jacobian computation
* ineqaulity and equality constraints, which can depend only on a subset of the optimization variable

**DISCLAIMER:** 

1) We have not (yet) extensively tested the solver on large-scale problems.  
2) The implemented solver is designed for nonsmooth, nonconvex problems, and as such, can solve a very general problem class. If your problem has a specific structure (e.g. convexity), then you will almost certainly get better performance by using software/solvers that are specifically written for the respective problem type. As starting point, check out [`cvxpy`](https://www.cvxpy.org/).



## Installation

For an editable version of this package in your Python environment, run the command

```
    python -m pip install --editable .
```


## Main Solver 

The main solver implemented in this package is called SQP-GS, and has been developed by Curtis and Overton in [1]. 
The SQP-GS algorithm can solve problems with nonconvex and nonsmooth objective and constraints. For details, we refer to [our documentation](src/ncopt/sqps/README.md) and the original paper [1].


## Getting started

### Solver interface
The solver can be called via 

```python
    from ncopt.sqpgs import SQPGS
    problem = SQPGS(f, gI, gE)
    problem.solve()
```
Here `f` is the objective function, and `gI` and `gE` are a list of inequality and equality constaints. 
The objective `f` and each element of `gI` and `gE` should be passed as an instance of [`ncopt.functions.ObjectiveOrConstraint`](src/ncopt/functions/main.py) (a simple wrapper around a `torch.nn.Module`). 

* Each constraint function is allowed to have multi-dimensional output (see example below).
* An empty list can be passed if no (in)equality constraints are needed.

For example, a linear constraint function `Ax - b <= 0` can be implemented as follows:

```python
    from ncopt.functions import ObjectiveOrConstraint
    A = ..                      # your data
    b = ..                      # your data
    g = ObjectiveOrConstraint(torch.nn.Linear(2, 2), dim_out=2)
    g.model.weight.data = A    # pass A
    g.model.bias.data = -b     # pass b
```

Note the argument `dim_out`, which needs to be passed for all constraint functions: it tells the solver what the output dimension of this constraint is.

### Example

A full example for solving a nonsmooth Rosenbrock function, constrained with a maximum function can be found [here](example_rosenbrock.py). This example is taken from Example 5.1 in [1]. The picture below shows the trajectory of the SQP-GS solver for different starting points. The final iterates are marked with the black plus while the analytical solution is marked with the golden star. We can see that the algorithm finds the minimizer consistently.

![SQP-GS trajectories for a 2-dim example](data/img/rosenbrock.png "SQP-GS trajectories for a 2-dim example")



## References
[1] Frank E. Curtis and Michael L. Overton, A sequential quadratic programming algorithm for nonconvex, nonsmooth constrained optimization, 
SIAM Journal on Optimization 2012 22:2, 474-500, https://doi.org/10.1137/090780201.