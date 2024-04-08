# ncOPT

This repository is designed for solving constrained optimization problems where objetive and/or constraint functions are Pytorch modules. It is mainly intended for optimization with pre-trained networks, but could be used for other purposes.

As main solver, this repository contains a `Python` implementation of the SQP-GS (*Sequential Quadratic Programming - Gradient Sampling*) algorithm by Curtis and Overton [1]. 

**DISCLAIMER:** 

1) This implementation is a **prototype code**, it has been tested only for a simple problem and it is not (yet) performance-optimized.
2) The implemented solver is designed for nonsmooth, nonconvex problems, and as such, can solve a very general problem class. If your problem has a specific structure (e.g. convexity), then you will almost certainly get better performance by using software/solvers that make use of this structure. As starting point, check out [`cvxpy`](https://www.cvxpy.org/).
3) The main algorithm SQP-GS has been developed by Curtis and Overton in [1]. A Matlab implementation is available from the authors of the paper, see [2].

## Installation

If you want to install an editable version of this package in your Python environment, run the command

```
    python -m pip install --editable .
```

## Mathematical description

The algorithm can solve problems of the form

```
    min  f(x)
    s.t. g(x) <= 0
         h(x) = 0
```

where `f`, `g` and `h` are locally Lipschitz functions. The SQP-GS algorithm can solve problems with nonconvex and nonsmooth objective and constraints. For details, we refer to the original paper.

## Implementation details
The solver can be called via 

```python
    from ncopt.sqpgs import SQPGS
    problem = SQPGS(f, gI, gE)
    problem.solve()
```
The three main arguments, called `f`, `gI` and `gE`, are the objective, the inequality and equality constaints respectively. Each argument should be passed as a list of instances of `ncopt.functions.ObjectiveOrConstraint` (see example below). 

* Each constraint function is allowed to have multi-dimensional output (see example below).
* An empty list can be passed if no (in)equality constraints are needed.

For example, a linear constraint function `Ax <= b` could be implmented as follows:

```python
    from ncopt.functions import ObjectiveOrConstraint
    A = ..                      # your data
    b = ..                      # your data
    g = ObjectiveOrConstraint(torch.nn.Linear(2, 2), dim_out=2)
    g.model.weight.data = A    # pass A
    g.model.bias.data = -b     # pass b
```

Note the argument `dim_out`, which needs to be passed for all constraint functions: it tells the solver what the output dimension of this constraint is.

The main function class is `ncopt.functions.ObjectiveOrConstraint`. It is a simple wrapper around a given Pytorch module (e.g. the checkpoint of your trained network). We can evaluate the function and compute gradient using the standard Pytorch `autograd` functionalities. 


## Example

The code was tested for a 2-dim nonsmooth version of the Rosenbrock function, constrained with a maximum function. See Example 5.1 in [1]. For this problem, the analytical solution is known. The picture below shows the trajectory of SQP-GS for different starting points. The final iterates are marked with the black plus while the analytical solution is marked with the golden star. We can see that the algorithm finds the minimizer consistently.

To reproduce this experiment, see the file `example_rosenbrock.py`.

![SQP-GS trajectories for a 2-dim example](data/img/rosenbrock.png "SQP-GS trajectories for a 2-dim example")


## References
[1] Frank E. Curtis and Michael L. Overton, A sequential quadratic programming algorithm for nonconvex, nonsmooth constrained optimization, 
SIAM Journal on Optimization 2012 22:2, 474-500, https://doi.org/10.1137/090780201.

[2] Frank E. Curtis and Michael L. Overton, MATLAB implementation of SLQP-GS, https://coral.ise.lehigh.edu/frankecurtis/software/.
