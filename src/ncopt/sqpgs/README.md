# The SQP-GS Algorithm

SQP-GS is a method for solving nonsmooth, nonconvex constrained optimization problems. It has been proposed in [1]. 
As the name suggests, it combines Sequential Quadratic Programming (SQP) with Gradient Sampling (GS) to handle nonconvexity as well as nonsmoothness. 

An example for how to call the SQP-GS solver:

```python
from ncopt.sqpgs import SQPGS

problem = SQPGS(f, gI, gE, x0=None, tol=1e-6, max_iter=100, verbose=True)
x = problem.solve()
```

## How SQP-GS works in short

Below we briefly describe the SQP-GS algorithm. **For more details on the algorithm, we refer to the paper [1].** The iteration cost of the algorithm splits mainly into two steps:

1) Evaluate and compute gradient/Jacobian for each function at multiple points in a neighborhood of the current iterate.

2) Approximate the original problem by a quadratic program (QP). Solve this QP to compute the update direction. 

The technique in 1. is called Gradient Sampling (GS) and is a widely used, robust technique for handling nonsmooth objective or constraint functions. 
As all functions are Pytorch modules in this package, this amounts to **batch evaluation and Jacobian computation**. This can be done efficiently using the `autograd` functionalities of Pytorch.

For 2., we solve the QP with the package `osqp`. We also implement a general interface to `cvxpy`, which seems slightly slower due to overhead costs, but more flexible as the solver can be exchanged easily. 
Further, the quadratic approximation of SQP naturally involves an approximation of the Hessian, which is done in L-BFGS style.



### Options

SQP-GS has many hyperparameters, for all of which we use the default values from the paper [1] and the Matlab code [2]. The values can be controlled via the argument `options`, which can be a dictionary with all values that need to be overwritten. See the default values [here](defaults.py).

One important hyperparameter is the number of sample points, see the next section for details.

The starting point can be specified with the argument `x0`, and we use the zero vector if not specified.

### Gradient Sampling with `autograd`

For the objective and each constraint function, in every iteration, a number of points is sampled in a neighborhood to the currrent iterate. The function is then evaluated at those points, plus at the iterate itself, and the Jacobian of the function is computed at all points.
This is done with the Pytorch `autograd` functionalities `jacrev` and `vmap`. See the function [`compute_value_and_jacobian`](main.py#L435).

* If a function is differentiable, setting `is_differentiable=True` when calling `ObjectiveOrConstraint` will have the effect that for this function only the current iterate itself is evaluated. 
* The authors of [1] suggest to set the number of sample points to roughly 2 times the (effective!) input dimension of each function. See end of section 4 in [1]. As this is hard to implement by default, we recommend to set the number of points manually. This can be done by adjusting (before calling `problem.solve()`) the values `problem.p0` (number of points for objective), `problem.pI_` (number of points for each inequality constraint) and `problem.pE_` (number of points for each equality constraint).

### Solving the QP

Solving the QP will likely amount to most of the runtime per iteration. There are two options for solving the QP:

1. (Default): Use the `osqp` solver [4]. This calls directly the `osqp` solver after constructing the subproblem. See the [implementation](osqp_subproblem.py).
2. Use `cvxpy` [3]. This creates the subproblem with `cvxpy` and then solves the QP with the specified solver. This options has some overhead costs, but it allows to exchange the solver flexibly. See the [implementation](cvxpy_subproblem.py).  

The authors of [1] report that MOSEK works well for their numerical examples, but here we use `osqp` by default.


### Further notes


* For numerical stability, we add a Tikhonov regularization to the approximate Hessian. It's magnitude can be controlled over the key `reg_H` in `options`.

## References
[1] Frank E. Curtis and Michael L. Overton, A sequential quadratic programming algorithm for nonconvex, nonsmooth constrained optimization, 
SIAM Journal on Optimization 2012 22:2, 474-500, https://doi.org/10.1137/090780201.

[2] Frank E. Curtis and Michael L. Overton, MATLAB implementation of SLQP-GS, https://coral.ise.lehigh.edu/frankecurtis/software/.

[3] Steven Diamond and Stephen Boyd, CVXPY: A Python-embedded modeling language for convex optimization, Journal of Machine Learning Research 2016, https://www.cvxpy.org/.

[4] Bartolomeo Stellato and Goran Banjac and Paul Goulart and Alberto Bemporad and Stephen Boyd, OSQP: an operator splitting solver for quadratic programs, Mathematical Programming Computation 2020, https://osqp.org/docs/index.html.

