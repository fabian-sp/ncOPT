# ncOPT
This repository contains a `Python` implementation of the SQP-GS (*Sequential Quadratic Programming - Gradient Sampling*) algorithm by Curtis and Overton [1]. 

**Note:** this implementation is a **prototype code**, it has been tested only for a simple problem and it is not performance-optimized. A Matlab implementation is available from the authors of the paper, see [2].

## Mathematical description

The algorithm can solve problems of the form

    min  f(x)
    s.t. g(x) <= 0
         h(x) = 0

where `f`, `g` and `h` are locally Lipschitz functions. Hence, the algorithm can solve problems with nonconvex and nonsmooth objective and constraints. For details, we refer to the original paper.

## Example

The code was tested for a 2-dim nonsmooth version of the Rosenbrock function, constrained with a maximum function. See Example 5.1 in [1]. For this problem, the analytical solution is known. The picture below shows the trajectory of SQP-GS for different starting points. The final iterates are marked with the black plus while the analytical solution is marked with the golden star. We can see that the algorithm finds the minimizer consistently.

To reproduce this experiment, see the file `test_rosenbrock.py`.

![SQP-GS trajectories for a 2-dim example](rosenbrock.png "SQP-GS trajectories for a 2-dim example")


## Implementation details

The solver has three main arguments, called `f`, `gI` and `gE`. The first is the objective while the latter are lists of inequality and equality constraint functions. Each element of `gI` and `gE` should be a scalar function. Each element of `gI` and `gE` as well as the objective `f` needs to be an instance of a class which contains the following properties.

### Attributes

* `self.dim`: integer, specifies dimension of the input argument.

### Methods

* `self.eval`: evaluates the function at a point `x`.
* `self.grad`: evaluates the gradient at a point `x`.

For an example, see the classes defined in `ncopt/rosenbrock.py`. Moreover, we implemented a class for a constraint coming from a Pytorch neural network (i.e. `g_i(x)` is an already trained neural network). For this, see `ncopt/torch_obj.py`.



## References
* [1] F. E. Curtis and M. L. Overton, A sequential quadratic programming algorithm for nonconvex, nonsmooth constrained optimization, SIAM Journal on Optimization, 22 (2012), pp. 474–500, https://doi.org/10.1137/090780201.

* [2] F. E. Curtis, MATLAB implementation of SLQP-GS, https://coral.ise.lehigh.edu/frankecurtis/software/.
