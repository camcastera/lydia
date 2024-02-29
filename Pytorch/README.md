This is the Pytorch version for the LYDIA algorithm introduced in the paper [Near-optimal Closed-loop Method via Lyapunov Damping for Convex Optimization](https://arxiv.org/abs/2311.10053)

The algorithm is available in 2 forms:

1. A [pytorch optimizer](https://github.com/camcastera/lydia/tree/master/LYDIA_optim) (as those implemented in torch.optim) which can simply be used like other optimizers via the ".step()" function.
2. A ["loop" implementation](https://github.com/camcastera/lydia/tree/master/Loop_version) where the optimizer is a python function that implements the whole optimization loop (like the [numpy](https://github.com/camcastera/lydia/tree/master/Numpy) version does) 

WARNING: To reduce computational cost, the pytorch optimizer version slightly differs from that introduced in the [paper](https://arxiv.org/abs/2311.10053). Namely, the gradient is evaluated at $x_k$ and not $y_k$ (i.e. no extrapolation). The loop version is faithful to the paper version.

The optimizer version can be used in the same fashion as the usual [Pytorch optimizers](https://pytorch.org/docs/stable/optim.html) with two small differences:

1. When initialized, the LYDIA optimizer (optionnally) takes the optimal (or best possible) value of the loss function. This is optional but recommended for optimal performance. If not provided it is set to zero.  
2. The step() function takes a "value" argument 

For the loop function, the algorithm takes the form of a python function (relying on pytorch) that takes the following parameters as input:

* x0: initial point (a numpy array of given dimension)
* niter_max: total number of iteration
* stepsize: the step-size to use for the algorithm
* oracle_f: a function taking a numpy array x as input and returning f(x)
* oracle_gradf: a function taking a numpy array x as input and returning gradf(x)
* xm1 (optional): previous (default: x0)
* fstar (optional): optimal value of the function f, required for optimal performance (default: 0)
