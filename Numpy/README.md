This is the python code for the LYDIA algorithm introduced in the paper [Near-optimal Closed-loop Method via Lyapunov Damping for Convex Optimization](https://arxiv.org/abs/2311.10053)

The algorithm is available in two forms: a Pytorch optimizer to use in the same fashion as the usual [Pytorch optimizers](https://pytorch.org/docs/stable/optim.html) and a python function (relying on either [numpy](https://github.com/camcastera/lydia/tree/master/Numpy) or [pytorch](https://github.com/camcastera/lydia/tree/master/Pytorch)) that takes the following parameters as input:

* x0: initial point (a numpy array of given dimension)
* niter_max: total number of iteration
* stepsize: the step-size to use for the algorithm
* oracle_f: a function taking a numpy array x as input and returning f(x)
* oracle_gradf: a function taking a numpy array x as input and returning gradf(x)
* xm1 (optional): previous (default: x0)
* fstar (optional): optimal value of the function f, required for optimal performance (default: 0)
