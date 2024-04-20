import warnings
import numpy as np

def Lydia(x0, niter_max, stepsize, oracle_f, oracle_gradf, xm1=None,fstar=None):
    """Args:
    x0: an initialization of size n
    niter_max: number of iterations to run
    stepsize: length of the gradient step to take
    oracle_f, oracle_gradf: functions taking x and return f(x) and gradf(x) respectively
    If x_{-1} not provided initialize at x0
    """

    ## Optional parameters
    xm1 = xm1 if xm1 is not None else x0
    if fstar is None:
        warnings.warn(('Warning: if known, consider providing optimal value fstar for optimal performance'), Warning)
        minf = 1e100

    ## Initialize
    x = x0
    list_values = [] ; list_E = []
    ## Iterate
    for niter in range(niter_max):
        # compute Lyapunov value
        f = oracle_f(x)
        # if fstar is unknown use best known value instead
        minf = min(minf,f) if fstar is None else fstar
        sqrtE = np.sqrt( f - minf + (1/2)*np.sum( (x-xm1)**2/stepsize, axis=-1) ) # compute Lyapunov value
        list_values.append(f)
        list_E.append(sqrtE**2) 
        if niter == 0:
            # store inital value of E, if fstar not provided use 0 by default
            sqrtE0 = np.copy(sqrtE) if fstar is not None else np.sqrt( f - 0. + (1/2)*np.sum( (x-xm1)**2/stepsize, axis=-1) ) # store initial first value
        # compute extrapolated point y
        y = x + (1.-sqrtE/sqrtE0) * (x-xm1)
        # update x
        xm1 = np.copy(x) # current x will become previous x
        gradf = oracle_gradf(y) # compute gradient at y
        x = y - stepsize * gradf # do a gradient step with extrapolated variable y

    ## Compute final values
    f = oracle_f(x)
    minf = min(minf, f) if fstar is None else fstar
    sqrtE = np.sqrt( f - minf + np.sum( (x-xm1)**2 / stepsize, axis=-1) ) # compute Lyapunov value
    list_values.append(f)
    list_E.append(sqrtE**2) 

    return list_values, list_E, x
