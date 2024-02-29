import torch

def Lydia(x0,niter_max,stepsize,oracle_f,oracle_gradf,xm1=None,fstar=None):
    ### Requires:
    #x0: an initialization of size n
    #niter_max: number of iterations to run
    #stepsize: length of the gradient step to take
    #oracle_f, oracle_gradf: functions taking x and return f(x) and gradf(x) respectively
    #If x_{-1} not provided initialize at x0

    ###Optional parameters
    xm1 = xm1 if xm1 is not None else x0
    if fstar is None:
        print('Warning: if known, consider providing optimal value fstar for optimal performance')
        minf=1e100

    ### Initialize
    x = x0 #initialize
    list_values = [] ; list_E = []
    ### Iterate
    for niter in range(niter_max):
        ##Compute Lyapunov value
        f = oracle_f(x)
        #If fstar is unknown use best known value instead
        with torch.no_grad():
            minf = min(minf,f) if fstar is None else fstar
            sqrtE = torch.sqrt( f - minf + (1/2)*torch.sum( (x-xm1)**2/stepsize, dim=-1) ) #Compute Lyapunov value
            list_values.append(f) ; list_E.append(sqrtE**2) #store values of f
            if niter==0:
                # store inital value of E, if fstar not provided use 0 by default
                sqrtE0=sqrtE.clone().detach() if fstar is not None else torch.sqrt( f - 0. + (1/2)*torch.sum( (x-xm1)**2/stepsize, dim=-1) ) #Store initial first value
        ##Compute extrapolated point y
        y = x + (1.-sqrtE/sqrtE0)*(x-xm1)
        ##Update x
        xm1 = x.clone().detach() #Current x will become previous x
        gradf = oracle_gradf(y,oracle_f) #compute gradient at y
        x = y - stepsize*gradf #do a gradient step with extrapolated variable y

    ## Compute final values
    f = oracle_f(x)
    minf = min(minf,f) if fstar is None else fstar
    sqrtE = torch.sqrt( f - minf + torch.sum( (x-xm1)**2/stepsize, dim=-1) ) #Compute Lyapunov value
    list_values.append(f) ; list_E.append(sqrtE**2) #store values of f

    return list_values, list_E, x


