import numpy as np
import torch
import matplotlib.pyplot as plt

from LYDIA_optimizer import LYDIA

plt.close('all') # close opened figures if any

n = 20 # dimension of the problem

# Example 1
x0 = 10 * torch.ones(n)
xm1 = x0.clone()
D = torch.arange(1, n+1)
def oracle_f(x):
    return 1/2 * torch.sum(D*x**2, dim=-1)
stepsize = 1. / torch.max(D) # There is no global Lipschitz constant

# Example 2
'''
x0 = torch.ones(n)
xm1 = x0.clone()
D = torch.arange(1, n+1)
def oracle_f(x):
    return 1/4 * torch.sum(D*x**4, dim=-1)
stepsize = 1. / (3*torch.max(D)) # There is no global Lipschitz constant
'''

## Parameters of the algorithm
niter_max = 500
fstar = 0.
# fstar = None
x = x0.clone().detach().requires_grad_(True)

## Initialize LYDIA as an optimizer
optim = LYDIA([x], lr=stepsize, fstar=fstar) # Takes the variable to optimize, a step-size and optionally fstar

## Run the algorithm
list_values_lydia = []
list_E_lydia = []
list_iterations = np.arange(1, niter_max+1)
# Iterate
for niter in range(niter_max):
    # compute Lyapunov value
    f = oracle_f(x)
    list_values_lydia.append(f.item()) # store values to plot later
    # set grad to zero
    if x.grad is not None:
        with torch.no_grad(): # reset grad value
            x.grad.mul_(0.)
    # compute gradient (stored in x.grad)
    f.backward()
    # do an optimization step
    damping = optim.step(value=f) # The optimizer requires the current value of f
    list_E_lydia.append(damping**2)

#########################

## Plot results
fig, axes = plt.subplots(1, 2, figsize=(12,6.5))
ax = axes[0]
# Plot LYDIA
ax.plot(list_iterations, list_values_lydia, color='dodgerblue', lw=3, label='LYDIA')
# Plot 1/k and 1/k^2
ax.plot(list_iterations, list_values_lydia[0]/list_iterations, color='black',lw=2,ls='--',label=r'$k^{-1}$')
ax.plot(list_iterations, list_values_lydia[0]/list_iterations**2, color='red', lw=2, ls='--', label=r'$k^{-2}$')

ax.set_ylabel(r'$f(x_k)-f^{\star}$', fontsize=16)
ax.set_xlabel(r'iteration $k$', fontsize=16)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, niter_max+2)
ax.legend(loc='lower left', fontsize=16)

# Plot Lyapunov function
ax = axes[1]
# Plot LYDIA
ax.plot(list_iterations, list_E_lydia, color='dodgerblue', lw=3, label='LYDIA')
# Plot 1/k and 1/k^2
ax.plot(list_iterations, list_E_lydia[0]/list_iterations, color='black', lw=2, ls='--', label=r'$k^{-1}$')
ax.plot(list_iterations, list_E_lydia[0]/list_iterations**2, color='red', lw=2, ls='--', label=r'$k^{-2}$')

ax.set_ylabel(r'$E_k$', fontsize=16)
ax.set_xlabel(r'iteration $k$', fontsize=16)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, niter_max+2)
ax.legend(loc='lower left', fontsize=16)

fig.show()
