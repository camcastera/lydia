import numpy as np
import matplotlib.pyplot as plt

from LYDIA import Lydia

plt.close('all') # close opened figures if any

n = 20 # dimension of the problem

x0 = 10 * np.ones(n)
xm1 = np.copy(x0)
D = np.arange(1, n+1)
def oracle_f(x):
    return 1/2 * np.sum(D*x**2, axis=-1)
def oracle_gradf(x):
    return D * x

## Parameters of the algorithm
niter_max = 500
fstar = 0.
# fstar = None
stepsize = 1. / np.max(D) # there is no global Lipschitz constant

## Run the algorithm
list_values_Lydia, list_E_lydia, _ = Lydia(
    x0=x0,
    niter_max=niter_max,
    stepsize=stepsize,
    oracle_f=oracle_f,
    oracle_gradf=oracle_gradf,
    xm1=xm1,
    fstar=fstar
    )

list_iterations = np.arange(1, niter_max+2)

## Plot results
fig, axes = plt.subplots(1,2, figsize=(12,6.5))
ax = axes[0]
# Plot LYDIA
ax.plot(list_iterations, list_values_Lydia, color='dodgerblue', lw=3, label='LYDIA')
# Plot 1/k and 1/k^2
ax.plot(list_iterations, list_values_Lydia[0]/list_iterations, color='black', lw=2, ls='--', label=r'$k^{-1}$')
ax.plot(list_iterations, list_values_Lydia[0]/list_iterations**2, color='red', lw=2, ls='--', label=r'$k^{-2}$')

ax.set_ylabel(r'$f(x_k)-f^{\star}$', fontsize=16)
ax.set_xlabel(r'iteration $k$', fontsize=16)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1, niter_max+2)
ax.legend(loc='lower left', fontsize=16)

## Plot Lyapunov function
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
