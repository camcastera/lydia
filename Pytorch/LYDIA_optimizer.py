## @author: Camille Castera
## inspired from the pytorch implementation of SGD (https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD)
import warnings

import torch
from torch.optim import Optimizer

class LYDIA(Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0, fstar=None):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if fstar is None:
            fstar = 0
            warnings.warn(('if known, consider providing optimal value fstar for optimal performance'), Warning)
        self.damping0 = None # computed later (to normalize)
        defaults = {'lr': lr, 'fstar': fstar, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad() # The function below is not part of the graph
    def compute_Lyapunov_damping(self, value, stepsize, fstar, weight_decay):
        diff_square = 0.
        param_L2_norm = 0. # if weight_decay>0 then also compute squared-norm of parameters
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                sum_along = list(range(len(p.shape))) # sum along all dimensions
                # compute (x_k-x_{k-1})**2
                diff_square += torch.sum((p-state['previous_step'])**2, dim=sum_along)
                if weight_decay>0:
                    param_L2_norm += torch.sum(p**2, dim=sum_along)
        return torch.sqrt( (value + weight_decay/2*param_L2_norm - fstar) + 1/(2*stepsize) * diff_square )

    @torch.no_grad() # The function below is not part of the graph
    def step(self, value=None):
        """Performs a single optimization step.
        Args:
            value: current value of the loss
        """
        if value is None:
            raise ValueError("Please provide the current value of the loss function via .step(value=loss) where loss is a scalar.")

        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['previous_step'] = p.clone().detach()

        # Compute Lyapunov value
        damping = self.compute_Lyapunov_damping(
            value=value,
            stepsize=self.defaults['lr'], 
            fstar=self.defaults['fstar'], 
            weight_decay=self.defaults['weight_decay']
            )

        # Store initial damping
        if self.damping0 is None:
            self.damping0 = damping

        ## Update parameters
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if self.defaults['weight_decay'] > 0:
                    grad += self.defaults['weight_decay'] * p

                state = self.state[p]
                temp = p.clone().detach() # store p to update previous step later

                p.add_( (1. - damping/self.damping0)*(p-state["previous_step"]) - self.defaults['lr']*grad ) #x_{k+1} = x_k - (1-sqrt(Ek/E0))*(x_k-x_{k-1}) - s*grad

                state['previous_step'].mul_(0.).add_(temp) # store new previous_step
                state['step'] += 1

        return damping
