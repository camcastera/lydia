## This optimizer is adapted from the pytorch implementation of SGD (https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD)


import torch
from torch.optim import Optimizer



class LYDIA(Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0, fstar=None):
        print('TODO: need to implement the case where fstar is unknown')
        print('TODO: implement weight_decay')
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        self.damping0 = None #Computed later (to normalize)
        defaults = dict(lr=lr, fstar=fstar, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad() #The function below is not part of the graph
    def compute_Lyapunov_damping(self,value,stepsize,fstar):
        diff_square = 0.
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # Compute (x_k-x_{k-1})**2
                diff_square += torch.sum( (p-state["previous_step"])**2, dim=-1)
        return torch.sqrt( (value - fstar) + 1/(2*stepsize) * diff_square )

    @torch.no_grad() #The function below is not part of the graph
    def step(self,value):
        """Performs a single optimization step.

        Args:
            value: current value of the loss
        """


        ## State initialization
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["previous_step"] = p.clone().detach()


        ## Compute Lyapunov value
        damping = self.compute_Lyapunov_damping(value=value,
            stepsize=self.defaults['lr'], fstar=self.defaults['fstar'])

        #Store initial damping
        if self.damping0 is None:
            self.damping0 = damping

        ##Update parameters
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                temp = p.clone().detach() #Store p to update previous step later for later


                p.add_( (1.-damping/self.damping0)*(p-state["previous_step"]) - self.defaults['lr']*grad  ) #x_{k+1} = x_k - (1-sqrt(Ek/E0))*(x_k-x_{k-1}) - s*grad

                state["previous_step"].mul_(0.).add_(temp) #store new previous_step
                state["step"] += 1

        return damping



