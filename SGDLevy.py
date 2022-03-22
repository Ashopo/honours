import numpy as np
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required
from typing import List, Optional
from scipy.stats import levy_stable

class SGDLevy(Optimizer):

    """
    Modified PyTorch SGD to include Levy noise dynamics.
    """

    def __init__(self, 
                params, 
                lr: float = required,
                noise_pc: float = required,
                alpha: float = required,
                momentum: float = 0, 
                dampening: float = 0,
                weight_decay: float = 0, 
                nesterov: bool = False
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(lr=lr, noise_pc=noise_pc, alpha=alpha, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)

        super(SGDLevy, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            _single_tensor_sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                noise_pc = group['noise_pc'],
                alpha = group['alpha'],
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov']
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

def _single_tensor_sgd(params: List[Tensor],
                       d_p_list: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]],
                       noise_pc: float,
                       alpha: float,
                       *,
                       weight_decay: float,
                       momentum: float,
                       lr: float,
                       dampening: float,
                       nesterov: bool
):
    
    """
    Implements the actual sgd calculation.
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf
        #print(d_p)

        
        lv_noise = torch.Tensor(levy_stable.rvs(alpha=alpha, beta=0, size=len(d_p)))

        # normalise the levy noise to a percentage of the update vector magnitude
        # use the Langevin Monte Carlo algorithm (LMC)
        lv_noise = lv_noise * np.sqrt(2*noise_pc/lr) * torch.norm(d_p)
        d_p.add_(lv_noise)

        param.add_(d_p, alpha=-lr)