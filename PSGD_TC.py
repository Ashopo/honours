import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional
import numpy as np
from scipy.stats import levy_stable
    
class PSGD_TC(Optimizer):
    """Implements stochastic gradient descent (optionally with momentum) WITH PERIODIC BOUNDARY CONDITIONS"""

    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False, foreach: Optional[bool] = None,
                 differentiable=False, func,
                 height: float, width: float, n_epochs: int):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        self.record = {'lr': lr, 'height': height, 'width': width, 'momentum': momentum}
        self.objfunc = func
        self.history = []
        self.alpha_record = []
        self.height = height
        self.width_denom = -0.5*(1/width)**2
        self.n_epochs = n_epochs    
        self.steps = 0

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,
                        differentiable=differentiable)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(PSGD_TC, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (Callable, optional): A closure that reevaluates the model
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
            has_sparse_grad = False

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        has_sparse_grad = True

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            self.sgd(params_with_grad,
                     d_p_list,
                     momentum_buffer_list,
                     weight_decay=group['weight_decay'],
                     momentum=group['momentum'],
                     lr=group['lr'],
                     dampening=group['dampening'],
                     nesterov=group['nesterov'],
                     maximize=group['maximize'],
                     has_sparse_grad=has_sparse_grad,
                     foreach=group['foreach'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


    def sgd(self,
            params: List[Tensor],
            d_p_list: List[Tensor],
            momentum_buffer_list: List[Optional[Tensor]],
            # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
            # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
            has_sparse_grad: bool = None,
            foreach: bool = None,
            *,
            weight_decay: float,
            momentum: float,
            lr: float,
            dampening: float,
            nesterov: bool,
            maximize: bool):
        r"""Functional API that performs SGD algorithm computation.
        See :class:`~torch.optim.SGD` for details.
        """

        if foreach is None:
            # Placeholder for more complex foreach logic to be added when value is not set
            foreach = False

        if foreach and torch.jit.is_scripting():
            raise RuntimeError('torch.jit.script not supported with foreach optimizers')

        if foreach and not torch.jit.is_scripting():
            func = self._multi_tensor_sgd
        else:
            func = self._single_tensor_sgd

        func(params,
            d_p_list,
            momentum_buffer_list,
            weight_decay=weight_decay,
            momentum=momentum,
            lr=lr,
            dampening=dampening,
            nesterov=nesterov,
            has_sparse_grad=has_sparse_grad,
            maximize=maximize)

    def _single_tensor_sgd(self,
                        params: List[Tensor],
                        d_p_list: List[Tensor],
                        momentum_buffer_list: List[Optional[Tensor]],
                        *,
                        weight_decay: float,
                        momentum: float,
                        lr: float,
                        dampening: float,
                        nesterov: bool,
                        maximize: bool,
                        has_sparse_grad: bool):

        for i, param in enumerate(params):
            d_p = d_p_list[i] if not maximize else -d_p_list[i]

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
            
            alpha = self.adapt_alpha()
            param.add_(d_p, alpha=-lr)
            param.add_(self.levy_noise(param, alpha, d_p, lr), alpha=lr)
            # Inbuilt periodic boundary conditions to ensure results are consistent.
            param = self.objfunc.pbc(param)
    
    def levy_noise(self, param, alpha, grad, lr):
        dim = param.size(dim=0)
        direction = hypersphere_sample(dim)
        
        alpha = 2 if alpha >= 2 else alpha
        scale = 0
        #scale = 0.05/alpha
        levy_r = levy_stable.rvs(alpha, 0, scale=scale, size=dim)
        levy_r = np.sqrt(sum(levy_r**2))

        #print('ADAM ratio:', levy_r/torch.norm(grad), end=' ')
        print(scale, levy_r, levy_r/scale, alpha)
        noise = levy_r * direction
        return noise

    def adapt_alpha(self):
        
        current = self.history[-1]
        past = self.history[0:-1]
        Vbias = 0

        for p in past:
            v = current - p
            Vbias += torch.exp(self.width_denom * torch.dot(v, v.T))
        
        Vbias = float(self.height * Vbias)
        alpha = 0.5 + Vbias/self.steps
        self.alpha_record.append(alpha)

        return alpha
    def _multi_tensor_sgd(self,
                          params: List[Tensor],
                          grads: List[Tensor],
                          momentum_buffer_list: List[Optional[Tensor]],
                          *,
                          weight_decay: float,
                          momentum: float,
                          lr: float,
                          dampening: float,
                          nesterov: bool,
                          maximize: bool,
                          has_sparse_grad: bool):
        
        raise RuntimeError("Multi-tensor PSGD_TC not supported yet.")

def hypersphere_sample(dim):
    
    direction = np.random.normal(size=dim)
    direction = np.sqrt(sum(direction**2))**(-1) * direction
    
    return torch.from_numpy(direction)

    #https://mathworld.wolfram.com/HyperspherePointPicking.html


def powlaw_samp(x_min, alpha, size=1):
    """
    Samples from powerlaw dist with min value x_min.
    """
    r = np.random.random(size=size)
    samp = x_min * (1 - r) ** (1 / (1-alpha))
    
    if size == 1:
        return float(samp)
    else:
        return samp

    # https://stats.stackexchange.com/questions/173242/random-sample-from-power-law-distribution
    # https://arxiv.org/pdf/0706.1062.pdf