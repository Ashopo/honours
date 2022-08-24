import math
from pyrsistent import v
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional
import numpy as np
from scipy.stats import levy_stable
import time

class ADAM_TC(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, *, foreach: Optional[bool] = None,
                 maximize: bool = False, capturable: bool = False, func = None,
                 height: float, width: float, n_epochs: int):
        
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        self.record = {'lr': lr, 'height': height, 'width': width, 'betas': betas}
        self.objfunc = func
        self.history = []
        self.alpha = 2 - height
        self.Vbias = 0
        self.alpha_record = []
        self.height = height
        self.width_denom = -0.5*(1/width)**2
        self.n_epochs = n_epochs
        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        maximize=maximize, foreach=foreach, capturable=capturable)
        super(ADAM_TC, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

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
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:

                    # Update history
                    self.history.append(p.detach().clone())
                    
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                            if self.defaults['capturable'] else torch.tensor(0.)
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    state_steps.append(state['step'])

            self.adam(params_with_grad,
                      grads,
                      exp_avgs,
                      exp_avg_sqs,
                      max_exp_avg_sqs,
                      state_steps,
                      amsgrad=group['amsgrad'],
                      beta1=beta1,
                      beta2=beta2,
                      lr=group['lr'],
                      weight_decay=group['weight_decay'],
                      eps=group['eps'],
                      maximize=group['maximize'],
                      foreach=group['foreach'],
                      capturable=group['capturable'])

        return loss


    def adam(self,
            params: List[Tensor],
            grads: List[Tensor],
            exp_avgs: List[Tensor],
            exp_avg_sqs: List[Tensor],
            max_exp_avg_sqs: List[Tensor],
            state_steps: List[Tensor],
            # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
            # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
            foreach: bool = None,
            capturable: bool = False,
            *,
            amsgrad: bool,
            beta1: float,
            beta2: float,
            lr: float,
            weight_decay: float,
            eps: float,
            maximize: bool):

        if len(params) > 1:
            raise RuntimeError(f"Should only have 1 parameter set. Parameter set: {params}")

        if not all(isinstance(t, torch.Tensor) for t in state_steps):
            raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

        if foreach and torch.jit.is_scripting():
            raise RuntimeError('torch.jit.script not supported with foreach optimizers')

        if foreach and not torch.jit.is_scripting():
            func = self._multi_tensor_adam
        else:
            func = self._single_tensor_adam

        func(params,
             grads,
             exp_avgs,
             exp_avg_sqs,
             max_exp_avg_sqs,
             state_steps,
             amsgrad=amsgrad,
             beta1=beta1,
             beta2=beta2,
             lr=lr,
             weight_decay=weight_decay,
             eps=eps,
             maximize=maximize,
             capturable=capturable)


    def _single_tensor_adam(self,
                            params: List[Tensor],
                            grads: List[Tensor],
                            exp_avgs: List[Tensor],
                            exp_avg_sqs: List[Tensor],
                            max_exp_avg_sqs: List[Tensor],
                            state_steps: List[Tensor],
                            *,
                            amsgrad: bool,
                            beta1: float,
                            beta2: float,
                            lr: float,
                            weight_decay: float,
                            eps: float,
                            maximize: bool,
                            capturable: bool):

        for i, param in enumerate(params):

            grad = grads[i] if not maximize else -grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step_t = state_steps[i]

            if capturable:
                assert param.is_cuda and step_t.is_cuda, "If capturable=True, params and state_steps must be CUDA tensors."

            # update step
            step_t += 1

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            if torch.is_complex(param):
                grad = torch.view_as_real(grad)
                exp_avg = torch.view_as_real(exp_avg)
                exp_avg_sq = torch.view_as_real(exp_avg_sq)
                param = torch.view_as_real(param)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

            if capturable:
                step = step_t

                # 1 - beta1 ** step can't be captured in a CUDA graph, even if step is a CUDA tensor
                # (incurs "RuntimeError: CUDA error: operation not permitted when stream is capturing")
                bias_correction1 = 1 - torch.pow(beta1, step)
                bias_correction2 = 1 - torch.pow(beta2, step)

                step_size = lr / bias_correction1
                step_size_neg = step_size.neg()

                bias_correction2_sqrt = bias_correction2.sqrt()

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                    # Uses the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
                else:
                    denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)

                param.addcdiv_(exp_avg, denom)
            else:
                step = step_t.item()

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1
                step_size = step_size/(1 + self.Vbias/50)

                bias_correction2_sqrt = math.sqrt(bias_correction2)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                param.addcdiv_(exp_avg, denom, value=-step_size)
                
            # Adapt levy noise properties then add noise vector
            alpha = self.adapt_alpha(step)

            param.add_(self.levy_noise(param, alpha, exp_avg, step_size, denom, step), alpha=step_size)
            
            # Inbuilt periodic boundary conditions to ensure results are consistent.
            param = self.objfunc.pbc(param)
                

    def levy_noise(self, param, alpha, exp_avg, step_size, denom, step):
        dim = param.size(dim=0)
        direction = hypersphere_sample(dim)
        
        # try powlaw samp first
        #levy_r = powlaw_samp(x_min=0.05, alpha=alpha) * torch.norm(grad)

        # now levy alpha stable samp #####
        #levy_r = levy_stable.rvs(alpha, 0, scale=step_size, size=dim) * torch.norm(grad)
        # Different ways of scaling noise
            #  levy_r = levy_stable.rvs(alpha, 0, scale=np.sqrt(step_size/step), size=dim)
            #  levy_r = np.sqrt(sum(levy_r**2))
        # if alpha <= 2:
        #     levy_r = levy_stable.rvs(alpha, 0, scale=np.sqrt(step_size/((alpha-0.9999)*step)), size=dim)
        #     levy_r = np.sqrt(sum(levy_r**2))
        # else:
        #     levy_r = 0 # behaviour too abrupt
        alpha = 2 if alpha >= 2 else alpha
        #scale = torch.abs(torch.div(exp_avg, denom) * step_size)
            # Alt: torch.sqrt(...)
            # Hypothesis : this tends to get stuck in ravines
        scale = torch.sqrt(torch.norm(exp_avg)/torch.norm(denom) * step_size)
            # removes problem in above hypothesis
            # New Hypothesis(4:04am 14/08) : anneal the scale
            # used vbias to adjust lr --> maybe dont anneal the scale
            # try search then converge (STC) to anneal lr - Darken
                    # need to set c --> use our adaptive alpha?
            # THOUGHTS:
                # adaptive alpha kinda shit because response is not instant
                # difference in behaviour arises in many-sample limit --> long time limit
                # switch to some observational results e.g. study the heavy tailed noise empirically
                # want to understand what makes it so good --> anisotropy?
                # supporting literature for anisotropy exists
                # MD paper --> stepsize is Gaussian width
            # TO DOS:
                # Zhu2019 appendix D.1 --> test function with 2 wells
                # finish implementing SGD with levy noise
                # set up typical neural network experiment
                    # use SGD + ADAM
                    # store gradients and compute gradient noise characteristics
                    # add artificial noise (inspire from eqns) and see how it changes shit
                    # compute how parallel the noise is with the actual gradient
                            # dot product with determinstic gradient + stochastic gradient
                            # normalise by norms
                            # find average over time, over alphas, etc etc
        #scale = 0.05/alpha
        levy_r = levy_stable.rvs(alpha, 0, scale=scale, size=dim)
        levy_r = np.sqrt(sum(levy_r**2))

        #print('ADAM ratio:', levy_r/torch.norm(grad), end=' ')
        print(scale, levy_r, levy_r/scale, alpha)
        noise = levy_r * direction
        return noise

    def adapt_alpha(self, step):
        
        current = self.history[-1]
        past = self.history[0:-1]
        Vbias = 0
        
        t1 = time.perf_counter() # Time performance
        for p in past:
            v = current - p
            Vbias += torch.exp(self.width_denom * torch.dot(v, v.T))
        t2 = time.perf_counter()
        print(f"Time for iteration {step}: {t2 - t1}")
        
        Vbias = float(self.height * Vbias)
        self.Vbias = Vbias
        self.alpha = (2 - self.height) + Vbias/step
        self.alpha_record.append(self.alpha)

        return self.alpha

    def _multi_tensor_adam(self,
                           params: List[Tensor],
                           grads: List[Tensor],
                           exp_avgs: List[Tensor],
                           exp_avg_sqs: List[Tensor],
                           max_exp_avg_sqs: List[Tensor],
                           state_steps: List[Tensor],
                           *,
                           amsgrad: bool,
                           beta1: float,
                           beta2: float,
                           lr: float,
                           weight_decay: float,
                           eps: float,
                           maximize: bool,
                           capturable: bool):

        raise RuntimeError("Multi-tensor ADAM_TC not supported yet.")

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