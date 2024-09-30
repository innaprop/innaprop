import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Union, Tuple


class INNAprop(Optimizer):
    def __init__(self, params, lr=0.001, alpha=0.1, beta = 0.9, sigma = 0.999, eps = 1e-8,
         weight_decay=0.01, *, maximize: bool = False,
         capturable: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= sigma < 1.0:
            raise ValueError(f"Invalid sigma parameter at index 0: {sigma}")
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, alpha=alpha, beta=beta, sigma=sigma, eps=eps,
                        weight_decay=weight_decay, 
                        maximize=maximize, capturable=capturable)
        super(INNAprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))
    

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            state_steps = []
            state_psis = []
            exp_avg_sqs = []
            alpha, beta = group['alpha'], group['beta']
            sigma = group['sigma']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                
                if p.grad.is_sparse:
                    raise RuntimeError('Hero does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['psi'] = (1 - alpha*beta)*torch.clone(p,memory_format=torch.preserve_format)
                              
                state_steps.append(state['step'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                state_psis.append(state['psi'])
                

            innaprop(params_with_grad,
                  grads,
                  state_steps,
                  state_psis,
                  exp_avg_sqs,
                  alpha=alpha,
                  beta=beta,
                  sigma=sigma,
                  eps=group['eps'],
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  maximize=group['maximize'],
                  capturable=group['capturable'])

        return loss

def innaprop(params: List[Tensor],
          grads: List[Tensor],
          state_steps: List[Tensor],
          state_psis: List[Tensor],
          exp_avg_sqs: List[Tensor],
          capturable: bool = False,
          *,
          alpha: float,
          beta: float,
          sigma: float,
          eps: float,
          lr: Union[float, Tensor],
          weight_decay: float,
          maximize: bool):

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    
    func = _single_tensor_innaprop

    func(params,
         grads,
         state_steps,
         state_psis,
         exp_avg_sqs,
         alpha=alpha,
         beta=beta,
         sigma=sigma,
         eps=eps,
         lr=lr,
         weight_decay=weight_decay,
         maximize=maximize,
         capturable=capturable)

def _single_tensor_innaprop(params: List[Tensor],
                         grads: List[Tensor],
                         state_steps: List[Tensor],
                         state_psis: List[Tensor],
                         exp_avg_sqs: List[Tensor],
                         *,
                         alpha: float,
                         beta: float,
                         sigma: float, 
                         eps : float,
                         lr: Union[float, Tensor],
                         weight_decay: float,
                         maximize: bool,
                         capturable: bool):
    
    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        step_t = state_steps[i]
        psi = state_psis[i]
        exp_avg_sq = exp_avg_sqs[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda 
            
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)
            psi = torch.view_as_real(psi)
        
        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg_sq.mul_(sigma).addcmul_(grad, grad, value=1 - sigma)

        inv_beta = 1.0 / beta

        if capturable:
            step = step_t
            scalar1 = (lr * (1 - beta * alpha)) / (beta - lr)
            scalar2 = lr / (beta - lr)
            
            bias_correction = 1 - sigma ** step
            
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction)).add_(eps)
            #denom = exp_avg_sq.sqrt().add_(eps)

            psi.mul_(1 - (lr / beta)).add_(param, alpha=lr * (inv_beta - alpha))
            param.mul_(1 + scalar1).sub_(psi, alpha=scalar2).addcdiv_(grad , denom, value= -lr * beta)
        else:

            step = step_t.item()
            scalar1 = (lr * (1 - beta * alpha)) / (beta - lr)
            scalar2 = lr / (beta - lr)
            
            bias_correction = 1 - sigma ** step

            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction)).add_(eps)
            #denom = exp_avg_sq.sqrt().add_(eps)
            

            psi.mul_(1 - (lr / beta)).add_(param, alpha=lr * (inv_beta - alpha))
            param.mul_(1 + scalar1).sub_(psi, alpha=scalar2).addcdiv_(grad , denom, value= -lr * beta)

