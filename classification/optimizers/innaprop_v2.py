import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Union, Tuple


class INNAprop_v2(Optimizer):
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
        super(INNAprop_v2, self).__init__(params, defaults)

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
            exp_avgs = []
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
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                              
                state_steps.append(state['step'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                exp_avgs.append(state['exp_avg'])
                

            innaprop_v2(params_with_grad,
                  grads,
                  state_steps,
                  exp_avgs,
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

def innaprop_v2(params: List[Tensor],
          grads: List[Tensor],
          state_steps: List[Tensor],
          exp_avgs: List[Tensor],
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

    
    func = _single_tensor_innaprop_v2

    func(params,
         grads,
         state_steps,
         exp_avgs,
         exp_avg_sqs,
         alpha=alpha,
         beta=beta,
         sigma=sigma,
         eps=eps,
         lr=lr,
         weight_decay=weight_decay,
         maximize=maximize,
         capturable=capturable)

def _single_tensor_innaprop_v2(params: List[Tensor],
                         grads: List[Tensor],
                         state_steps: List[Tensor],
                         exp_avgs: List[Tensor],
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
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda 
            
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)
            exp_avg = torch.view_as_real(psi)
        
        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the second moment running average coefficient
        exp_avg_sq.mul_(sigma).addcmul_(grad, grad, value=1 - sigma)

        scalar1 = 1 - (alpha * lr)
        scalar2 =  lr ** 2 * ((1 - (alpha * beta)) / scalar1) #((lr ** 2) * (1 - (alpha * beta))) / scalar1
        scalar3 = (lr * (beta - lr)) / scalar1

        if capturable:
            step = step_t
            
            bias_correction = 1 - sigma ** step
            
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction)).add_(eps)
            #denom = exp_avg_sq.sqrt().add_(eps)

            exp_avg.mul_(scalar1).addcdiv_(grad, denom, value = scalar2)
            param.sub_(exp_avg).addcdiv_(grad , denom, value= -scalar3)
        else:

            step = step_t.item()
            
            bias_correction = 1 - sigma ** step
            
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction)).add_(eps)
            #denom = exp_avg_sq.sqrt().add_(eps)

            exp_avg.mul_(scalar1).addcdiv_(grad, denom, value = scalar2)
            param.sub_(exp_avg).addcdiv_(grad , denom, value= -scalar3)

