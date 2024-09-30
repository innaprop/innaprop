import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Union, Tuple

class INNAdam(Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.1, beta = 0.1, sigmas: Tuple[float, float] = (0.9, 0.999), eps = 1e-8,
         weight_decay=1e-1, *, maximize: bool = False,
         capturable: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= sigmas[0] < 1.0:
            raise ValueError(f"Invalid sigma parameter at index 0: {sigmas[0]}")
        if not 0.0 <= sigmas[1] < 1.0:
            raise ValueError(f"Invalid sigma parameter at index 1: {sigmas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, alpha=alpha, beta=beta, sigmas=sigmas, eps=eps,
                        weight_decay=weight_decay, 
                        maximize=maximize, capturable=capturable)
        super(INNAdam, self).__init__(params, defaults)

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
            exp_avgs = []
            exp_avg_sqs = []
            alpha, beta = group['alpha'], group['beta']
            sigma1, sigma2 = group['sigmas']

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
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['psi'] = (1 - alpha*beta)*torch.clone(p,memory_format=torch.preserve_format).detach()
                              
                state_steps.append(state['step'])
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                state_psis.append(state['psi'])
                

            innadam(params_with_grad,
                  grads,
                  state_steps,
                  state_psis,
                  exp_avgs,
                  exp_avg_sqs,
                  alpha=alpha,
                  beta=beta,
                  sigma1=sigma1,
                  sigma2=sigma2,
                  eps=group['eps'],
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  maximize=group['maximize'],
                  capturable=group['capturable'])

        return loss

def innadam(params: List[Tensor],
          grads: List[Tensor],
          state_steps: List[Tensor],
          state_psis: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          capturable: bool = False,
          *,
          alpha: float,
          beta: float,
          sigma1: float,
          sigma2: float,
          eps: float,
          lr: float,
          weight_decay: float,
          maximize: bool):

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    
    func = _single_tensor_innadam

    func(params,
         grads,
         state_steps,
         state_psis,
         exp_avgs,
         exp_avg_sqs,
         alpha=alpha,
         beta=beta,
         sigma1=sigma1,
         sigma2=sigma2,
         eps=eps,
         lr=lr,
         weight_decay=weight_decay,
         maximize=maximize,
         capturable=capturable)

def _single_tensor_innadam(params: List[Tensor],
                         grads: List[Tensor],
                         state_steps: List[Tensor],
                         state_psis: List[Tensor],
                         exp_avgs: List[Tensor],
                         exp_avg_sqs: List[Tensor],
                         *,
                         alpha: float,
                         beta: float,
                         sigma1: float, 
                         sigma2: float,
                         eps : float,
                         lr: float,
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
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda 
            
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)
            psi = torch.view_as_real(psi)
        
        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - sigma1)
        exp_avg_sq.mul_(sigma2).addcmul_(grad, grad, value=1 - sigma2)

        inv_beta = 1.0 / beta

        if capturable:
            step = step_t

            bias_correction1 = 1 - sigma1 ** step
            bias_correction2 = 1 - sigma2 ** step

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)

            phase_update = (alpha - inv_beta) * param + inv_beta * psi
            geom_update = (beta * exp_avg) / denom 

            psi.add_(phase_update, alpha=step_size_neg)
            param.add_(phase_update + geom_update, alpha=step_size_neg)
        else:

            step = step_t.item()

            bias_correction1 = 1 - sigma1 ** step
            bias_correction2 = 1 - sigma2 ** step

            step_size = lr / bias_correction1
            step_size_neg = -step_size
            

            bias_correction2_sqrt = torch.sqrt(torch.tensor(bias_correction2))

            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            phase_update = (alpha - inv_beta) * param + (inv_beta) * psi
            geom_update = (beta * exp_avg) / denom 

            psi.add_(phase_update, alpha=step_size_neg)
            param.add_(phase_update + geom_update, alpha=step_size_neg)
            

