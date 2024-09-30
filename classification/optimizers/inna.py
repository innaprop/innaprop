import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional
 
class INNA(Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.1, beta = 0.1,
         weight_decay=0.0, *, maximize: bool = False,
         capturable: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, alpha=alpha, beta=beta,
                        weight_decay=weight_decay,
                        maximize=maximize, capturable=capturable)
        super(INNA, self).__init__(params, defaults)
 
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
            alpha, beta = group['alpha'], group['beta']
 
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
                    state['step'] = torch.zeros((), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['psi'] = (1 - alpha*beta)*torch.clone(p,memory_format=torch.preserve_format).detach()
                              
                state_psis.append(state['psi'])
                state_steps.append(state['step'])
               
 
            inna(params_with_grad,
                  grads,
                  state_steps,
                  state_psis,
                  alpha=alpha,
                  beta=beta,
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  maximize=group['maximize'],
                  capturable=group['capturable'])
 
        return loss
 
def inna(params: List[Tensor],
          grads: List[Tensor],
          state_steps: List[Tensor],
          state_psis: List[Tensor],
          capturable: bool = False,
          *,
          alpha: float,
          beta: float,
          lr: float,
          weight_decay: float,
          maximize: bool):
 
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")
 
   
    func = _single_tensor_inna
 
    func(params,
         grads,
         state_steps,
         state_psis,
         alpha=alpha,
         beta=beta,
         lr=lr,
         weight_decay=weight_decay,
         maximize=maximize,
         capturable=capturable)

    
def _single_tensor_inna(params: List[Tensor],
                         grads: List[Tensor],
                         state_steps: List[Tensor],
                         state_psis: List[Tensor],
                         *,
                         alpha: float,
                         beta: float,
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
 
        if capturable:
            assert param.is_cuda and step_t.is_cuda
 
        # Perform stepweight decay
      
        param.mul_(1 - lr * weight_decay)
           
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            param = torch.view_as_real(param)
 
        step_t += 1
        inv_beta = 1 / beta
 
        if capturable:
            step = step_t
            step_size = lr
 
            phase_update = (alpha - inv_beta) * param + inv_beta * psi
            geom_update = beta * grad
 
            psi.add_(phase_update, alpha=-step_size)
            param.add_(phase_update + geom_update, alpha=-step_size)
        else:
            step = step_t.item()
            step_size = lr
 
            phase_update = (alpha - inv_beta) * param + inv_beta * psi
            geom_update = beta * grad
        
 
            psi.add_(phase_update, alpha=-step_size)
            param.add_(phase_update + geom_update, alpha=-step_size)
 
