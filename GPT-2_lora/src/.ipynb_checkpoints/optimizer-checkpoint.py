#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import math
import os
from collections import OrderedDict 
import argparse


from torch import Tensor
from typing import List, Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


def add_optimizer_params(parser: argparse.ArgumentParser):
    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay rate')
    parser.add_argument('--correct_bias', action='store_true', help='correct adam bias term')
    parser.add_argument('--epsilon', default=1e-6, type=float, help='adam epsilon')
    parser.add_argument('--no_decay_bias', action='store_true', help='no weight decay on bias weigh')
    parser.add_argument('--adam_beta1', default=0.9, type=float, help='adam beta1 term')
    parser.add_argument('--adam_beta2', default=0.98, type=float, help='adam beta2 term')
    parser.add_argument('--innaprop_alpha', default=0.1, type=float, help='innaprop alpha term')
    parser.add_argument('--innaprop_beta', default=0.9, type=float, help='innaprop beta term')

    
    
    parser.add_argument('--scheduler', default='linear', type=str,
                        choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant', 'linear', 'cycle'],
                        help='lr scheduler to use.')

    parser.add_argument('--max_step', type=int, default=None, help='upper epoch limit')

    parser.add_argument('--max_epoch', type=int, default=None, help='max epoch of training')

    parser.add_argument('--warmup_step', type=int, default=0, help='upper epoch limit')

    parser.add_argument('--i_steps', type=str, default='0', help='interval_steps')
    parser.add_argument('--i_lrs', type=str, default='0.00025', help='interval_lrs')
    


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


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(
        self,
        optimizer : torch.optim.Optimizer,
        max_lr : float = 0.1,
        min_lr : float = 0.0,
        warmup_steps : int = 0,
        max_steps : int = 1,
        alpha : float = 0.,
        last_epoch : int = -1
    ):
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        
        self.alpha = alpha # decrease rate of max learning rate by cycle
        self.max_steps = max_steps
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        self.init_lr()
    
    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            curr_lr = self.max_lr * self.last_epoch / self.warmup_steps
            return curr_lr
        else:
            _step = min(self.last_epoch, self.max_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * _step / self.max_steps))
            decayed = (1 - self.alpha) * cosine_decay + self.alpha
            return self.max_lr * decayed # learning_rate * decayed

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = math.floor(epoch)
        _lr = self.get_lr()
        for param_group in self.optimizer.param_groups: 
            param_group['lr'] = _lr


class CyclicScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        interval_steps = [],
        interval_lrs = [],
        last_epoch = -1,
    ):        
        self.optimizer = optimizer

        self.interval_steps = interval_steps
        self.interval_lrs = interval_lrs

        self.last_epoch = last_epoch

        super(CyclicScheduler, self).__init__(optimizer, last_epoch)
        
        self.init_lr()
    
    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.interval_lrs[0]
    
    def get_lr(self):
        for _i in range(0, len(self.interval_steps)-1):
            if self.last_epoch >= self.interval_steps[_i] and self.last_epoch < self.interval_steps[_i + 1]:
                _alpha = (self.last_epoch - self.interval_steps[_i]) / (self.interval_steps[_i + 1] - self.interval_steps[_i] + 1e-6)
                if _alpha < 0:
                    _alpha = 0
                if _alpha >= 1:
                    _alpha = 1
                curr_lr = _alpha * self.interval_lrs[_i + 1] + (1.0 - _alpha) * self.interval_lrs[_i]             
                return curr_lr
        return self.interval_lrs[-1]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        #self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        _lr = self.get_lr()
        for param_group in self.optimizer.param_groups: #, self.get_lr()):
            param_group['lr'] = _lr



def get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    last_epoch=-1
):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    last_epoch=-1
):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def create_grouped_parameters(model, no_decay_bias): # args):
    if not no_decay_bias:
        optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()], # if not any(nd in n for nd in no_decay)],
        }]
    else:
        no_decay = ["bias", "layer_norm.weight"]

        optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
            "weight_decay": 0.0,
        }]
    return optimizer_grouped_parameters


def create_adam_optimizer(
    model, 
    lr, 
    weight_decay, 
    optimizer_grouped_parameters=None, 
    beta1=0.9, 
    beta2=0.98, 
    epsilon=1e-6, 
    no_decay_bias=False
):
    if optimizer_grouped_parameters is None:
        optimizer_grouped_parameters = create_grouped_parameters(model, no_decay_bias)

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, 
        lr=lr, 
        betas=(beta1, beta2), 
        eps=epsilon, 
        weight_decay=weight_decay
    )
    return optimizer

def create_innaprop_optimizer(
    model, 
    lr, 
    weight_decay, 
    optimizer_grouped_parameters=None, 
    alpha=0.1, 
    beta=0.9, 
    epsilon=1e-6, 
    no_decay_bias=False
):
    if optimizer_grouped_parameters is None:
        optimizer_grouped_parameters = create_grouped_parameters(model, no_decay_bias)

    optimizer = INNAprop(
        optimizer_grouped_parameters, 
        lr=lr, 
        alpha=alpha,
        beta=beta, 
        eps=epsilon, 
        weight_decay=weight_decay)
    return optimizer



def create_sgd_optimizer(model, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    return optimizer
    

def create_adam_optimizer_from_args(model, args, grouped_parameters=None):
    if grouped_parameters is None:
        grouped_parameters = create_grouped_parameters(model, args.no_decay_bias)

    optimizer = torch.optim.AdamW(
        grouped_parameters, 
        lr=args.lr, 
        betas=(args.adam_beta1, args.adam_beta2), 
        eps=args.epsilon, 
        weight_decay=args.weight_decay
    )
    return optimizer

def create_innaprop_optimizer_from_args(model, args, grouped_parameters=None):
    if grouped_parameters is None:
        grouped_parameters = create_grouped_parameters(model, args.no_decay_bias)

    optimizer = INNAprop(
        grouped_parameters, 
        lr=args.lr, 
        alpha=args.innaprop_alpha,
        beta=args.innaprop_beta,
        eps=args.epsilon, 
        weight_decay=args.weight_decay    
    )
    return optimizer


def create_optimizer_scheduler(optimizer, args):
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, 
            max_lr=args.lr, 
            min_lr=0.0, 
            warmup_steps=args.warmup_step, 
            max_steps=args.max_step, alpha=0
        )
    elif args.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, args.warmup_step, args.max_step, last_epoch=-1
        )
    elif args.scheduler == 'cycle':
        if args.i_steps is not None:
            args.i_steps = [int(_i) for _i in args.i_steps.split(',')]
            args.i_lrs = [float(_i) for _i in args.i_lrs.split(',')]
        args.max_step = args.i_steps[-1]
        print('max_step is rest to', args.max_step)
        scheduler = CyclicScheduler(
            optimizer, interval_steps=args.i_steps, interval_lrs=args.i_lrs
        )
    elif args.scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer, args.warmup_step, args.max_step, last_epoch=-1
        )
    else:
        # constant leanring rate.
        scheduler = None
    return scheduler
