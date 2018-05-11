#June 9, Zheng Xu, xuzhustc@gmail.com

#make loss
#reference: GAN https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py


# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

import torchvision as thv
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as func

import torch.optim as optim
import torch.backends.cudnn as cudnn


import utils
import load_data as ld

import matplotlib.pyplot as plt
import numpy as np

import argparse
import os
import time
from datetime import datetime
import shutil
import math


################################################### ADAM with prediction method in ICLR 2018 #############################
class PredAdam(optim.Optimizer):
    """Implements Adam algorithm with prediction

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, pred_gm=1):
	#\bar u^{k+1} = u^{k+1} + pred_gm *(u^{k+1} - u^k) = u^k - (1+pred_gam)*stepsize*grad
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(PredAdam, self).__init__(params, defaults)
        self.pred_gm=pred_gm

    def __setstate__(self, state):
        super(PredAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def pred_step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    #state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg'] = p.data.new(p.data.size()).zero_()
                    # Exponential moving average of squared gradient values
                    #state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = p.data.new(p.data.size()).zero_()
                    state['actual_step'] = p.data.new(p.data.size()).zero_()
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        #state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                        state['max_exp_avg_sq'] = p.data.new(p.data.size()).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                state['actual_step'].copy_(p.data)
                state['actual_step'].addcdiv_(-step_size, exp_avg, denom)  #actual step
                p.data.addcdiv_(-step_size*(1+self.pred_gm), exp_avg, denom)        #predicitive step

        return loss


    def pred_restore(self):
        """restore the actual step from predictiv step for next iteration
        """

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                p.data.copy_(state['actual_step'])

############################################################################################



def get_optimizer(net, args, flag, lr):
    if flag == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(args.adam_b1, args.adam_b2), weight_decay=args.weight_decay)
    elif flag == 'padam':
        optimizer = PredAdam(net.parameters(), lr=lr, betas=(args.adam_b1, args.adam_b2), weight_decay=args.weight_decay, pred_gm=args.pred_gm)
    elif flag == 'rmsp':
        optimizer = optim.RMSprop(net.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif flag == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        print 'unknown optimizer name, use SGD!'
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return optimizer


def get_optimizer_var(var, args, flag, lr): #for variables
    if flag == 'adam':
        optimizer = optim.Adam(var, lr=lr, betas=(args.adam_b1, args.adam_b2), weight_decay=args.weight_decay)
    elif flag == 'padam':
        optimizer = PredAdam(var, lr=lr, betas=(args.adam_b1, args.adam_b2), weight_decay=args.weight_decay, pred_gm=args.pred_gm)
    elif flag == 'rmsp':
        optimizer = optim.RMSprop(var, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif flag == 'sgd':
        optimizer = optim.SGD(var, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        print 'unknown optimizer name, use SGD!'
        optimizer = optim.SGD(var, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return optimizer


def adjust_learning_rate(optimizer, init_lr, args, epoch, flag='linear'):
    """Sets the learning rate to the initial LR decayed by 10 every x epochs"""
    if flag == 'segment':
        if epoch % args.lr_freq == 0:
            lr = init_lr * (0.1 ** (epoch // args.lr_freq))
            print 'epoch %d learning rate schedule to %f'%(epoch, lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    elif flag == 'linear':
        s = epoch//args.lr_freq  #starting from lr_freq, learning rate drop linearly to 0.1 for each lr_freq
        p =  (s+1)*args.lr_freq - epoch
        elr = init_lr * (0.1**s)
        lr = elr + ( min(init_lr, elr*10) - elr)*float(p)/float(args.lr_freq)
        print 'epoch %d learning rate schedule to %f'%(epoch, lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        print 'make_opt: unknown flag for adjusting learning rates'


