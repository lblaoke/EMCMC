import os
import sys
sys.path.append('.')

import torch
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np

import argparse
from time import time
from datetime import datetime
from tqdm import tqdm
from math import *

import utils
import data_loader
import models
import test_eval

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
    model.apply(_enable)

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

# parse options
parser = argparse.ArgumentParser()

parser.add_argument('--data-dir'        , type=str  , default='~'           )
parser.add_argument('--save'            , type=str  , default='cifar10_sam' )

parser.add_argument('--num-class'       , type=int  , default=10            )
parser.add_argument('--gpu'             , type=int  , default=0             )
parser.add_argument('--seed'            , type=int  , default=None          )

parser.add_argument('--epoch'           , type=int  , default=200           )
parser.add_argument('--batch-size'      , type=int  , default=128           )
parser.add_argument('--lr0'             , type=float, default=0.5           )
parser.add_argument('--decay-scheme'    , type=str  , default='cyclical'    )
parser.add_argument('--lr-end'          , type=float, default=0             )

args = parser.parse_args()

# setup GPU
utils.GPU_setup(args.gpu,args.seed)

# load data
print('==> Loading Data...')
trainloader,testloader = data_loader.cifar(args,num_classes=args.num_class)
oodloader = data_loader.svhn(args)

# build model
print('==> Building Model...')
net = models.ResNet18(num_classes=args.num_class).to(args.gpu)

# training at each epoch
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct,total = 0,0
    for batch_idx,(inputs,targets) in tqdm(enumerate(trainloader)):
        inputs,targets = inputs.to(args.gpu),targets.to(args.gpu)

        lr = utils.lr_decay(args,opt,epoch,batch_idx,num_batch,T,M)

        enable_running_stats(net)
        outputs = net(inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        opt.first_step(zero_grad=True)

        disable_running_stats(net)
        criterion(net(inputs), targets).backward()
        opt.second_step(zero_grad=True)

        outputs,loss = outputs.detach(),loss.detach()
        train_loss += loss.data.item()
        _,predicted = torch.max(outputs.data,1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).sum().item()

    print('Loss: %.3f | ACC: %.3f%% (%d/%d)' % (train_loss/num_batch,100.*correct/total,correct,total))

datasize = 50000
num_batch = datasize/args.batch_size+1
M = 4 # number of cycles
T = args.epoch*num_batch # total number of iterations
criterion = torch.nn.CrossEntropyLoss()

base_opt = torch.optim.SGD
opt = SAM(net.parameters(),base_opt,lr=args.lr0,weight_decay=5e-4)

# training loop
print('==> Training...')
_time = datetime.now()
path = f'.checkpoints/{args.save}_{_time.year}_{_time.month}_{_time.day}'
os.system(f'mkdir -p {path}')
w_list = []

_time = time()

for epoch in range(args.epoch):
    train(epoch)

    if (epoch%50)+1>46:
        acc = test_eval.test(args.gpu, net, testloader, oodloader)
        w_list.append(utils.save_sample(net, f'{path}/{epoch}.pt'))

    else:
        # acc = test_eval.test(args.gpu,net,testloader)
        acc = 0

# report time usage
minute = (time() - _time) / 60
if minute<=60:
    print(f'Training finished in {minute:.1f} min.')
else:
    print(f'Training finished in {minute/60:.1f} h.')
