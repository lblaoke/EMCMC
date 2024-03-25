import os
import sys
sys.path.append('.')

import torch
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

# parse options
parser = argparse.ArgumentParser()

parser.add_argument('--data-dir'        , type=str  , default='~'                   )
parser.add_argument('--save'            , type=str  , default='cifar10_entropy_sgld')

parser.add_argument('--num-class'       , type=int  , default=10                    )
parser.add_argument('--gpu'             , type=int  , default=0                     )
parser.add_argument('--seed'            , type=int  , default=None                  )

parser.add_argument('--epoch'           , type=int  , default=200                   )
parser.add_argument('--batch-size'      , type=int  , default=128                   )
parser.add_argument('--lr0'             , type=float, default=0.5                   )
parser.add_argument('--decay-scheme'    , type=str  , default='cyclical'            )
parser.add_argument('--lr-end'          , type=float, default=0                     )
parser.add_argument('--temperature'     , type=float, default=1e-4                  )
parser.add_argument('--eta'             , type=float, default=4e-4                  )
parser.add_argument('--anchor'          , type=int  , default=20                    )

args = parser.parse_args()

# setup GPU
utils.GPU_setup(args.gpu,args.seed)

# load data
print('==> Loading Data...')
trainloader,testloader = data_loader.cifar(args,num_classes=args.num_class)
oodloader = data_loader.svhn(args)

# build model
print('==> Building Model...')
net = models.ResNet18(num_classes=args.num_class)
net_sampler = models.ResNet18(num_classes=args.num_class)
net_mean = models.ResNet18(num_classes=args.num_class)

net_sampler = utils.resample(net_sampler, net)
net_mean = utils.resample(net_mean, net)

net.to(args.gpu)
net_sampler.to(args.gpu)
net_mean.to(args.gpu)

def noise(net,coeff):
    _noise = 0
    for param in net.parameters():
        _noise += torch.sum(param*torch.randn_like(param.data)*coeff)
    return _noise

def reg(net1,net2,coeff):
    _reg = 0
    for param1,param2 in zip(net1.parameters(),net2.parameters()):
        sub = param1-param2.detach()
        _reg += torch.sum(sub*sub*coeff)
    return _reg

# training at each epoch
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net_sampler.train()
    net_mean.train()
    net.train()
    train_loss = 0
    correct,total = 0,0
    for batch_idx,(inputs,targets) in tqdm(enumerate(trainloader)):
        inputs,targets = inputs.to(args.gpu),targets.to(args.gpu)

        if batch_idx%args.anchor==0:
            utils.resample(net_mean,net_sampler)

        lr = utils.lr_decay(args,opt_sampler,epoch,batch_idx,num_batch,T,M)
        reg_coeff = 0.5/(args.eta*datasize)
        noise_coeff = sqrt(2/lr/datasize*args.temperature)

        opt_sampler.zero_grad()
        outputs = net_sampler(inputs)
        loss = criterion(outputs,targets)+reg(net_sampler,net,reg_coeff)+noise(net_sampler,noise_coeff)
        loss.backward()
        opt_sampler.step()

        for param1,param2 in zip(net_mean.parameters(),net_sampler.parameters()):
            param1.data *= 1-5/args.anchor
            param1.data += 5/args.anchor*param2.data

        if batch_idx%args.anchor==args.anchor-1:
            utils.resample(net,net_mean,eta=lr*args.eta)

        outputs,loss = outputs.detach(),loss.detach()
        train_loss += loss.data.item()
        _,predicted = torch.max(outputs.data,1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).sum().item()

    print('Loss: %.3f | ACC: %.3f%% (%d/%d)' % (train_loss/num_batch,100.*correct/total,correct,total))

# additional forward to re-compute BatchNorm
def additional_forward(net):
    net.train()
    with torch.no_grad():
        for _,(inputs,_) in enumerate(trainloader):
            net(inputs.to(args.gpu))

datasize = 50000
num_batch = datasize/args.batch_size+1
M = 4 # number of cycles
T = args.epoch*num_batch # total number of iterations
criterion = torch.nn.CrossEntropyLoss()
opt_sampler = torch.optim.SGD(net_sampler.parameters(),lr=args.lr0,weight_decay=5e-4)

# training loop
print('==> Training...')
_time = datetime.now()
path = f'.checkpoints/{args.save}_{_time.year}_{_time.month}_{_time.day}'
os.system(f'mkdir -p {path}')
w_list = []

_time = time()

for epoch in range(args.epoch):
    train(epoch)
    additional_forward(net)

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

# final testing
print('\n==> Final Testing...')
test_eval.multi_test(args.gpu, net, w_list, testloader, oodloader)
