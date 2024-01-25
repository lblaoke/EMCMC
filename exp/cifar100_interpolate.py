import sys
sys.path.append('.')

import torch
import torch.nn.functional as F
from torchvision import transforms,datasets
import numpy as np
import random
from time import time
from tqdm import tqdm
from math import *
import argparse
import data_loader
import models
import utils
import metric

# parse options
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir'        , type=str  , default='~'           )
parser.add_argument('--batch-size'      , type=int  , default=1000          )
parser.add_argument('--gpu'             , type=int  , default=0             )
parser.add_argument('--seed'            , type=int  , default=1             )
parser.add_argument('--model1'          , type=str  , default='emcmc1_net' )
parser.add_argument('--model2'          , type=str  , default=None         )
args = parser.parse_args()

# setup GPU
utils.GPU_setup(args.gpu,args.seed)

# load data
print('==> Loading Data...')
trainloader,testloader = data_loader.cifar(args,num_classes=100)
val_targets = torch.tensor(testloader.dataset.targets,dtype=torch.int64)

# build model
print('==> Building Model...')
net = models.ResNet18(num_classes=100)
net_random = models.ResNet18(num_classes=100)
net_mix = models.ResNet18(num_classes=100)

net.load_state_dict(torch.load('%s.pt' % args.model1))
if args.model2:
    net_random.load_state_dict(torch.load('%s.pt' % args.model2))

def weight_decay(_net,decay_rate):
    loss = 0.
    for param in _net.parameters():
        loss += (param*param).sum()
    return decay_rate*loss

def test(_net):

    # train set
    _net.train()
    loss = 0
    for _,(inputs,targets) in enumerate(trainloader):
        o = _net(inputs.to(args.gpu))
        loss += criterion(o,targets.to(args.gpu)).item()

    loss = loss/50000+weight_decay(_net,5e-4).item()

    # test set
    _net.eval()
    output = []
    for _,(inputs,_) in enumerate(testloader):
        with torch.no_grad():
            o = _net(inputs.to(args.gpu))
        output.append(o.detach().cpu())

    acc = metric.score(torch.cat(output,dim=0),None,val_targets,verbose=False)

    return loss,acc

criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(args.gpu)
distance = 0
for param1,param2 in zip(net.parameters(),net_random.parameters()):
    sub = param1.data-param2.data
    distance += torch.sum(sub*sub).item()
    # distance += torch.sum(param2.data*param2.data).item()
distance = sqrt(distance)
print(distance)

dis,loss,acc = [],[],[]
for i in range(0,30):
    ratio = i*0.1/distance
    utils.resample(net_mix,net)
    w_net_mix = net_mix.state_dict()
    w_net_random = net_random.state_dict()
    for (n1,w1),(n2,w2) in zip(w_net_mix.items(),w_net_random.items()):
        if w1.type()=='torch.LongTensor':
            continue
        w1.mul_(1-ratio)
        w1.add_(ratio*w2)
    net_mix.load_state_dict(w_net_mix)

    net_mix = net_mix.to(args.gpu)
    l,a = test(net_mix)
    net_mix = net_mix.cpu()

    print(i*0.1,':',l,a)
    dis.append(i*0.1)
    loss.append(l)
    acc.append(a)

print('dis =',dis)
print('loss =',loss)
print('acc =',acc)
