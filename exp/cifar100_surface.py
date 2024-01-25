import sys
sys.path.append('.')

import torch
import numpy as np
from time import time
from tqdm import tqdm
import argparse
import data_loader
import models
import utils
import test_eval
import cmocean
import matplotlib.pyplot as plt
from math import *

def weight_decay(net,decay_rate):
    loss = 0.
    for param in net.parameters():
        loss += (param*param).sum()
    return decay_rate*loss

def _evaluate(args, model, loader):
    model = model.to(args.gpu)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    loss = 0.
    with torch.no_grad():
        correct, total = 0, 0
        for (x, y) in loader:
            x, y = x.cuda(), y.cuda()
            logits = model(x)
            preds = torch.argmax(logits, axis=1)
            correct += (preds == y).sum().item()
            loss += criterion(logits, y).item()
            total += len(x)
    model = model.cpu()
    return correct/total,loss/total+weight_decay(model,5e-4).item()

def loss_surface(args, weights_2, weights_1, theta, N, model, loader):
    u = weights_2 - weights_1
    v = theta - weights_1
    v -= u*(u@v)/(u@u) # orthogonalize

    us = torch.linspace(-.6, 9., N)
    vs = torch.linspace(-3., 3., N)
    grid_us, grid_vs = torch.meshgrid(us, vs, indexing='ij')

    accs = torch.zeros_like(grid_us)
    losses = torch.zeros_like(grid_us)

    with torch.no_grad():
        for i in tqdm(range(N)):
            for j in range(N):
                x, y = grid_us[i][j], grid_vs[i][j]
                w = x * u + y * v + weights_1
                torch.nn.utils.vector_to_parameters(w, model.parameters())
                acc, loss = _evaluate(args, model, loader)
                accs[i, j] = acc
                losses[i, j] = loss

    return grid_us, grid_vs, accs, losses

# parse options
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir'        , type=str  , default='~'               )
parser.add_argument('--batch-size'      , type=int  , default=128               )
parser.add_argument('--gpu'             , type=int  , default=0                 )
parser.add_argument('--seed'            , type=int  , default=None              )
parser.add_argument('--model'           , type=str  , default='100emcmc_surface')
args = parser.parse_args()

# setup GPU
utils.GPU_setup(args.gpu,args.seed)

# load data
print('==> Loading Data...')
trainloader,testloader = data_loader.cifar(args,num_classes=100)

# build model
print('==> Building Model...')
net_s = models.ResNet18(num_classes=100)
net_a = models.ResNet18(num_classes=100)

w_net_s,w_net_a = [],[]

for epoch in range(180,200,2):
    w_net_s.append(torch.load(args.model+'_s%d.pt' % epoch))
    w_net_a.append(torch.load(args.model+'_a%d.pt' % epoch))
print(len(w_net_s),len(w_net_a))

net_a.load_state_dict(w_net_a[0])
w1 = torch.nn.utils.parameters_to_vector(net_a.parameters())
net_a.load_state_dict(w_net_a[5])
w2 = torch.nn.utils.parameters_to_vector(net_a.parameters())
net_a.load_state_dict(w_net_a[9])
w3 = torch.nn.utils.parameters_to_vector(net_a.parameters())

grid_us,grid_vs,accs,losses = loss_surface(args,w1,w2,w3,N=10,model=net_a,loader=trainloader)

levels = [0.01, 0.25, 0.5, 0.75, 0.85, 0.90, 0.95, 0.98, 0.99,  1.,]

xs = np.linspace(1, 250, len(levels))
cmap_colors = [cmocean.cm.solar(int(x)) for x in xs]

plt.contour(grid_us, grid_vs, losses, zorder=1)#, levels=levels, colors=[cmap_colors[i] for i in range(len(levels))])
plt.contourf(grid_us, grid_vs, losses,  zorder=0)#, alpha=0.8, levels=levels, colors=[cmap_colors[i] for i in range(len(levels))])
plt.colorbar()

u = w2-w1
v = w3-w1
v -= u*(u@v)/(u@u) # orthogonalize

# plt.scatter([0,1,((w3-w1)@u).item()/sqrt((u@u).item())],[0,0,((w3-w1)@v).item()/sqrt((v@v).item())])
# /sqrt((u@u).item())

x,y = [],[]
for w_ in w_net_s:
    net_s.load_state_dict(w_)
    wi = torch.nn.utils.parameters_to_vector(net_s.parameters())
    x.append(((wi-w1)@u).item()/sqrt((u@u).item()))
    y.append(((wi-w1)@v).item()/sqrt((v@v).item()))
plt.scatter(x,y,color='black',label=r'$\theta$')

x,y = [],[]
for w_ in w_net_a:
    net_a.load_state_dict(w_)
    wi = torch.nn.utils.parameters_to_vector(net_a.parameters())
    x.append(((wi-w1)@u).item()/sqrt((u@u).item()))
    y.append(((wi-w1)@v).item()/sqrt((v@v).item()))
plt.scatter(x,y,color='red',label=r'$\theta_a$')

plt.savefig('../a.png')
