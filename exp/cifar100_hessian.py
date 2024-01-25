import sys
sys.path.append('.')

import torch
import torch.nn.functional as F
from torch import autograd
from torchvision import transforms,datasets
import numpy as np
import random
import argparse
import data_loader
import models
import utils
from tqdm import tqdm

# parse options
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir'    , type=str, default='~'         )
parser.add_argument('--batch-size'  , type=int, default=2500        )
parser.add_argument('--seed'        , type=int, default=None        )
parser.add_argument('--gpu'         , type=int, default=0           )
parser.add_argument('--model'       , type=str, default='sgd_net'   )
args = parser.parse_args()

# setup GPU
utils.GPU_setup(args.gpu,args.seed)

# load data
print('==> Loading Data...')
_train_transform = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)),
])
_train_set = datasets.CIFAR100(root=args.data_dir,train=True,download=True,transform=_train_transform)
trainloader = torch.utils.data.DataLoader(_train_set,batch_size=args.batch_size,shuffle=False,drop_last=True,num_workers=0)

# build model
print('==> Building Model...')
net = models.ResNet18(num_classes=100)
net.load_state_dict(torch.load('%s.pt' % args.model))
net = net.to(args.gpu)

opt = torch.optim.SGD(net.parameters(),lr=1)
criterion = torch.nn.CrossEntropyLoss(reduction='none').to(args.gpu)

def weight_decay(net,decay_rate):
    loss = 0.
    for param in net.parameters():
        loss += (param*param).sum()
    return decay_rate*loss

print('==> Computing...')
net.train()
eigen = []
for batch_idx,(inputs,targets) in enumerate(trainloader):
    print('Batch %d:' % batch_idx)
    inputs,targets = inputs.to(args.gpu),targets.to(args.gpu)
    outputs = net(inputs)
    losses = criterion(outputs,targets)+weight_decay(net,5e-4)

    # compute avg(grad)
    opt.zero_grad()
    losses.mean().backward(retain_graph=True)

    I,g_avg = [],[]
    for param in net.parameters():
        I.append(torch.zeros_like(param.grad))
        g_avg.append(param.grad.clone())

    # compute each grad
    for l in tqdm(losses):
        opt.zero_grad()
        l.backward(retain_graph=True)
        for I_i,param,g_avg_i in zip(I,net.parameters(),g_avg):
            g_i = param.grad
            sub_i = g_i-g_avg_i
            I_i += sub_i*sub_i

    eigenvalues = [(I_i/len(losses)).cpu().flatten() for I_i in I]
    eigen.append(torch.cat(eigenvalues))

eigen = sum(eigen)/len(eigen)
np.save('%s_eigen.npy' % args.model,eigen.numpy())
