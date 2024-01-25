import sys
sys.path.append('.')

import torch
import torch.nn.functional as F
from torchvision import transforms,datasets
import numpy as np
import random
from time import time
from tqdm import tqdm
import argparse
import data_loader
import models
import utils

# parse options
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir'        , type=str  , default='~'       )
parser.add_argument('--epoch'           , type=int  , default=200       )
parser.add_argument('--batch-size'      , type=int  , default=128       )
parser.add_argument('--gpu'             , type=int  , default=0         )
parser.add_argument('--seed'            , type=int  , default=None      )
parser.add_argument('--lr'              , type=float, default=0.5       )
parser.add_argument('--decay'           , type=str  , default='cyclical')
parser.add_argument('--decay-rate'      , type=float, default=0         )
parser.add_argument('--check-point'     , type=str  , default=None      )
args = parser.parse_args()

# setup GPU
utils.GPU_setup(args.gpu,args.seed)

# load data
print('==> Loading Data...')
trainloader,testloader = data_loader.cifar(args,num_classes=10)
oodloader = data_loader.svhn(args)
val_targets = torch.tensor(testloader.dataset.targets,dtype=torch.int64)
val_targets_ood = torch.tensor(oodloader.dataset.labels,dtype=torch.int64)

# build model
print('==> Building Model...')
net0 = models.ResNet18().to(args.gpu)
net1 = models.ResNet18().to(args.gpu)
net2 = models.ResNet18().to(args.gpu)
nets = [net0,net1,net2]

def train(epoch):
    print('\nEpoch: %d' % epoch)
    for net in nets:
        net.train()
    train_loss = 0
    correct,total = 0,0
    for batch_idx,(inputs,targets) in tqdm(enumerate(trainloader)):
        inputs,targets = inputs.to(args.gpu),targets.to(args.gpu)

        opt.zero_grad()
        lr = utils.lr_decay(args,opt,epoch,batch_idx,num_batch,T,M)

        loss,outputs = [],[]
        for net in nets:
            o = net(inputs)
            loss.append(criterion(o,targets))
            outputs.append(o)
        loss = sum(loss)
        outputs = sum(outputs)/len(outputs)

        loss.backward()
        opt.step()

        outputs,loss = outputs.detach(),loss.detach()
        train_loss += loss.data.item()
        _,predicted = torch.max(outputs.data,1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).sum().item()

    print('Loss: %.3f | ACC: %.3f%% (%d/%d)' % (train_loss/num_batch,100.*correct/total,correct,total))

def test(nets,OOD:bool=False):
    for net in nets:
        net.eval()
    output,uncertainty = [],[]
    output_ood,uncertainty_ood = [],[]

    # test set
    for _,(inputs,_) in enumerate(testloader):
        inputs = inputs.to(args.gpu)

        with torch.no_grad():
            o = []
            for net in nets:
                o.append(net(inputs))
            o = sum(o)/len(o)
            u = -torch.sum(F.softmax(o,dim=1)*F.log_softmax(o,dim=1),dim=1)

        output.append(o.detach().cpu())
        uncertainty.append(u.detach().cpu())

    output = torch.cat(output,dim=0)
    uncertainty = torch.cat(uncertainty)

    pred = torch.argmax(output,dim=1)
    correct = (pred==val_targets)

    acc = metric.evaluate(correct,val_targets,uncertainty)

    if not OOD:
        return acc,output

    # OOD set
    for _,(inputs,_) in enumerate(oodloader):
        inputs = inputs.to(args.gpu)

        with torch.no_grad():
            o = []
            for net in nets:
                o.append(net(inputs))
            o = sum(o)/len(o)
            u = -torch.sum(F.softmax(o,dim=1)*F.log_softmax(o,dim=1),dim=1)

        output_ood.append(o.detach().cpu())
        uncertainty_ood.append(u.detach().cpu())

    output_ood = torch.cat(output_ood,dim=0)
    uncertainty_ood = torch.cat(uncertainty_ood)

    correct = torch.ones_like(val_targets)
    correct_ood = torch.zeros_like(val_targets_ood)

    metric.evaluate_ood(correct,correct_ood,uncertainty,uncertainty_ood)

    return acc,output,output_ood

datasize = 50000
num_batch = datasize/args.batch_size+1
M = 4 # number of cycles
T = args.epoch*num_batch # total number of iterations
criterion = torch.nn.CrossEntropyLoss()
opt = torch.optim.SGD(list(net0.parameters())+list(net1.parameters())+list(net2.parameters()),lr=args.lr,weight_decay=5e-4)
acc_list = []

start = time()

print('==> Training...')
for epoch in range(args.epoch):
    train(epoch)

    acc,_ = test(nets)

    if args.check_point is not None:
        acc_list.append(acc)

end = time()

# report time usage
minute = (end-start)/60
if minute<=60:
    print(f'Finished in {minute:.1f} min')
else:
    print(f'Finished in {minute/60:.1f} h')

# save model and acc
if args.check_point is not None:
    torch.save(net.state_dict(),'%s_net.pt' % args.check_point)
    np.save('%s_acc.npy' % args.check_point,np.array(acc_list))
