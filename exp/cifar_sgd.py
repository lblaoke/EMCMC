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

# parse options
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir'        , type=str  , default='~'       )
parser.add_argument('--num-class'       , type=int  , default=10        )
parser.add_argument('--epoch'           , type=int  , default=200       )
parser.add_argument('--batch-size'      , type=int  , default=128       )
parser.add_argument('--gpu'             , type=int  , default=0         )
parser.add_argument('--seed'            , type=int  , default=None      )
parser.add_argument('--lr0'             , type=float, default=0.5       )
parser.add_argument('--decay-scheme'    , type=str  , default='cyclical')
parser.add_argument('--lr-end'          , type=float, default=0         )
parser.add_argument('--save'            , type=str  , default=None      )
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

# setup training
num_batch = 50000/args.batch_size+1
M = 4 # number of cycles
T = args.epoch*num_batch # total number of iterations

criterion = torch.nn.CrossEntropyLoss().to(args.gpu)
opt = torch.optim.SGD(net.parameters(),lr=args.lr0,weight_decay=5e-4)

def noise(net,coeff):
    _noise = 0
    for param in net.parameters():
        _noise += torch.sum(param*torch.randn_like(param.data)*coeff)
    return _noise

# training at each epoch
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss,correct = 0,0

    for batch_idx,(inputs,targets) in tqdm(enumerate(trainloader)):
        inputs,targets = inputs.to(args.gpu),targets.to(args.gpu)

        lr = utils.lr_decay(args,opt,epoch,batch_idx,num_batch,T,M)
        outputs = net(inputs)
        loss = criterion(outputs,targets)

        opt.zero_grad()
        loss.backward()
        opt.step()

        outputs,loss = outputs.detach(),loss.detach()
        train_loss += loss.data.item()
        _,predicted = torch.max(outputs.data,1)
        correct += predicted.eq(targets.data).sum().item()

    print('Loss: %.3f | ACC: %.3f%% (%d/50000)' % (train_loss/num_batch,100.*correct/50000,correct))

# main loop
print('==> Training...')
acc_list = []
w_list = []

start = time()

for epoch in range(args.epoch):
    train(epoch)
    acc = test_eval.test(args.gpu,net,testloader)

    if args.save:
        acc_list.append(acc)

end = time()

print('\n==> Final Testing...')
test_eval.multi_test(args.gpu,net,[net.state_dict()],testloader,oodloader,corrupt=(args.num_class==10))

# report time usage
minute = (end-start)/60
if minute<=60:
    print(f'Finished in {minute:.1f} min')
else:
    print(f'Finished in {minute/60:.1f} h')

# save acc
if args.save:
    torch.save(net.state_dict(),'%s_%d.pt' % (args.save,epoch))
    np.save('%s_acc.npy' % args.save,np.array(acc_list))
