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
parser.add_argument('--num-class'       , type=int  , default=100       )
parser.add_argument('--epoch'           , type=int  , default=200       )
parser.add_argument('--batch-size'      , type=int  , default=128       )
parser.add_argument('--gpu'             , type=int  , default=0         )
parser.add_argument('--seed'            , type=int  , default=None      )
parser.add_argument('--eta'             , type=float, default=8e-3      )
parser.add_argument('--lr0'             , type=float, default=0.5       )
parser.add_argument('--decay-scheme'    , type=str  , default='cyclical')
parser.add_argument('--lr-end'          , type=float, default=0         )
parser.add_argument('--temperature'     , type=float, default=1e-4      )
parser.add_argument('--norm'            , type=str  , default='BN'      )
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
if args.norm=='BN':
    net_s = models.ResNet18(num_classes=args.num_class)
    net_a = models.ResNet18(num_classes=args.num_class)
elif args.norm=='FRN':
    net_s = models.ResNet18_FRN(num_classes=args.num_class)
    net_a = models.ResNet18_FRN(num_classes=args.num_class)
else:
    assert False, 'Normalization method not supported!'

net_a = utils.resample(net_a,net_s)
net_s,net_a = net_s.to(args.gpu),net_a.to(args.gpu)

# setup training
num_batch = 50000/args.batch_size+1
M = 4 # number of cycles
T = args.epoch*num_batch # total number of iterations

criterion = torch.nn.CrossEntropyLoss().to(args.gpu)
opt = torch.optim.SGD(list(net_s.parameters())+list(net_a.parameters()),lr=args.lr0,weight_decay=5e-4)

# training at each epoch
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net_s.train()
    train_loss,correct = 0,0

    for batch_idx,(inputs,targets) in tqdm(enumerate(trainloader)):
        inputs,targets = inputs.to(args.gpu),targets.to(args.gpu)

        lr = utils.lr_decay(args,opt,epoch,batch_idx,num_batch,T,M)
        outputs = net_s(inputs)
        loss = criterion(outputs,targets)+utils.reg_noise(net_s,net_a,50000,lr,args.eta,args.temperature)

        opt.zero_grad()
        loss.backward()
        opt.step()

        outputs,loss = outputs.detach(),loss.detach()
        train_loss += loss.data.item()
        _,predicted = torch.max(outputs.data,1)
        correct += predicted.eq(targets.data).sum().item()

    print('Loss: %.3f | ACC: %.3f%% (%d/50000)' % (train_loss/num_batch,100.*correct/50000,correct))

# additional forward to re-compute BatchNorm
def additional_forward(net):
    net.train()
    for _,(inputs,_) in enumerate(trainloader):
        net(inputs.to(args.gpu))

# main loop
print('==> Training...')
acc_s_list,acc_a_list = [],[]
w_s_list,w_a_list = [],[]

start = time()

for epoch in range(args.epoch):
    train(epoch)
    if args.norm=='BN': additional_forward(net_a)

    if (epoch%50)+1>48:
        # sampler
        acc_s = test_eval.test(args.gpu,net_s,testloader,oodloader)
        w_net_s = net_s.state_dict()
        w_s_list.append(w_net_s)
        print('Sampler collected')

        # anchor
        acc_a = test_eval.test(args.gpu,net_a,testloader,oodloader)
        w_net_a = net_a.state_dict()
        w_a_list.append(w_net_a)
        print('Anchor collected')

        if args.save:
            torch.save(w_net_s,'%s_s%d.pt' % (args.save,epoch))
            torch.save(w_net_a,'%s_a%d.pt' % (args.save,epoch))

    else:
        acc_s = test_eval.test(args.gpu,net_s,testloader)
        acc_a = test_eval.test(args.gpu,net_a,testloader)

    if args.save:
        acc_s_list.append(acc_s)
        acc_a_list.append(acc_a)

end = time()

print('\n==> Final Testing...')
test_eval.multi_test(args.gpu,net_s,w_s_list+w_a_list,testloader,oodloader)

# report time usage
minute = (end-start)/60
if minute<=60:
    print(f'Finished in {minute:.1f} min')
else:
    print(f'Finished in {minute/60:.1f} h')

# save acc
if args.save:
    np.save('%s_s_acc.npy' % args.save,np.array(acc_s_list))
    np.save('%s_a_acc.npy' % args.save,np.array(acc_a_list))
