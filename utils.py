import torch
import numpy as np
import random
from math import *

# setup GPU
def GPU_setup(gpu:int=0,seed:int=None):
    print('==> Using GPU %d' % gpu)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if seed:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        print('==> Random seed: %d' % seed)

# adjust learning rate
def lr_decay(args,opt,epoch:int,batch_idx:int,num_batch:int,T:int,M:int):
    lr0,lr1 = args.lr0,args.lr_end

    if args.decay_scheme=='cyclical':
        rcounter = epoch*num_batch+batch_idx
        cos_inner = pi*(rcounter%(T//M))
        cos_inner /= T//M
        cos_out = cos(cos_inner)+1
        lr = lr1+(lr0-lr1)/2*cos_out
    elif args.decay_scheme=='exp':
        lr = lr0*((lr1/lr0)**(epoch/args.epoch))
    elif args.decay_scheme=='linear':
        lr = lr1+(args.epoch-epoch)*(lr0-lr1)/args.epoch
    elif args.decay_scheme=='step':
        if epoch<=args.epoch-40:
            lr = lr0
        elif epoch<=args.epoch-20:
            lr = lr0/5
        else:
            lr = lr0/25
    else:
        lr = lr0

    if opt:
        for param_group in opt.param_groups:
            param_group['lr'] = lr

    return lr

# resample net1 from net2
def resample(net1,net2,eta:float=0):
    net1.load_state_dict(net2.state_dict())

    if eta>0:
        for param in net1.parameters():
            param.data += sqrt(eta)*torch.randn_like(param.data)
    elif eta<0:
        assert False, 'Invalid eta!'

    return net1

# smooth regularization & random noise
def reg_noise(net1,net2,datasize:int,alpha:float,eta:float,temperature:float):
    reg_coeff = 0.5/(eta*datasize)
    noise_coeff = sqrt(2/alpha/datasize*temperature)
    loss = 0

    for param1,param2 in zip(net1.parameters(),net2.parameters()):
        sub = param1-param2
        reg = sub*sub*reg_coeff
        noise1 = param1*torch.randn_like(param1.data)*noise_coeff
        noise2 = param2*torch.randn_like(param2.data)*noise_coeff
        loss += torch.sum(reg-noise1-noise2)

    return loss
