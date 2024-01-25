from __future__ import print_function
import sys
sys.path.append('.')

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import numpy as np
from torch.autograd import Variable
import utils
from math import sqrt
torch.set_default_tensor_type('torch.DoubleTensor')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 1, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1,784)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

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

def train(args, net, net_anchor, train_loader, test_loader, optimizer, epoch):
    _loss,_acc = [],[]
    for batch_idx, (data, target) in enumerate(train_loader):
        net.train()
        data, target = Variable(data.type(torch.DoubleTensor)), Variable(target)

        optimizer.zero_grad()
        output = net(data)

        log_softmax = [torch.log(1.0-output),torch.log(output)]
        log_softmax = torch.cat(log_softmax,1)

        loss = F.nll_loss(log_softmax, target)+utils.reg_noise(net,net_anchor,60000,args.lr,1,1)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
            _loss.append(loss.data.item())
            _acc.append(test(args,net,test_loader))

    return _loss,_acc

def test(args, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data.type(torch.DoubleTensor)), Variable(target)
        output = model(data)
        nll = [torch.log(1.0-output),torch.log(output)]
        nll = torch.cat(nll,1)
        test_loss += F.nll_loss(nll, target).data.item() # sum up batch loss
        pred = nll.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.03, metavar='LR',
                        help='learning rate (default: 0.01)')# normalized images: lr = 0.001
    parser.add_argument('--momentum', type=float, default=0., metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    train_set = MNIST('~', train=True, download=True,transform=transforms.Compose([
                   transforms.ToTensor(),
                #    transforms.Normalize((0.1307,), (0.3081,))
                   ]))

    idx = np.where(np.logical_or(train_set.targets.numpy() == 1, train_set.targets.numpy() == 7))[0]
    train_set.targets = train_set.targets[idx]
    train_set.data = train_set.data[idx]
    train_set.targets[train_set.targets == 1] = 0
    train_set.targets[train_set.targets == 7] = 1
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    test_set = MNIST('~', train=False, download=True,transform=transforms.Compose([
                       transforms.ToTensor(),
                    #    transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    idx = np.where(np.logical_or(test_set.targets.numpy() == 1, test_set.targets.numpy() == 7))[0]
    test_set.targets = test_set.targets[idx]
    test_set.data = test_set.data[idx]
    test_set.targets[test_set.targets == 1] = 0
    test_set.targets[test_set.targets == 7] = 1

    test_loader = torch.utils.data.DataLoader(test_set,
    batch_size=args.test_batch_size, shuffle=True)

    net = Net()#.to(device)
    net_anchor = Net()

    net_anchor = utils.resample(net_anchor,net)

    optimizer = optim.SGD(list(net.parameters())+list(net_anchor.parameters()), lr=args.lr, weight_decay=1e-5)

    train_loss,test_acc = [],[]
    for epoch in range(1, args.epochs + 1):
        _train_loss,_test_acc = train(args, net, net_anchor,  train_loader, test_loader, optimizer, epoch)
        train_loss += _train_loss
        test_acc += _test_acc

    # torch.save(model.state_dict(),'checkpoints/model_sgd_e%d.pt'%(args.epochs))
    np.save('logistic_reg_emcmc%d_loss.npy' % args.seed,np.array(train_loss))
    np.save('logistic_reg_emcmc%d_acc.npy' % args.seed,np.array(test_acc))

if __name__ == '__main__':
    main()