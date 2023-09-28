import torch
from torchvision import transforms,datasets
import numpy as np

_dataset_mean = {
    'cifar10'   : (0.4914, 0.4822, 0.4465),
    'cifar100'  : (0.5071, 0.4867, 0.4408),
    'imagenet'  : (0.4850, 0.4560, 0.4060),
    'svhn'      : (0.5000, 0.5000, 0.5000)
}
_dataset_std = {
    'cifar10'   : (0.2023, 0.1994, 0.2010),
    'cifar100'  : (0.2675, 0.2565, 0.2761),
    'imagenet'  : (0.2290, 0.2240, 0.2250),
    'svhn'      : (0.5000, 0.5000, 0.5000)
}

# CIFAR10/100
def cifar(args,num_classes:int):
    if num_classes==10:
        _dataset_name = 'cifar10'
    elif num_classes==100:
        _dataset_name = 'cifar100'
    else:
        assert False, 'Dataset not supported!'

    _train_transform = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_dataset_mean[_dataset_name],_dataset_std[_dataset_name])
    ])
    _test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_dataset_mean[_dataset_name],_dataset_std[_dataset_name])
    ])

    if _dataset_name=='cifar10':
        _train_set = datasets.CIFAR10(root=args.data_dir,train=True,download=True,transform=_train_transform)
        _test_set = datasets.CIFAR10(root=args.data_dir,train=False,download=True,transform=_test_transform)
    elif _dataset_name=='cifar100':
        _train_set = datasets.CIFAR100(root=args.data_dir,train=True,download=True,transform=_train_transform)
        _test_set = datasets.CIFAR100(root=args.data_dir,train=False,download=True,transform=_test_transform)
    else:
        assert False, 'Dataset not supported!'

    train_loader = torch.utils.data.DataLoader(_train_set,batch_size=args.batch_size,shuffle=True,drop_last=False,num_workers=0)
    test_loader = torch.utils.data.DataLoader(_test_set,batch_size=1000,shuffle=False,drop_last=False,num_workers=0)

    return train_loader,test_loader

# ImageNet
def imagenet(args):
    _train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0),
        transforms.ToTensor(),
        transforms.Normalize(_dataset_mean['imagenet'],_dataset_std['imagenet'])
    ])
    _test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_dataset_mean['imagenet'],_dataset_std['imagenet'])
    ])

    _train_set = datasets.ImageNet(root=args.data_dir,split='train',download=None,transform=_train_transform)
    _test_set = datasets.ImageNet(root=args.data_dir,split='val',download=None,transform=_test_transform)

    train_loader = torch.utils.data.DataLoader(_train_set,batch_size=args.batch_size,shuffle=True,drop_last=False,num_workers=12)
    test_loader = torch.utils.data.DataLoader(_test_set,batch_size=2*args.batch_size,shuffle=False,drop_last=False,num_workers=12)

    return train_loader,test_loader

# SVHN (OOD)
def svhn(args):
    _ood_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_dataset_mean['svhn'],_dataset_std['svhn'])
    ])

    _ood_set = datasets.SVHN(root=args.data_dir,split="test",download=True,transform=_ood_transform)

    ood_loader = torch.utils.data.DataLoader(_ood_set,batch_size=1000,shuffle=False,drop_last=False,num_workers=0)

    return ood_loader

# corrupted CIFAR10
class CIFAR10_C(torch.utils.data.Dataset):
    def __init__(self,corrupt_option,severity,transform):
        assert type(severity) is int and 1<=severity<=5, 'Invalid severity!'

        self.root = '/home/bolian/CIFAR-10-C/'

        self.data = np.load(self.root+corrupt_option)[10000*(severity-1):10000*severity]
        self.targets = np.load(self.root+'labels.npy')[10000*(severity-1):10000*severity]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self,index):
        img,target = self.data[index],self.targets[index]
        return self.transform(img),target

def corrupted_cifar10(corrupt_option,severity):
    _test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_dataset_mean['cifar10'],_dataset_std['cifar10'])
    ])

    _test_set = CIFAR10_C(corrupt_option=corrupt_option,severity=severity,transform=_test_transform)
    test_loader = torch.utils.data.DataLoader(_test_set,batch_size=1000,shuffle=False,drop_last=False,num_workers=0)

    return test_loader
