import torch
import torch.nn.functional as F
import utils
import data_loader
import metric

# file names at CIFAR-10-C
corrupt_types = [
    'brightness.npy',
    'contrast.npy',
    'defocus_blur.npy',
    'elastic_transform.npy',
    'fog.npy',
    'frost.npy',
    'gaussian_blur.npy',
    'gaussian_noise.npy',
    'glass_blur.npy',
    'impulse_noise.npy',
    'jpeg_compression.npy',
    'motion_blur.npy',
    'pixelate.npy',
    'saturate.npy',
    'shot_noise.npy',
    'snow.npy',
    'spatter.npy',
    'speckle_noise.npy',
    'zoom_blur.npy'
]

# test procedure for a single sample
def test(gpu,net,testloader,oodloader=None):
    net.eval()

    # test set
    val_targets = torch.tensor(testloader.dataset.targets,dtype=torch.int64)
    output,uncertainty = [],[]

    for _,(inputs,_) in enumerate(testloader):
        with torch.no_grad():
            o = net(inputs.to(gpu))
            u = -torch.sum(F.softmax(o,dim=1)*F.log_softmax(o,dim=1),dim=1)

        output.append(o.detach().cpu())
        uncertainty.append(u.detach().cpu())

    output = torch.cat(output,dim=0)
    uncertainty = torch.cat(uncertainty)

    acc = metric.score(output,uncertainty,val_targets)

    if not oodloader: return acc

    # OOD set
    val_targets_ood = torch.tensor(oodloader.dataset.labels,dtype=torch.int64)
    output_ood,uncertainty_ood = [],[]

    for _,(inputs,_) in enumerate(oodloader):
        with torch.no_grad():
            o = net(inputs.to(gpu))
            u = -torch.sum(F.softmax(o,dim=1)*F.log_softmax(o,dim=1),dim=1)

        output_ood.append(o.detach().cpu())
        uncertainty_ood.append(u.detach().cpu())

    output_ood = torch.cat(output_ood,dim=0)
    uncertainty_ood = torch.cat(uncertainty_ood)

    acc_ood = metric.score_ood(uncertainty,uncertainty_ood,val_targets,val_targets_ood)

    return acc

# test procedure for multiple samples
def multi_test(gpu,net,w_list,testloader,oodloader,corrupt:bool=False):
    print('%d samples in total' % len(w_list))

    # test set
    val_targets = torch.tensor(testloader.dataset.targets,dtype=torch.int64)
    prob_list = []

    for w in w_list:
        net.load_state_dict(w)
        net.eval()
        prob = []

        for _,(inputs,_) in enumerate(testloader):
            with torch.no_grad():
                o = net(inputs.to(gpu))
                # p = F.softmax(o,dim=1)
            prob.append(o.detach().cpu())

        prob_list.append(torch.cat(prob,dim=0))

    final_prob = sum(prob_list)/len(prob_list)
    uncertainty = -torch.sum(F.softmax(final_prob,dim=1)*F.log_softmax(final_prob,dim=1),dim=1)

    print('\nTest set:')
    metric.score(final_prob,uncertainty,val_targets)

    # OOD set
    val_targets_ood = torch.tensor(oodloader.dataset.labels,dtype=torch.int64)
    prob_list = []

    for w in w_list:
        net.load_state_dict(w)
        net.eval()
        prob = []

        for _,(inputs,_) in enumerate(oodloader):
            with torch.no_grad():
                o = net(inputs.to(gpu))
                # p = F.softmax(o,dim=1)
            prob.append(o.detach().cpu())

        prob_list.append(torch.cat(prob,dim=0))

    final_prob = sum(prob_list)/len(prob_list)
    uncertainty_ood = -torch.sum(F.softmax(final_prob,dim=1)*F.log_softmax(final_prob,dim=1),dim=1)

    print('\nOOD Set:')
    metric.score_ood(uncertainty,uncertainty_ood,val_targets,val_targets_ood)

    if not corrupt: return

    # corrupted set
    print('\nCorrupt Set:')
    for c_name in corrupt_types:
        accs = []
        for severity in range(1,6):
            c_loader = data_loader.corrupted_cifar10(c_name,severity)
            val_targets = torch.tensor(c_loader.dataset.targets,dtype=torch.int64)
            prob_list = []

            for w in w_list:
                net.load_state_dict(w)
                net.eval()
                prob = []

                for _,(inputs,_) in enumerate(c_loader):
                    with torch.no_grad():
                        o = net(inputs.to(gpu))
                        # p = F.softmax(o,dim=1)
                    prob.append(o.detach().cpu())

                prob_list.append(torch.cat(prob,dim=0))

            final_prob = sum(prob_list)/len(prob_list)
            uncertainty = -torch.sum(F.softmax(final_prob,dim=1)*F.log_softmax(final_prob,dim=1),dim=1)

            accs.append(100.*metric.score(final_prob,uncertainty,val_targets,verbose=False))

        print('\'%s\':' % c_name,accs)
