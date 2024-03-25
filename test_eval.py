import torch
import torch.nn.functional as F
import data_loader
import metric
import utils
from math import *
from tqdm import tqdm

# file names in MNIST-C
corrupt_types_mnist = [
    "brightness",
    "canny_edges",
    "dotted_line",
    "fog",
    "glass_blur",
    "identity",
    "impulse_noise",
    "motion_blur",
    "rotate",
    "scale",
    "shear",
    "shot_noise",
    "spatter",
    "stripe",
    "translate",
    "zigzag"
]

# file names in CIFAR-10-C
corrupt_types_cifar = [
    "brightness.npy",
    "contrast.npy",
    "defocus_blur.npy",
    "elastic_transform.npy",
    "fog.npy",
    "frost.npy",
    "gaussian_blur.npy",
    "gaussian_noise.npy",
    "glass_blur.npy",
    "impulse_noise.npy",
    "jpeg_compression.npy",
    "motion_blur.npy",
    "pixelate.npy",
    "saturate.npy",
    "shot_noise.npy",
    "snow.npy",
    "spatter.npy",
    "speckle_noise.npy",
    "zoom_blur.npy",
]

# predictive distributions
def _BMA(gpu, net, w_list, testloader):
    if w_list is None:
        log_final_prob, confidence, uncertainty = [], [], []
        net.eval()

        for _, (inputs, _) in enumerate(testloader):
            with torch.no_grad():
                o = net(inputs.to(gpu))
                log_p = F.log_softmax(o, dim=1)
                
                # Max-class probability as confidence
                c = torch.exp(log_p).max(dim=1).values

                # Predictive uncertainty
                u = -torch.sum(log_p * torch.exp(log_p), dim=1)

            log_final_prob.append(log_p.detach().cpu())
            confidence.append(c.detach().cpu())
            uncertainty.append(u.detach().cpu())

        log_final_prob = torch.cat(log_final_prob, dim=0)
        confidence = torch.cat(confidence)
        uncertainty = torch.cat(uncertainty)

    else:
        assert len(w_list) > 0, "No sample in w_list!"

        prob_list = []

        for w in tqdm(w_list):
            net = utils.load_sample(net, w)
            net.eval()
            prob = []

            for _,(inputs,_) in enumerate(testloader):
                with torch.no_grad():
                    o = net(inputs.to(gpu))
                    p = F.softmax(o, dim=1)
                prob.append(p.detach().cpu())

            prob_list.append(torch.cat(prob, dim=0))

        final_prob = sum(prob_list) / len(prob_list)
        log_final_prob = torch.log(final_prob + 5e-44)

        # Max-class probability as confidence
        confidence = final_prob.max(dim=1).values

        # Predictive uncertainty
        uncertainty = -torch.sum(final_prob * log_final_prob, dim=1)

    return log_final_prob, confidence, uncertainty

# test procedure for a single sample
def test(gpu, net, testloader, oodloader=None):
    net.to(gpu)
    net.eval()

    # test set
    val_targets = torch.tensor(testloader.dataset.targets, dtype=torch.int64)
    log_prob, confidence, uncertainty = _BMA(gpu, net, None, testloader)
    acc = metric.score(log_prob, confidence, val_targets)

    # OOD set
    if oodloader:
        _, _, uncertainty_ood = _BMA(gpu, net, None, oodloader)
        acc_ood = metric.score_ood(uncertainty, uncertainty_ood)

    return acc

# test procedure for multiple samples
def multi_test(gpu, net, w_list, testloader, oodloader=None, corrupt_mnist:bool=False, corrupt_cifar10:bool=False, distance:bool=False):

    # sample standard deviation
    if distance:
        net.cpu()
        distance = []
        for param in net.parameters():
            param.mean = torch.zeros_like(param.data)
        for w in w_list:
            net = utils.load_from_param_list(net, w)
            for param in net.parameters():
                param.mean += param.data / len(w_list)
        for w in w_list:
            net = utils.load_from_param_list(net, w)
            d2 = 0
            for param in net.parameters():
                sub_d = param.data - param.mean
                d2 += (sub_d * sub_d).sum().item()
            distance.append(sqrt(d2))

        assert len(distance) == len(w_list), "Miss or overcount some samples!"
        print("Avg. Distance: %.4f" % (sum(distance) / len(distance)))
        net.to(gpu)

    # test set
    val_targets = torch.tensor(testloader.dataset.targets, dtype=torch.int64)
    log_final_prob, confidence, uncertainty = _BMA(gpu, net, w_list, testloader)
    metric.score(log_final_prob, confidence, val_targets)

    # OOD set
    if oodloader:
        _, _, uncertainty_ood = _BMA(gpu, net, w_list, oodloader)
        metric.score_ood(uncertainty, uncertainty_ood)

    # corrupted set
    assert not (corrupt_mnist and corrupt_cifar10), "Cannot test MNIST-C and CIFAR10-C at the same time!"

    if corrupt_mnist:
        print("\nNoisy MNIST:")

        for c_name in corrupt_types_mnist:
            c_loader = data_loader.corrupted_mnist("/home/bolian/mnist_c/", c_name)
            val_targets = torch.tensor(c_loader.dataset.targets, dtype=torch.int64)
            log_final_prob, confidence, _ = _BMA(gpu, net, w_list, c_loader)
            acc = metric.score(log_final_prob, confidence, val_targets)
            print("\"%s\": %.2f" % (c_name, acc*100))

    if corrupt_cifar10:
        print('\nCorrupt CIFAR10:')

        for c_name in corrupt_types_mnist:
            accs = []

            for severity in range(1,6):
                c_loader = data_loader.corrupted_cifar10("/home/bolian/CIFAR-10-C/", c_name,severity)
                val_targets = torch.tensor(c_loader.dataset.targets, dtype=torch.int64)
                log_final_prob, confidence, _ = _BMA(gpu, net, w_list, c_loader)
                accs.append(100. * metric.score(log_final_prob, confidence, val_targets))

            print('\'%s\':' % c_name, accs)

# compare samples
def compare_sample(gpu,net1,net2,w_list1,w_list2,testloader):
    val_targets = torch.tensor(testloader.dataset.targets,dtype=torch.int64)

    log_prob1,_,_ = _BMA(gpu,net1,w_list1,testloader)
    log_prob2,_,_ = _BMA(gpu,net2,w_list2,testloader)

    # agreement
    top1_pred1 = torch.argmax(log_prob1,dim=1)
    top1_pred2 = torch.argmax(log_prob2,dim=1)
    agreement = (top1_pred1==top1_pred2).count_nonzero().item()/len(val_targets)
    print("agreement: %.4f" % agreement)

    # total variation
    prob1,prob2 = torch.exp(log_prob1),torch.exp(log_prob2)
    total_var = torch.sum(torch.abs(prob1-prob2))/(2*len(val_targets))
    print("total variation: %.4f" % total_var)
