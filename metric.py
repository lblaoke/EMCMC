import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def _ece_score(y_true, y_pred, num_bins:int=10):
    sorted_idx = torch.argsort(y_pred)
    bin_size = len(sorted_idx) / num_bins

    bin_l = 0
    ece = 0
    for i in range(1, num_bins + 1):
        bin_r = int(i * bin_size)

        binned_y_true = y_true[sorted_idx[bin_l:bin_r]]
        binned_y_pred = y_pred[sorted_idx[bin_l:bin_r]]

        ece += (bin_r - bin_l) * abs(binned_y_true.count_nonzero().item() / len(binned_y_true) - binned_y_pred.mean().item())

        bin_l = bin_r

    return ece / len(sorted_idx)

def score(output, confidence, target, verbose:bool=True):
    pred = torch.argmax(output, dim=1)
    correct = (pred == target)
    acc = correct.count_nonzero().item() / len(correct)

    if verbose:
        nll = torch.nn.functional.cross_entropy(output, target)
        auroc = roc_auc_score(correct, confidence)
        ece = _ece_score(correct, confidence)

        print(f'ACC   (%): {acc*100:.2f}')
        print(f'NLL      : {nll:.3f}')
        print(f'AUROC (%): {auroc*100:.2f}')
        print(f'ECE   (%): {ece*100:.2f}')

    return acc

def score_ood(uncertainty, uncertainty_ood):
    is_ID = torch.cat([torch.zeros_like(uncertainty), torch.ones_like(uncertainty_ood)])
    u = torch.cat([uncertainty, uncertainty_ood])

    # normalize uncertainty to [0, 1]
    u_min, u_max = u.min(), u.max()
    u = (u - u_min) / (u_max - u_min)

    auroc = roc_auc_score(is_ID, u)
    aupr = average_precision_score(is_ID, u)

    print(f'OOD AUROC (%): {auroc*100:.2f}')
    print(f'OOD AUPR  (%): {aupr*100:.2f}')

if __name__ == '__main__':
    y_pred = 1 - torch.rand(10000)
    # print(y_pred)
    y_true = torch.rand(10000)
    y_true = torch.bernoulli(y_pred)
    # print(y_true)
    print(average_precision_score(y_true, y_pred))
    # print(expected_calibration_error(y_true, y_pred))
