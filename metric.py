import torch
from sklearn.metrics import roc_auc_score

def _ece_score(y_true,y_pred,bins:int=10):
    ece = 0.
    for i in range(bins):
        c_start,c_end = i/bins,(i+1)/bins
        mask = (c_start<=y_pred)&(y_pred<c_end)

        ni = mask.count_nonzero().item()
        if ni<=0: continue

        acc,conf = y_true[mask].sum()/ni,y_pred[mask].mean()
        ece += ni*(acc-conf).abs()

    return float(ece)/len(y_true)

def score(output,uncertainty,target,verbose:bool=True):
    pred = torch.argmax(output,dim=1)
    correct = (pred==target)
    acc = correct.sum().item()/len(target)

    if verbose:
        confidence = 1-uncertainty

        nll = torch.nn.functional.cross_entropy(output,target)
        auc = roc_auc_score(correct,confidence)
        ece = _ece_score(correct,confidence)

        print(f'ACC (%): {acc*100:.2f}')
        print(f'NLL    : {nll:.3f}')
        print(f'AUC (%): {auc*100:.2f}')
        print(f'ECE (%): {ece*100:.2f}')

    return acc

def score_ood(uncertainty,uncertainty_ood,target,target_ood):
    uncertainty = torch.cat([uncertainty,uncertainty_ood])
    confidence = 1-uncertainty
    correct = torch.cat([torch.ones_like(target),torch.zeros_like(target_ood)])

    auc = roc_auc_score(correct,confidence)
    ece = _ece_score(correct,confidence)
    print(f'OOD AUC (%): {auc*100:.2f}')
    print(f'OOD ECE (%): {ece*100:.2f}')
