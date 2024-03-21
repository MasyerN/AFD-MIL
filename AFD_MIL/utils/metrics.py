import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize

def compute_score(target, pred, probs, avg='macro'):
    results = {}
    results['precision'] = metrics.precision_score(target, pred, average=avg)
    results['recall'] = metrics.recall_score(target, pred, average=avg)
    results['f1'] = metrics.f1_score(target, pred, average=avg)
    results['acc'] = metrics.accuracy_score(target, pred)
    results['auc'] = metrics.roc_auc_score(label_binarize(target, classes=[0, 1, 2, 3, 4]), probs, multi_class='ovo')
    return results


def compute_score_2class(target, pred, probs, avg='macro'):
    
    results = {}
    results['precision'] = metrics.precision_score(target, pred)
    results['recall'] = metrics.recall_score(target, pred)
    results['f1'] = metrics.f1_score(target, pred)
    results['acc'] = metrics.accuracy_score(target, pred)
    results['auc'] = metrics.roc_auc_score(target, pred)
    results['auc'] = metrics.roc_auc_score(target, probs)

    return results



def compute_score_4class(target, pred, avg='macro'):
    bcc = 0
    mela = 0
    sk = 0
    scc = 0
    for i in range(len(target)):
        if target[i] == 0 and pred[i] == 0:
            bcc += 1
        elif target[i] == 1 and pred[i] == 1:
            mela += 1
        elif target[i] == 2 and pred[i] == 2:
            sk += 1
        elif target[i] == 3 and pred[i] == 3:
            scc += 1 
    results = {}
    results['precision'] = metrics.precision_score(target, pred, average=avg)
    results['recall'] = metrics.recall_score(target, pred, average=avg)
    results['f1'] = metrics.f1_score(target, pred, average=avg)
    results['acc'] = metrics.accuracy_score(target, pred)
    # results['auc'] = metrics.roc_auc_score(target, pred, average=avg)
    return results, [bcc, mela, sk, scc]



