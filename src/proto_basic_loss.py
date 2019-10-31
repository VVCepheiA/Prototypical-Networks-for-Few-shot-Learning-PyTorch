# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def get_idxs(n_support, target):
    """Get indexes of support and query sets.
        n_support: number of examples in support set
        target: target values for current batch
    """
    classes = torch.unique(target)
    support_idxs = list(map(lambda c: target.eq(c).nonzero()[:n_support].squeeze(1), classes))
    l = list(map(lambda c: target.eq(c).nonzero()[n_support:], classes))
    query_idxs = torch.cat(list(map(lambda c: target.eq(c).nonzero()[n_support:], classes)))
    query_idxs  = query_idxs.view(-1)
    
    #query_idxs = torch.stack(list(map(lambda c: target.eq(c).nonzero()[n_support:], classes))).view(-1)
    #query_idxs  = query_idxs.view(-1)
    
    return support_idxs, query_idxs

def prototypical_loss(input, target, n_support):
    input_cpu = input.to('cpu')
   
    support_idxs, query_idxs = get_idxs(n_support, target)
    prototypes = torch.stack([input_cpu[idx_class].mean(0) for idx_class in support_idxs])

    query_samples = input_cpu[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1)

    ######## Experimental ########
    target = target[query_idxs]
    target_inds = target.long()
    ######## Experimental ########
    
    loss = torch.nn.NLLLoss()
    loss_val = loss(log_p_y, target_inds)
     
    _, y_hat = log_p_y.max(1)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    target_flattened = torch.flatten(target_inds.squeeze())
    y_hat_flattened = torch.flatten(y_hat)

    f1_macro = f1_score(target_flattened, y_hat_flattened, average='macro')
    f1_micro = f1_score(target_flattened, y_hat_flattened, average='micro')

    metrics = (acc_val, f1_macro, f1_micro)
    
    return loss_val,  metrics


