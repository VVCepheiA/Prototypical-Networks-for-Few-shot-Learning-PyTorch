# coding=utf-8
import torch

class EpochSampler(object):
    '''
    EpochSampler: yield permuted indexes at each epoch.
   
    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, nepochs):
        '''
        Initialize the EpochSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - iterations: number of epochs
        '''
        super(EpochSampler, self).__init__()
        self.labels = labels
        self.nepochs = nepochs


    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        
        for it in range(self.nepochs):
            epoch = torch.LongTensor(torch.randperm(len(self.labels)))
            
            yield epoch

