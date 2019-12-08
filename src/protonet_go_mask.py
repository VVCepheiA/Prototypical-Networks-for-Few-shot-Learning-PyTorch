import torch.nn as nn
import random
import torch
import numpy as np
import torch.nn.functional as F

def generate_random_go_mask(x_dim, num_GOs=3, num_genes=200):
    """
    Debug tool. Generate random go2gene mask
    :param x_dim:
    :return:
    """
    go_mask = [random.sample(range(1, x_dim), num_genes) for _ in range(num_GOs)]
    return go_mask

def generate_simple_go_mask(x_dim, num_GOs=3):
    """
    Debug tool. Generate GO mask equally dividing all genes into numGO groups
    :param x_dim:
    :return:
    """
    go_mask = np.array_split(range(x_dim), num_GOs)

    return go_mask


class AttentionProtonet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, x_dim, go_mask, hid_dim=64, z_dim=64, dropout=0.2):
        print("Using AttentionProtoNet with x_dim {}".format(x_dim))
        super(AttentionProtonet, self).__init__()

        self.go_mask = generate_simple_go_mask(x_dim=x_dim, num_GOs=3)
        self.num_GOs = len(self.go_mask)
        self.masks = None
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            self.full_block(x_dim, hid_dim),
            self.full_block(hid_dim, z_dim),
        )

        # attention
        self.a = nn.Parameter(torch.zeros(size=(z_dim * self.num_GOs, self.num_GOs)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)


    def full_block(self, in_features, out_features, dropout=0.2):
        return nn.Sequential(
            nn.Linear(in_features, out_features, bias=True),
            nn.BatchNorm1d(self.num_GOs),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

    def generate_masks(self, x):
        batch, num_genes = x.shape
        self.masks = torch.zeros(self.num_GOs, batch, num_genes)
        for i, genes in enumerate(self.go_mask):
            self.masks[i, :, genes] = 1

    def forward(self, x):
        batch, num_genes = x.shape
        # need to generate masks if the batch size change or
        if self.masks is None or self.masks.shape[1] != batch:
            self.generate_masks(x)
        # x before applying mask: (batch, numGenes)
        x = x.view(1, batch, -1)
        # x after applying mask: (numGOs, batch, numGenes)
        x = self.masks * x
        # change to (batch, numGOs, numGenes)
        x = x.permute(1, 0, 2)
        # after encoder should be (batch, numGOs, z_dim)
        x = self.encoder(x)

        ############# use attention #############
        # reshape to (batch, numGOs * z_dim) to get attention scores for each GOs
        # result will be (batch, numGOs)
        attention = torch.mm(x.view(batch, -1), self.a)
        attention = F.softmax(attention, dim=1)
        x = (torch.t(attention).view(self.num_GOs, batch, 1).repeat(1, 1, self.z_dim) * x.permute(1, 0, 2)).sum(0)
        ############# use attention #############

        ############# if no attention #############
        # x = x.mean(1)
        ############# if no attention #############
        return x


    # def forward(self, x):
    #     x = self.encoder(x)
    #     x = x.view(x.size(0), -1)
    #     return x
