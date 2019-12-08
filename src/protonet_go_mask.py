import torch.nn as nn


class MaskedDropout(nn.Module):
    def __init__(self, go_mask=1):
        super(MaskedDropout, self).__init__()
        self.go_mask = go_mask

    def forward(self, input):
        return input * self.go_mask


def attention_block(in_features, out_features, go_mask):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        MaskedDropout(),
    )

def full_block(in_features, out_features, dropout):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(p=dropout),
    )

class AttentionProtonet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, x_dim, go_mask, hid_dim=64, z_dim=64, dropout=0.2):
        print("Using AttentionProtoNet with x_dim {}".format(x_dim))
        super(AttentionProtonet, self).__init__()

        self.go_mask = go_mask
        self.encoder = nn.Sequential(
            attention_block(x_dim, hid_dim, self.go_mask),
            full_block(hid_dim, z_dim, dropout),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
