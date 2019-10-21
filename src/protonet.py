import torch.nn as nn


def full_block(in_features, out_features, dropout):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(p=dropout),
    )


def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ProtoNet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64, nn_architecture='conv', dropout=0.2):
        print("Using nn_architecture {} with x_dim {}".format(nn_architecture, x_dim))
        super(ProtoNet, self).__init__()
        if nn_architecture == 'conv':
            self.encoder = nn.Sequential(
                conv_block(x_dim, hid_dim),
                conv_block(hid_dim, hid_dim),
                conv_block(hid_dim, hid_dim),
                conv_block(hid_dim, z_dim),
            )
        elif nn_architecture == 'fully_connected':
            self.encoder = nn.Sequential(
                full_block(x_dim, hid_dim, dropout),
                full_block(hid_dim, z_dim, dropout),
            )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
