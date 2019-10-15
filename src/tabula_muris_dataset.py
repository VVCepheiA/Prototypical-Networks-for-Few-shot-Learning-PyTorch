# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import shutil
import errno
from pprint import pprint
import torch
import os
from tqdm import tqdm

'''
Inspired by https://github.com/pytorch/vision/pull/46
'''


class TabulaMurisDataset(data.Dataset):

    splits_folder = os.path.join('splits', 'vinyals')
    processed_folder = 'data'
    tabula_muris_split = {'train': ['Bladder', 'Lung', 'Kidney', 'Heart', 'Pancreas', 'Brain_Myeloid', 'Marrow',
                                    'Mammary_Gland', 'Brain_Non-Myeloid', 'Trachea', 'Fat'],
                          'val': ['Large_Intestine', 'Liver', 'Thymus'],
                          'test': ['Skin', 'Tongue', 'Spleen', 'Limb_Muscle']}

    def __init__(self, mode='train', root='../../data/tabula_muris'):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        '''

        super(TabulaMurisDataset, self).__init__()
        self.root = root
        self.mode = mode

        # load self.x, self.y, self.classes, self.idx_classes
        self.x, self.y, self.idx_classes, self.n_items = None, [], {}, 0
        self.load_data()

    def load_data(self):
        self.idx_classes = {}
        tensors = []
        self.y = []
        tissues = self.tabula_muris_split[self.mode]
        print("Loading data...")
        for tissue in tqdm(tissues):
            tissue_dir = os.path.join(self.root, tissue)
            cell_ontology_classes = os.listdir(tissue_dir)
            for cell_ontology_class in cell_ontology_classes:
                label = tissue + "/" + cell_ontology_class
                self.idx_classes[label] = len(self.idx_classes)
                path = os.path.join(tissue_dir, cell_ontology_class, "genes.csv")
                tensor = read_csv_to_tensor(path)
                tensors.append(tensor)
                self.y += tensor.shape[0] * [self.idx_classes[label]]
        self.x = torch.cat(tensors, 0)
        self.n_items = self.x.shape[0]
        print("Mode: {}. Loaded {} classes.".format(self.mode, len(self.idx_classes)))
        print("X shape: {}. Y length: {}.".format(self.x.shape, len(self.y)))

    def __getitem__(self, idx):
        x = self.x[idx]
        return x, self.y[idx]

    def __len__(self):
        return self.n_items


def read_csv_to_tensor(path, trim=150):
    df = pd.read_csv(path, header=None)
    x = torch.from_numpy(df.to_numpy(dtype=np.float32))
    num_data = x.shape[0]
    x = x[:, :trim**2].view(num_data, 1, trim, trim)
    return x

