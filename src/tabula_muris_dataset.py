# coding=utf-8
from anndata import read_h5ad
import torch.utils.data as data
import numpy as np
import torch
import os
import math
import json
from preprocessing import get_processed_tabula_muris_GO_data
'''
Inspired by https://github.com/pytorch/vision/pull/46
'''


class TabulaMurisDataset(data.Dataset):

    splits_folder = os.path.join('splits', 'vinyals')
    processed_folder = 'data'

    def __init__(self, opt, mode='train', root='../../data/tabula_muris'):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - mode: train, test or val
        - root: the directory where the dataset will be stored
        '''

        super(TabulaMurisDataset, self).__init__()
        self.root = root
        self.mode = mode
        self.opt = opt
        self.nn_architecture = opt.nn_architecture
        # Default split
        self.split = {'train': ['Bladder', 'Lung', 'Kidney', 'Heart', 'Pancreas', 'Brain_Myeloid', 'Marrow',
                                    'Mammary_Gland', 'Brain_Non-Myeloid', 'Trachea', 'Fat'],
                          'val': ['Large_Intestine', 'Liver', 'Thymus'],
                          'test': ['Skin', 'Tongue', 'Spleen', 'Limb_Muscle']}
        if opt.split_file:
            self.load_split()
        # load anndata
        self.adata = read_h5ad(os.path.join(self.root, "tabula-muris-official-processed.h5ad"))

        # load self.x, self.y
        self.x, self.y, self.n_items = None, [], 0
        self.load_data()
        self.go2gene = get_processed_tabula_muris_GO_data(adata=self.adata, GO_min_genes=5000,
                                                          GO_max_genes=None, GO_min_level=3)
        self.go_mask = self.create_go_mask()

    def create_go_mask(self):
        genes = self.adata.var_names
        gene2index = {g: i for i, g in enumerate(genes)}
        GO_IDs = sorted(self.go2gene.keys())
        go_mask = []
        for go in GO_IDs:
            go_genes = self.go2gene[go]
            go_mask.append([gene2index[gene] for gene in go_genes])
        return go_mask

    def load_split(self):
        with open(self.opt.split_file) as f:
            self.split = json.load(f)

    def load_data(self):
        self.idx_classes = {}
        self.y = []
        tissues = self.split[self.mode]

        # subset data based on target tissues
        self.adata = self.adata[self.adata.obs['tissue'].isin(tissues)]

        # filter gene less than required amount
        if self.mode == 'train':
            min_samples = self.opt.num_support_tr + self.opt.num_query_tr
        else:
            # TODO: change this if there is num_support_test in the future
            min_samples = self.opt.num_support_val + self.opt.num_query_val
        filtered_index = self.adata.obs.groupby(["label"]) \
                                       .filter(lambda group: len(group) >= min_samples) \
                                       .reset_index()['index']
        self.adata = self.adata[filtered_index]

        # convert gene to torch tensor x
        self.x = self.adata.to_df().to_numpy(dtype=np.float32)
        self.process_x_tensor()
        # convert label to torch tensor y
        self.y = self.adata.obs['label'].cat.codes.to_numpy(dtype=np.int32)

        self.n_items = self.x.shape[0]
        print("Mode: {}. Loaded {} classes.".format(self.mode, len(self.adata.obs['label'].cat.codes.unique())))
        print("X shape: {}. Y length: {}.".format(self.x.shape, len(self.y)))

    def __getitem__(self, idx):
        x = self.x[idx]
        return x, self.y[idx]

    def __len__(self):
        return self.n_items

    def get_dim(self):
        if self.nn_architecture == 'fully_connected':
            return self.x[0].shape[0]
        else:
            return 1

    def process_x_tensor(self):
        if self.nn_architecture == 'conv':
            num_data = self.x.shape[0]
            num_dim = self.x.shape[1]
            sqrt = int(math.ceil(math.sqrt(num_dim)))
            # pad zero and then reshape to square matrix
            self.x = np.pad(self.x, ((0, 0), (0, sqrt**2 - num_dim)), 'constant')
            self.x = torch.from_numpy(self.x)
            self.x = self.x[:, :sqrt**2].view(num_data, 1, sqrt, sqrt)
        else:
            self.x = torch.from_numpy(self.x)
        return

