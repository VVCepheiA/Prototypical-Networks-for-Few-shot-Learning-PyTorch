# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import numpy as np
import shutil
import errno
from pprint import pprint
import torch
import os

'''
Inspired by https://github.com/pytorch/vision/pull/46
'''

IMG_CACHE = {}


class TabulaMurisDataset(data.Dataset):

    splits_folder = os.path.join('splits', 'vinyals')
    processed_folder = 'data'

    def __init__(self, mode='train', root='..' + os.sep + 'dataset', transform=None, target_transform=None):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        '''
        super(TabulaMurisDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.classes = get_current_classes(os.path.join(
            self.root, self.splits_folder, mode + '.txt'))
        self.all_items = find_items(os.path.join(
            self.root, self.processed_folder), self.classes)

        self.idx_classes = index_classes(self.all_items)

        paths, self.y = zip(*[self.get_path_label(pl)
                              for pl in range(len(self))])

        self.x = map(load_img, paths, range(len(paths)))
        self.x = list(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        return x, self.y[idx]

    def __len__(self):
        return len(self.all_items)

    def get_path_label(self, index):
        '''
        Return image path with rotation and the class id
        '''
        filename = self.all_items[index][0]
        rot = self.all_items[index][-1]
        img = str.join(os.sep, [self.all_items[index][2], filename]) + rot
        target = self.idx_classes[self.all_items[index]
                                  [1] + self.all_items[index][-1]]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


def find_items(root_dir, classes):
    '''
    Find the file information for the classes.
    Return (image_file_name, class, image_dir, rotation)
    '''
    retour = []
    rots = [os.sep + 'rot000', os.sep + 'rot090', os.sep + 'rot180', os.sep + 'rot270']
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            r = root.split(os.sep)
            lr = len(r)
            label = r[lr - 2] + os.sep + r[lr - 1]
            for rot in rots:
                if label + rot in classes and (f.endswith("png")):
                    retour.extend([(f, label, root, rot)])
    print("== Dataset: Found %d items " % len(retour))
    return retour


def index_classes(items):
    '''
    items: (image_file_name, class, image_dir, rotation)
    return: assigned unique int id for each class
    '''
    idx = {}
    for i in items:
        if (not i[1] + i[-1] in idx):
            idx[i[1] + i[-1]] = len(idx)
    print("== Dataset: Found %d classes" % len(idx))
    return idx


def get_current_classes(fname):
    '''
    return the list of classes in *.txt (the label file)
    '''
    with open(fname) as f:
        classes = f.read().replace('/', os.sep).splitlines()
    return classes


def load_img(path, idx):
    '''
    Load image, rotate and resize
    '''
    path, rot = path.split(os.sep + 'rot')
    if path in IMG_CACHE:
        x = IMG_CACHE[path]
    else:
        x = Image.open(path)
        IMG_CACHE[path] = x
    x = x.rotate(float(rot))
    x = x.resize((28, 28))

    shape = 1, x.size[0], x.size[1]
    x = np.array(x, np.float32, copy=False)
    x = 1.0 - torch.from_numpy(x)
    # 1 * 28 * 28
    x = x.transpose(0, 1).contiguous().view(shape)

    return x
