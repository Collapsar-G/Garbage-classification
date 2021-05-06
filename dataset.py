#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $classes.py

@Time    :   $2021.5.6 $18:40

@Desc    :   dataset
"""

from PIL import Image
import torch
from torch.utils.data import Dataset


class GarbageDataSet(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        super(GarbageDataSet, self).__init__()
        self.imgs = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)
