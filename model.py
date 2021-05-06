#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $classes.py

@Time    :   $2021.5.6 $18:40

@Desc    :   模型

"""
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
from miscc.config import cfg


def base_resnet():
    model = models.resnet50(pretrained=False)
    model.load_state_dict(torch.load(cfg.model_local))
    for param in model.parameters():
        param.requires_grad = False
    num_fc_if = model.fc.in_features
    model.fc = nn.Linear(num_fc_if, cfg.num_classes)

    return model


if __name__ == "__main__":
    models = base_resnet()
    print(models)
