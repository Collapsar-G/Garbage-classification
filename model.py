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


class base_resnet(nn.Module):
    def __init__(self, ):
        super(base_resnet, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.load_state_dict(torch.load(cfg.model_local))
        for param in self.model.parameters():
            param.requires_grad = False
        num_fc_if = self.model.fc.in_features
        self.model.fc = nn.Linear(num_fc_if, cfg.num_classes)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = self.model.fc(x)

        return x


if __name__ == "__main__":
    models = base_resnet()
    print(models)
