#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $classes.py

@Time    :   $2021.5.6 $18:50

@Desc    :   参数文件

"""

from easydict import EasyDict as edict

__c = edict()
cfg = __c

__c.gpu_num = 0

__c.num_classes = 14

__c.batch_size = 128

__c.num_workers = 1

__c.num_epochs = 3

__c.learning_rate = 5e-3

__c.momentum = .9

__c.input_size = 224

__c.test_percent = .3

__c.model_local = "./models/resnet50-19c8e357.pth"

__c.save_dir_obs = './output/model/model.pth'

__c.data_path = './data/'