"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $classes.py

@Time    :   $2021.5.6 $18:40

@Desc    :   模型训练入口
"""
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from dataset import GarbageDataSet
from torch.utils.data import DataLoader

from model import base_resnet
from miscc.utils import dataloader, custom_split
from miscc.config import cfg


def train_model(model, dataloaders, criterion, optimizer, num_epochs=cfg.num_epochs):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for index, (inputs, labels) in enumerate(dataloaders[phase]):
                if phase == 'train':
                    print('batch: %d, left: %d, percent: %6.4f' % (
                        index, float(len(dataloaders[phase].dataset)) / cfg.batch_size - 1 - index,
                        float((index + 1) * cfg.batch_size) / len(dataloaders[phase].dataset)))
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    labels = labels.to(outputs.device)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def save_model(model, save_dir):
    torch.save(model, cfg.save_dir_obs)
    print('save model_state success')


if __name__ == "__main__":
    model = base_resnet()
    img_paths, labels = dataloader()
    print("num of img:", len(img_paths), "num of labels:", len(labels))
    print("-" * 25)
    train_data, val_data = custom_split(img_paths, labels)
    print("训练集和测试集分割完毕")
    print("-" * 25)
    composed_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(cfg.input_size),
        # transforms.RandomCrop(input_size),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_transforms = {
        'train': composed_transform,
        'val': composed_transform,
    }
    print("__" * 55)
    print("Initializing Datasets and Dataloaders...")
    image_datasets = {
        'train': GarbageDataSet(data=train_data, transform=data_transforms['train']),
        'val': GarbageDataSet(data=val_data, transform=data_transforms['val'])
    }

    dataloaders_dict = {
        x: DataLoader(
            image_datasets[x],
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True
        ) for x in ['train', 'val']
    }
    print("Initialization finished...")
    print("__" * 55)
    print("GPU count: %d" % torch.cuda.device_count())

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)

    optimizer = optim.SGD(params_to_update, lr=cfg.learning_rate, momentum=cfg.momentum)
    criterion = nn.CrossEntropyLoss()

    model, history = train_model(
        model,
        dataloaders_dict,
        criterion,
        optimizer
    )
