"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $classes.py

@Time    :   $2021.5.6 $18:40

@Desc    :   永远的工具人
"""
from glob import glob
import random
import os
import math

from miscc.config import cfg


def dataloader():
    label_files = glob(cfg.data_path + "new_shu_label.txt")
    random.shuffle(label_files)
    img_paths = []
    labels = []
    for _, file_path in enumerate(label_files):
        for line in open(file_path):
            line_split = line.strip().split(',')
            if len(line_split) != 2:
                print('%s contain error label' % os.path.basename(file_path), line_split)
                continue
            img_name = line_split[0]
            label = int(line_split[1])
            img_paths.append(os.path.join(".", img_name))
            labels.append(label)

    return img_paths, labels


def custom_split(img_paths, labels):
    label_dict = dict()
    for i, l in enumerate(labels):
        str_l = str(l)
        l_count = label_dict.get(str_l)
        if l_count is None:
            l_count = []
        l_count.append(i)
        label_dict[str_l] = l_count

    test_labels = []
    train_labels = []
    test_paths = []
    train_paths = []
    for val in label_dict.values():
        len_val = len(val)
        rand_index = random.choices(
            range(len_val), k=math.floor(len_val * cfg.test_percent))
        for i, label_i in enumerate(val):
            if i in rand_index:
                test_paths.append(img_paths[label_i])
                test_labels.append(labels[label_i])
            else:
                train_paths.append(img_paths[label_i])
                train_labels.append(labels[label_i])
    print('train data count: %d, test data count: %d' % (len(train_paths), len(test_paths)))
    results1 = []
    results2 = []
    for i in range(len(train_paths)):
        results1.append((train_paths[i], train_labels[i]))
    for i in range(len(test_paths)):
        results2.append((test_paths[i], test_labels[i]))
    return results1, results2


if __name__ == "__main__":
    img_paths, labels = dataloader()
    custom_split(img_paths, labels)
