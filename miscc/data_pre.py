#!/usr/bin/env python3

# -*- encoding: utf-8 -*-

"""
@Author  :   Collapsar-G

@License :   (C) Copyright 2021-*

@Contact :   gjf840513468@gmail.com

@File    :   $data_pre.py

@Time    :   $2021.5.6 $16:50

@Desc    :   数据集预处理

"""

import os
from shutil import copyfile
from miscc.classes import class2id


def traverseFile(root):
    """
    return list with all files in root
    """
    file_list = []
    # file_name = []
    for f in os.listdir(root):
        f_path = root + '/' + f
        # f_name = f
        if os.path.isfile(f_path):
            file_list.append(f)
            # file_name.append(f_name)
        else:
            file_list_2, file_name_2 = traverseFile(f_path)
            # file_name += file_name_2
            # file_list += file_list_2
    return file_list


if __name__ == "__main__":
    source_data_path = "../pic_with_all"
    destination_data_path = "../data/garbage_classify"
    num = 20000
    for f in os.listdir(source_data_path):
        file_list = traverseFile(source_data_path + "/" + f)
        # print(file_list)
        print("=" * 55)
        for photo in file_list:
            try:
                copyfile(source_data_path + "/" + f + "/" + photo,
                         destination_data_path + "/img_" + str(num).zfill(5) + ".jpg")
                with open(destination_data_path + "/img_" + str(num).zfill(5) + ".txt", "a",
                          encoding="utf-8") as file:
                    file.write("/img_" + str(num).zfill(5) + ", " + class2id[f])
                with open("../data/new_shu_label.txt", "a", encoding="utf-8") as file:
                    file.write("/data/garbage_classify/img_" + str(num).zfill(5) + ".jpg," + class2id[f] + "\n")
                num += 1

            except:
                print(source_data_path + "/" + f + "/" + photo, "文件复制错误")
