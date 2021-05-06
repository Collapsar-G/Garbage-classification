import torch
# import torch.nn as nn

import numpy as np

from miscc.config import cfg
from miscc.classes import classes as classes
import cv2 as cv


def load_model():
    resnet = torch.load(cfg.test_model_path, map_location=torch.device('cpu'))
    return resnet


def test_result(path):
    model = load_model()
    src = cv.imread(path)
    image = cv.resize(src, (224, 224))
    image = np.float32(image) / 255.0
    image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    image = image.transpose((2, 0, 1))
    input_x = torch.from_numpy(image).unsqueeze(0)
    # print(input_x.size())
    pred = model(input_x)
    pred_index = torch.argmax(pred, 1).detach().numpy()
    print(pred_index)
    print("current predict class name : %s" % classes[str(pred_index[0])])

    return classes[str(pred_index[0])]


if __name__ == '__main__':
    path = './data/6.jpg'
    result = test_result(path)
