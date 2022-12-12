import json
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Tuple, Union, Optional, Sequence

from scipy.stats import gaussian_kde
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.optim import SGD, swa_utils, lr_scheduler
import torchvision
import PIL
import random

from networksood import EnvClassifier, get_backbone

from torchvision import datasets
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
import sys
from datetime import datetime

import sklearn
from sklearn.preprocessing import StandardScaler

from zipfile import ZipFile
import argparse
import tarfile
import gdown
from torchvision import transforms
from torchvision.datasets import ImageFolder

pwd = os.getcwd()


# current_time = str(datetime.now().strftime("%m_%d-%I_%M_%S_%p"))
# path = os.path.join(pwd, "/temp")

# if not os.path.exists(path):
#     os.mkdir(path)

# print(path)
# quit()
# class Logger(object):
#     def __init__(self):
#         self.terminal = sys.stdout
#         self.log = open(path + "/logfile.log", "a")
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass
#
#
# sys.stdout = Logger()


# Build environments

def make_environment(images, labels, e):
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()

    def torch_xor(a, b):
        return (a - b).abs()  # Assumes both inputs are either 0 or 1

    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

    x = images.float().div_(255.0)
    y = labels.view(-1).long()

    return TensorDataset(x, y)



def PACS_datasets():
    root = os.path.join(pwd, "data/PACS")

    environments = [f.name for f in os.scandir(root) if f.is_dir()]
    environments = sorted(environments)

    augment = False

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    augment_transform = transforms.Compose([
        # transforms.Resize((224,224)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    datasets = []
    for i, environment in enumerate(environments):

        if augment and (i not in test_envs):
            env_transform = augment_transform
        else:
            env_transform = transform

        path = os.path.join(root, environment)
        env_dataset = ImageFolder(path,
            transform=env_transform)

        datasets.append(env_dataset)

    # input_shape = (3, 224, 224,)
    # num_classes = len(datasets[-1].classes)
    
    return datasets

# Making DataLoaders
# batch_size = 32
# num_workers = 4

# dataloaders = []
# for i, dataset in enumerate(datasets):
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         drop_last=True
#     )
#     dataloaders.append(dataloader)

# print(datasets[0][250][1])

# class PACS(MultipleEnvironmentImageFolder):

#     ENVIRONMENTS = ["A", "C", "P", "S"]
#     def __init__(self, root, test_envs, hparams):
#         self.dir = os.path.join(root, "PACS/")
#         super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)
        