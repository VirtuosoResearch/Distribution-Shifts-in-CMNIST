from cProfile import label
from turtle import color
from torchvision import datasets
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
import torchvision.datasets.utils as dataset_utils
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
# from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import os
import torch.nn.init as init
from tqdm import trange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
mnist_train = (mnist.data[:50000], mnist.targets[:50000])
mnist_val = (mnist.data[50000:], mnist.targets[50000:])


def color_dataset(images, labels, e):
    # # Subsample 2x for computational convenience
    # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit
    original_labels = labels
    labels = (labels < 5).float()
    # Flip label with probability 0.25
    labels = torch_xor_(labels,
                        torch_bernoulli_(0.25, len(labels)))

    # Assign a color (red) based on the label; flip the color with probability e
    colors = torch_xor_(labels,
                        torch_bernoulli_(e,
                                         len(labels)))
    images = torch.stack([images, images, images], dim=1)
    # Apply the color to the image by zeroing out the other color channels

    images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
    images[torch.tensor(range(len(images))), 2, :, :] *= 0

    # images = torch.stack([images], dim=1)
    # images = torch.stack([images, images], dim=1)
    # images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

    x = images.float().div_(255.0)
    y = labels.view(-1).long()

    return TensorDataset(x, original_labels)


def torch_bernoulli_(p, size):
    return (torch.rand(size) < p).float()


def torch_xor_(a, b):
    return (a - b).abs()


domains = [
    color_dataset(mnist_train[0][::2], mnist_train[1][::2], 0.1),
    color_dataset(mnist_train[0][1::2], mnist_train[1][1::2], 0.2),
    color_dataset(mnist_val[0], mnist_val[1], 0.9)
]

domain_tr = torch.utils.data.ConcatDataset([domains[0], domains[1]])
# domain_tr = domains[0]
domain_ts = domains[2]
batch_size = 100
train_dl = DataLoader(domain_tr, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
test_dl = DataLoader(domain_ts, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)


def domain_sweep(train_data, test_data, batch_size, E):
    domains = [
        color_dataset(train_data[0][::2], train_data[1][::2], E[0]),
        color_dataset(train_data[0][1::2], train_data[1][1::2], E[1]),
        color_dataset(test_data[0], test_data[1], E[2])
    ]
    domain_tr = torch.utils.data.ConcatDataset([domains[0], domains[1]])
    domain_ts = domains[2]
    train_dl = DataLoader(domain_tr, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    test_dl = DataLoader(domain_ts, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)

    return train_dl, test_dl


# To visualize the dataset
# def show_images(images, labels, nmax=64):
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_xticks([]); ax.set_yticks([])
#     ax.imshow(make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
#     plt.savefig(path+'/test.png')
#     print(labels)

# def show_batch(dl, nmax=64):
#     for images, labels in dl:
#         show_images(images, labels, nmax)
#         break

# show_batch(train_dl)


# print(domain_ts[0][0].size())
# quit()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


def Featurizer(input_shape, hparams):
    return MNIST_CNN(input_shape)


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = Featurizer(input_shape, self.hparams)
        self.classifier = Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class Covariate_Shift(Algorithm):
    """
    Discrepency Distance
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Covariate_Shift, self).__init__(input_shape, num_classes, num_domains,
                                              hparams)
        self.featurizer = Featurizer(input_shape, self.hparams)
        self.classifier = Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network1 = nn.Sequential(self.featurizer, self.classifier)
        self.network2 = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            (list(self.network1.parameters()) + list(self.network2.parameters())),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches_tr, minibatches_ts, unlabeled=None):
        all_xtr = torch.cat([x for x, y in minibatches_tr])
        all_xts = torch.cat([x for x, y in minibatches_ts])
        loss_tr = F.cross_entropy(self.predict1(all_xtr), self.predict2(all_xtr))
        loss_ts = F.cross_entropy(self.predict1(all_xts), self.predict2(all_xts))
        loss = abs(loss_tr - loss_ts)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict1(self, x):
        return self.network1(x)

    def predict2(self, x):
        return self.network2(x)


hparams = {'data_augmentation': True,
           'resnet_18': False,
           'resnet_dropout': 0.0,
           'class_balanced': False,
           'nonlinear_classifier': False,
           'lr': 0.001,
           'weight_decay': 0.0,
           'batch_size': 100,
           }


# algorithm = ERM(input_shape= [3, 28, 28], num_classes=10, num_domains=2, hparams=hparams)
# algorithm.to(device)


def run_one_epoch(train_loader, test_loader, algorithm, device):
    for index, (value1, value2) in enumerate(zip(train_loader, test_loader)):
        img_tr, labels_tr = value1
        img_tr, labels_tr = img_tr.to(device), labels_tr.to(device)
        img_ts, labels_ts = value2
        img_ts, labels_ts = img_ts.to(device), labels_ts.to(device)
        steps_val = algorithm.update([(img_tr, labels_tr)], [(img_ts, labels_ts)])

    loss = steps_val['loss']

    return loss


max_epochs = 3
e = []
l = []
l2 = []
env = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for e1 in env:
    for e2 in env:
        for e3 in env:
            avg_shift = []
            for run in range(3):
                algorithm = Covariate_Shift(input_shape=[3, 28, 28], num_classes=10, num_domains=2, hparams=hparams)
                algorithm.to(device)
                train_dl, test_dl = domain_sweep(mnist_train, mnist_val, E=[e1, e2, e3], batch_size=100)
                for epoch in range(max_epochs):
                    shift = run_one_epoch(train_dl, test_dl, algorithm, device)
                avg_shift.append(shift)
            print('Shift for train e = ', e1, e2, ' and test e = ', e3, ': ', np.mean(avg_shift))
