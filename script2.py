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


def make_environment(images, labels, color_prob, label_prob=0.25, n_classes=2):
    """Build an environment where the label is spuriously correlated with
    a specific "color" channel w.p. `color_prob`.
    The label is also corrupted w.p. `label_prob`, such that
    "color" is more correlated to the true label during training.
    `n_classes` determines how many label classes are used.
        - one color channel per class is created.
        - label corruption shifts label "to the right":
            0 to 1, 1 to 2, ..., and 9 to 0.
    """

    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()

    def collapse_labels(labels, n_classes):
        """Collapse 10 classes into n_classes classes."""
        assert n_classes in [2, 3, 5, 10]
        bin_width = 10 // n_classes
        return (labels / bin_width).clamp(max=n_classes - 1)

    def corrupt(labels, n_classes, prob):
        """Corrupt a fraction of labels by shifting it +1 (mod n_classes),
        according to bernoulli(prob).
        Generalizes torch_xor's role of label flipping for the binary case.
        """
        is_corrupt = torch_bernoulli(prob, len(labels)).bool()
        return torch.where(is_corrupt, (labels + 1) % n_classes, labels)

    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a label based on the digit
    labels = collapse_labels(labels, n_classes).float()
    # *Corrupt* label with probability 0.25 (default)
    labels = corrupt(labels, n_classes, label_prob)
    # Assign a color based on the label; flip the color with probability e
    colors = corrupt(labels, n_classes, color_prob)
    # Apply the color to the image by only giving image in the assigned color channel
    n, h, w = images.size()
    colored_images = torch.zeros((n, n_classes, h, w)).to(images)
    colored_images[torch.tensor(range(n)), colors.long(), :, :] = images

    images = (colored_images.float() / 255.)
    labels = labels.long()

    return TensorDataset(images, labels)


train_env = make_environment(mnist_train[0], mnist_train[1], 0.1, 0.25)
test_env = make_environment(mnist_val[0], mnist_val[1], 0.1, 0.25)

domain_tr = train_env
# domain_tr = domains[0]
domain_ts = test_env
batch_size = 100
train_dl = DataLoader(domain_tr, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
test_dl = DataLoader(domain_ts, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)


# To visualize the dataset

# def show_images(images, labels, nmax=64):
#     # fig, ax = plt.subplots(figsize=(8, 8))
#     # ax.set_xticks([]); ax.set_yticks([])
#     # ax.imshow(make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
#     # plt.savefig(path+'/test.png')
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


# algorithm = ERM(input_shape= domain_ts[0][0].size(), num_classes=2, num_domains=2, hparams=hparams)
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


def train_one_epoch(train_loader, algorithm, device):
    for i, (img, labels) in enumerate(train_loader):
        img, labels = img.to(device), labels.to(device)
        steps_val = algorithm.update([(img, labels)])

    return steps_val['loss']


def test_one_epoch(test_loader, algorithm, device):
    test_loss = 0
    num_correct = 0
    with torch.no_grad():
        for i, (img, labels) in enumerate(test_loader):
            img, labels = img.to(device), labels.to(device)

            output = algorithm.predict(img)
            pred = torch.argmax(output, dim=1)
            criterion = torch.nn.NLLLoss()

            for ii in range(len(labels)):
                if pred[ii] == labels[ii]:
                    num_correct += 1

            for jj in range(len(labels)):
                test_loss += criterion(output[jj], labels[jj])

    test_loss /= len(test_loader.dataset)
    return test_loss, num_correct / len(test_loader.dataset)


max_epoch = 10


# for epoch in range(max_epoch):
#     train_loss = train_one_epoch(train_dl, algorithm, device)
#     test_loss, test_acc = test_one_epoch(test_dl, algorithm, device)
#     print('Epoch: {} \tTraining Loss: {:.4f} \tTest Loss: {:.4f} \tTest Acc: {:.4f}'.format(epoch, train_loss, test_loss, test_acc))


def run_one_epoch(train_loader, test_loader, algorithm, device):
    for index, (value1, value2) in enumerate(zip(train_loader, test_loader)):
        img_tr, labels_tr = value1
        img_tr, labels_tr = img_tr.to(device), labels_tr.to(device)
        img_ts, labels_ts = value2
        img_ts, labels_ts = img_ts.to(device), labels_ts.to(device)
        steps_val = algorithm.update([(img_tr, labels_tr)], [(img_ts, labels_ts)])

    loss = steps_val['loss']

    return loss


def domain_sweep(train_data, test_data, batch_size, E):
    domains = [
        make_environment(train_data[0][::2], train_data[1][::2], E[0]),
        make_environment(train_data[0][1::2], train_data[1][1::2], E[1]),
        make_environment(test_data[0], test_data[1], E[2])
    ]
    domain_tr = torch.utils.data.ConcatDataset([domains[0], domains[1]])
    domain_ts = domains[2]
    train_dl = DataLoader(domain_tr, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    test_dl = DataLoader(domain_ts, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)

    return train_dl, test_dl


max_epochs = 10
e = []
l = []
l2 = []
env = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for e1 in env:
    for e2 in env:
        for e3 in env:
            avg_shift = []
            for run in range(5):
                algorithm = Covariate_Shift(input_shape=domain_ts[0][0].size(), num_classes=2, num_domains=2,
                                            hparams=hparams)
                algorithm.to(device)
                train_dl, test_dl = domain_sweep(mnist_train, mnist_val, E=[e1, e2, e3], batch_size=100)
                for epoch in range(max_epochs):
                    shift = run_one_epoch(train_dl, test_dl, algorithm, device)
                avg_shift.append(shift)
            print('Shift for train e = ', e1, e2, ' and test e = ', e3, ': ', format(np.mean(avg_shift), '.3E'))
