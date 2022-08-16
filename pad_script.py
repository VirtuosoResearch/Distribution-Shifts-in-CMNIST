from torchvision import datasets
import torch
import torch.nn.functional as F
import torchvision.datasets.utils as dataset_utils
import torchvision.transforms as T
import torch.nn as nn
from torch.utils.data import TensorDataset
from tqdm import trange
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
mnist_train = (mnist.data[:50000], mnist.targets[:50000])
mnist_val = (mnist.data[50000:], mnist.targets[50000:])

rng_state = np.random.get_state()
np.random.shuffle(mnist_train[0].numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist_train[1].numpy())


def make_environment(images, labels, e, lab):
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()

    def torch_xor(a, b):
        return (a - b).abs()  # Assumes both inputs are either 0 or 1

    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

    x = images.float().div_(255.0)

    if lab == 0:
        labels = 0 * labels
    elif lab == 1:
        labels = labels / labels

    y = labels.view(-1).long()

    return TensorDataset(x, y)


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


hparams = {'data_augmentation': True,
           'resnet_18': False,
           'resnet_dropout': 0.0,
           'class_balanced': False,
           'nonlinear_classifier': False,
           'lr': 0.1,
           'weight_decay': 0.0,
           'batch_size': 64,
           }


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
            criterion = F.cross_entropy

            for ii in range(len(labels)):
                if pred[ii] == labels[ii]:
                    num_correct += 1

            for jj in range(len(labels)):
                test_loss += criterion(output[jj], labels[jj])

    test_loss /= len(test_loader.dataset)
    return test_loss, num_correct / len(test_loader.dataset)


def domain_sweep(train_data, test_data, batch_size, E):
    domains = [
        make_environment(train_data[0][::2], train_data[1][::2], E[0], lab=0),
        make_environment(train_data[0][1::2], train_data[1][1::2], E[1], lab=1),
    ]
    domain_tr = domains[0]
    torch.manual_seed(0)
    indices = torch.randperm(len(domain_tr)).tolist()
    domain_tr_0 = torch.utils.data.Subset(domain_tr, indices[:len(indices) // 2])
    domain_tr_1 = torch.utils.data.Subset(domain_tr, indices[len(indices) // 2:])

    domain_ts = domains[1]
    indices = torch.randperm(len(domain_ts)).tolist()
    domain_ts_0 = torch.utils.data.Subset(domain_ts, indices[:len(indices) // 2])
    domain_ts_1 = torch.utils.data.Subset(domain_ts, indices[len(indices) // 2:])

    ds_0 = torch.utils.data.ConcatDataset([domain_tr_0, domain_ts_0])
    ds_1 = torch.utils.data.ConcatDataset([domain_tr_1, domain_ts_1])

    train_dl = DataLoader(ds_0, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    test_dl = DataLoader(ds_1, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)

    return train_dl, test_dl


max_epoch = 10
E_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for e1 in E_range:
    for e2 in E_range:

        train_dl, test_dl = domain_sweep(mnist_train, mnist_val, E=[e1, e2], batch_size=64)

        v = []

        for i in range(5):
            for epoch in range(max_epoch):
                algorithm = ERM(input_shape=[2, 14, 14], num_classes=2, num_domains=2, hparams=hparams)
                algorithm.to(device)
                train_loss = train_one_epoch(train_dl, algorithm, device)
                test_loss, test_acc = test_one_epoch(test_dl, algorithm, device)
                # print('Epoch: {} \tTraining Loss: {:.4f} \tTest Loss: {:.4f} \tTest Acc: {:.4f}'.format(epoch, train_loss, test_loss, test_acc))
            # print(test_acc)
            v.append(test_acc)
        print(e1, e2, np.mean(v), np.std(v))
        # print(np.mean(v))
        # print(np.std(v))