from torchvision import datasets
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
import torchvision.datasets.utils as dataset_utils
from torchvision import transforms

mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
mnist_train = (mnist.data[:50000], mnist.targets[:50000])
mnist_val = (mnist.data[50000:], mnist.targets[50000:])


# to do - add shuffle


def make_domain(images, labels, e):
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()

    def torch_xor(a, b):
        return (a - b).abs()

    # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    original_labels = labels
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    imgs = []
    for i in range(len(images)):
        im = images[i]
        color = colors[i]
        im_arr = np.array(im)
        h, w = im_arr.shape
        dtype = im_arr.dtype
        im_arr = np.reshape(im_arr, [h, w, 1])
        if int(color):
            im_arr = np.concatenate([im_arr, np.zeros((h, w, 2), dtype=dtype)], axis=2)
        else:
            im_arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype), im_arr, np.zeros((h, w, 1), dtype=dtype)],
                                    axis=2)
        imgs.append(Image.fromarray(im_arr))
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
    return {
        'images': imgs,
        'labels': original_labels[:, None],
        'binary_labels': labels[:, None],
    }


domains = [
    make_domain(mnist_train[0][::2], mnist_train[1][::2], 0.2),
    make_domain(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
    make_domain(mnist_val[0], mnist_val[1], 0.9)
]

domaintr1 = domains[0]

domaints = domains[2]


# debug
# rr = np.random.randint(low=0, high=len(domains[0]['images']), size=10)
# fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(1.5 * 10, 2 * 1))
# for i, ax in enumerate(axes):
#     ax.set_title(f"Label: {int(domains[0]['labels'][rr[i]])}")
#     ax.imshow(domains[0]['images'][rr[i]])
# plt.savefig('domain1.png')


# deprecated, left for reference
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(3 * 28 * 28, 512)
#         self.fc2 = nn.Linear(512, 512)
#         self.fc3 = nn.Linear(512, 1)
#
#     def forward(self, x):
#         x = x.view(-1, 3 * 28 * 28)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         logits = self.fc3(x).flatten()
#         return logits
#
#
# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 20, 5, 1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         self.fc1 = nn.Linear(4 * 4 * 50, 500)
#         self.fc2 = nn.Linear(500, 1)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4 * 4 * 50)
#         x = F.relu(self.fc1(x))
#         logits = self.fc2(x).flatten()
#         return logits
#
#
# def test_model(model, device, test_loader, set_name="test set"):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device).float()
#             output = model(data)
#             test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()  # sum up batch loss
#             pred = torch.where(torch.gt(output, torch.Tensor([0.0]).to(device)),
#                                torch.Tensor([1.0]).to(device),
#                                torch.Tensor([0.0]).to(device))  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#     test_loss /= len(test_loader.dataset)
#
#     print('\nPerformance on {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
#         set_name, test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
#
#     return 100. * correct / len(test_loader.dataset)
#
#
# def erm_train(model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device).float()
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.binary_cross_entropy_with_logits(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 10 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                        100. * batch_idx / len(train_loader), loss.item()))
#
#
# def train_and_test_erm():
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#
#     kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
#     all_train_loader = torch.utils.data.DataLoader(
#         ColoredMNIST(root='./data', env='all_train',
#                      transform=transforms.Compose([
#                          transforms.ToTensor(),
#                          transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
#                      ])),
#         batch_size=64, shuffle=True, **kwargs)
#
#     test_loader = torch.utils.data.DataLoader(
#         ColoredMNIST(root='./data', env='test', transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
#         ])),
#         batch_size=1000, shuffle=True, **kwargs)
#
#     model = ConvNet().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=0.01)
#
#     for epoch in range(1, 2):
#         erm_train(model, device, all_train_loader, optimizer, epoch)
#         test_model(model, device, all_train_loader, set_name='train set')
#         test_model(model, device, test_loader)
#
#
# train_and_test_erm()
