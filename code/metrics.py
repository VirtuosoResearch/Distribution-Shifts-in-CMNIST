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

from dataset import make_environment

import itertools

# pwd = os.getcwd()


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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
mnist_train = (mnist.data[:50000], mnist.targets[:50000])
mnist_val = (mnist.data[50000:], mnist.targets[50000:])

# rng_state = np.random.get_state()
# np.random.shuffle(mnist_train[0].numpy())
# np.random.set_state(rng_state)
# np.random.shuffle(mnist_train[1].numpy())
torch.manual_seed(0)
np.random.seed(0)


# def domainsweep(E):
#     domains = [
#         make_environment(mnist_train[0][::2], mnist_train[1][::2], E[0]),
#         make_environment(mnist_train[0][1::2], mnist_train[1][1::2], E[1]),
#         make_environment(mnist_val[0], mnist_val[1], E[2]),
#     ]

#     batch_size = 100

#     ds_tr = torch.utils.data.ConcatDataset([domains[0], domains[1]])

#     train_dl = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
#     test_dl = DataLoader(domains[2], batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)

#     return train_dl, test_dl


class ExponentialMovingAverage:
    def __init__(self, update_ratio: float, init_val: Optional[float] = None) -> None:
        if not (0 < update_ratio < 1):
            raise ValueError
        self._update_ratio = update_ratio
        self.ema = init_val

    def update(self, value: float) -> float:
        if self.ema is None:
            self.ema = value
        else:
            self.ema *= (1 - self._update_ratio)
            self.ema += self._update_ratio * value


def evaluate(model: nn.Module, zipped_minibatches: Iterable, device: str) -> float:
    model.eval()

    n_examples = 0
    n_correct_preds = 0
    for i, minibatches in enumerate(zipped_minibatches):
        x, y = map(torch.cat, zip(*minibatches))
        e = torch.cat([torch.zeros(x.size(0) // 2, dtype=torch.long),
                       torch.ones(x.size(0) // 2, dtype=torch.long)])
        x = x.to(device)
        y = F.one_hot(y, num_classes=model.class_dim).float().to(device)
        e = e.to(device)

        with torch.no_grad():
            z, logits = model(x, y)

        _, pred = torch.max(logits.data, 1)
        n_correct_preds += (pred == e).long().sum().item()
        n_examples += x.size(0)

    model.train()
    return n_correct_preds / n_examples


def update_swa_bn(model: nn.Module, zipped_minibatches: Iterable, device: str) -> None:
    momenta = {}
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    with torch.no_grad():
        for i, minibatches in enumerate(zipped_minibatches):
            x, y = map(torch.cat, zip(*minibatches))
            x = x.to(device)
            y = F.one_hot(y, num_classes=model.class_dim).float().to(device)
            model(x, y)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


# printing pairs p,q from 0.1 to 0.9 with step 0.1 without repating pairs like 0.1 0.2 and 0.2 0.1
domains = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# selecting all sets of 4 values from domains with replacement and without repetition
E1 = list(itertools.combinations_with_replacement(domains, 2))
E2 = list(itertools.combinations_with_replacement(domains, 2))

# creating a list of E1 and E2
E = list(itertools.product(E1, E2))
print(len(E))

with open("metrics_output_mnist_double.txt","r") as f:
    # reading the last line of the file
    last_line = f.readlines()[-1]
    # getting the last value of p and q from the last line
    last_p1 = float(last_line.split()[0])
    last_p2 = float(last_line.split()[1])
    last_q1 = float(last_line.split()[2])
    last_q2 = float(last_line.split()[3])
    print(last_p1, last_p2, last_q1, last_q2)


flag = 0

for ((p_value1, p_value2), (q_value1, q_value2)) in E:
    
    print("p_value1: ", p_value1, "p_value2: ", p_value2, "q_value1: ", q_value1, "q_value2: ", q_value2)
    
    if last_p1 > p_value1:
        pass
    elif last_p1 == p_value1:
        if last_p2 > p_value2:
            pass
        elif last_p2 == p_value2:
            if last_q1 > q_value1:
                pass
            elif last_q1 == q_value1:
                if last_q2 > q_value2:
                    pass
                elif last_q2 == q_value2:
                    pass
                else:
                    flag = 1
            else:
                flag = 1
        else:
            flag = 1
    else:
        flag = 1
        
    
            
    if flag == 1:
    
        in_p1 = make_environment(mnist_train[0][::4], mnist_train[1][::4], p_value1)
        in_p2 = make_environment(mnist_train[0][2::4], mnist_train[1][2::4], p_value2)
        in_q1 = make_environment(mnist_train[0][1::4], mnist_train[1][1::4], q_value1)
        in_q2 = make_environment(mnist_train[0][3::4], mnist_train[1][3::4], q_value2)
        out_p1 = make_environment(mnist_val[0][::4], mnist_val[1][::4], p_value1)
        out_p2 = make_environment(mnist_val[0][2::4], mnist_val[1][2::4], p_value2)
        out_q1 = make_environment(mnist_val[0][1::4], mnist_val[1][1::4], q_value1)
        out_q2 = make_environment(mnist_val[0][3::4], mnist_val[1][3::4], q_value2)
        
        
        batch_size = 100
        
        in_p = torch.utils.data.ConcatDataset([in_p1, in_p2])
        in_q = torch.utils.data.ConcatDataset([in_q1, in_q2])
        out_p = torch.utils.data.ConcatDataset([out_p1, out_p2])
        out_q = torch.utils.data.ConcatDataset([out_q1, out_q2])

        train_dl_in = DataLoader(in_p, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
        train_dl_out = DataLoader(out_p, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
        test_dl_in = DataLoader(in_q, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
        test_dl_out = DataLoader(out_q, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

        in_dataloader_p, out_dataloader_p, in_dataloader_q, out_dataloader_q = train_dl_in, train_dl_out, test_dl_in, test_dl_out

        backbone = get_backbone('mlp', (2, 14, 14),
                                pretrained_model_path=None)
        model = EnvClassifier(backbone, 2, 8, 2)
        model = model.to(device)

        best_dict = {'acc': 0, 'step': -1, 'state': None}
        val_acc = evaluate(model, zip(out_dataloader_p, out_dataloader_q), device)
        print(f'validation accuracy before training: {val_acc:.4f}')

        optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9,
                        weight_decay=1e-4)
        swa_model = swa_utils.AveragedModel(model)
        if not hasattr(swa_model, 'class_dim'):
            setattr(swa_model, 'class_dim', model.class_dim)
        else:
            raise Exception
        T_max = len(in_dataloader_p) * 5
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max,
                                                eta_min=0.025)

        n_epochs = 10

        training_loss = ExponentialMovingAverage(0.1)
        training_acc = ExponentialMovingAverage(0.1)
        for epoch in range(1, n_epochs + 1):
            epoch_start_time = time.time()
            for i, minibatches in enumerate(zip(in_dataloader_p, in_dataloader_q)):
                model.train()
                x, y = map(torch.cat, zip(*minibatches))
                e = torch.cat([torch.zeros(batch_size, dtype=torch.long),
                            torch.ones(batch_size, dtype=torch.long)])
                x = x.to(device)
                y = F.one_hot(y, num_classes=model.class_dim).float().to(device)
                e = e.to(device)

                z, logits = model(x, y)
                loss = F.cross_entropy(logits, e)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch <= 5:
                    scheduler.step()
                else:
                    swa_model.update_parameters(model)

                _, pred = torch.max(logits.data, 1)
                correct = (pred == e)
                minibatch_acc = correct.float().sum().item() / correct.size(0)
                training_loss.update(loss.item())
                training_acc.update(minibatch_acc)
            epoch_time_spent = int(time.time() - epoch_start_time)
            epoch_time_spent = timedelta(seconds=epoch_time_spent)

            n_epochs_per_ckpt = 1

            if epoch % n_epochs_per_ckpt == 0:
                print(f'epoch {epoch} / {n_epochs} finished in {epoch_time_spent}')
                print(f'training loss (ema): {training_loss.ema:.4f}')
                print(f'     accuracy (ema): {training_acc.ema:.4f}')
                val_acc = evaluate(model, zip(out_dataloader_p, out_dataloader_q), device)
                print(f'validation accuracy: {val_acc:.4f}')
                if epoch > 5:
                    update_swa_bn(swa_model, zip(in_dataloader_p, in_dataloader_q), device)
                    val_acc = evaluate(swa_model, zip(out_dataloader_p, out_dataloader_q),
                                    device)
                    print(f'validation accuracy: {val_acc:.4f} (swa)')

        state_dict = swa_model.state_dict()
        del state_dict['n_averaged']
        for name in list(state_dict.keys()):
            if name.startswith('module'):
                state_dict[name[7:]] = state_dict[name]
                del state_dict[name]

        model = get_backbone('mlp', (2, 14, 14),
                            pretrained_model_path=None)

        model = EnvClassifier(model, 2, 8, 2)
        model.load_state_dict(state_dict)
        model = model.g
        model = model.to(device)
        model.eval()

        save_dict = {}
        for k, dataloader in enumerate([out_dataloader_p, out_dataloader_q]):
            envs_code = ('p', 'q')[k]
            print(f'extracting features from envs {envs_code}')
            y_minibatches = []
            z_minibatches = []
            for i, (x, y) in enumerate(dataloader):
                x = x.to(device)
                with torch.no_grad():
                    z = model(x)
                y_minibatches.append(y)
                z_minibatches.append(z.cpu())
            y_cat = torch.cat(y_minibatches)
            z_cat = torch.cat(z_minibatches)
            save_dict[f'y_{envs_code}'] = y_cat.numpy()
            save_dict[f'z_{envs_code}'] = z_cat.numpy()


        # print(save_dict)


        class gaussian_kde_(gaussian_kde):
            """ A gaussian kde that is friendly to small samples. """

            def _compute_covariance(self) -> None:
                """Computes the covariance matrix for each Gaussian kernel using
                covariance_factor().
                """
                self.factor = self.covariance_factor()
                # Cache covariance and inverse covariance of the data
                if not hasattr(self, '_data_inv_cov'):
                    self._data_covariance = np.atleast_2d(
                        np.cov(self.dataset, rowvar=1, bias=False, aweights=self.weights))
                    w, v = np.linalg.eigh(self._data_covariance)
                    # Set near-zero eigenvalues to a small number, avoiding singular covariance
                    # matrices when the sample do not span the whole feature space
                    w[np.where(abs(w) < 1e-9)[0]] = 0.01
                    self._data_inv_cov = np.linalg.inv(v @ np.diag(w) @ v.T)

                self.covariance = self._data_covariance * self.factor ** 2
                self.inv_cov = self._data_inv_cov / self.factor ** 2


        def compute_div(p: Sequence[float], q: Sequence[float], probs: Sequence[int],
                        eps_div: float) -> float:
            if not len(p) == len(q) == len(probs):
                raise ValueError
            div = 0
            for i in range(len(probs)):
                if p[i] < eps_div or q[i] < eps_div:
                    div += abs(p[i] - q[i]) / probs[i]
            div /= len(probs) * 2
            return div


        def compute_cor(y_p: np.ndarray, z_p: np.ndarray, y_q: np.ndarray, z_q: np.ndarray,
                        p: Sequence[float], q: Sequence[float], probs: Sequence[int],
                        points: np.ndarray, eps_cor: float, strict: bool = False) -> float:
            if not len(p) == len(q) == len(probs):
                raise ValueError
            y_p_unique, y_q_unique = map(np.unique, (y_p, y_q))
            if not np.all(y_p_unique == y_q_unique):
                raise ValueError
            classes = sorted(y_p_unique)
            n_classes = len(classes)
            sample_sizes = np.zeros(n_classes, dtype=int)
            cors = np.zeros(n_classes, dtype=float)

            for i in range(n_classes):
                y = classes[i]
                indices_p = np.where(y_p == y)[0]
                indices_q = np.where(y_q == y)[0]
                # if indices_p.shape != indices_q.shape:
                #     raise ValueError(f'Number of datapoints mismatch (y={y}): '
                #                      f'{indices_p.shape} != {indices_q.shape}')
                try:
                    kde_p = gaussian_kde_(z_p[indices_p].T)
                    kde_q = gaussian_kde_(z_q[indices_q].T)
                    p_given_y = kde_p(points)
                    q_given_y = kde_q(points)
                except (np.linalg.LinAlgError, ValueError) as exception:
                    if strict:
                        raise exception
                    print(f'WARNING: skipping y={y} because scipy.stats.gaussian_kde '
                        f'failed. This usually happens when there is too few datapoints.')
                    print(f'y={y}: #datapoints=({len(indices_p)}, {len(indices_q)}), '
                        f'skipped')
                    continue
                sample_sizes[i] = len(indices_p)

                for j in range(len(probs)):
                    if p[j] > eps_cor and q[j] > eps_cor:
                        integrand = abs(p_given_y[j] * np.sqrt(q[j] / p[j])
                                        - q_given_y[j] * np.sqrt(p[j] / q[j]))
                        cor_j = integrand / probs[j]
                    else:
                        integrand = cor_j = 0
                    cors[i] += cor_j
                cors[i] /= len(probs) * 2
                print(f'y={y}: #datapoints=({len(indices_p)}, {len(indices_q)}), '
                    f'value={cors[i]:.4f}')
            cor = np.sum(sample_sizes * cors) / np.sum(sample_sizes)
            return cor


        random.seed(44)
        np.random.seed(44)

        data = save_dict
        y_p, z_p, y_q, z_q = data['y_p'], data['z_p'], data['y_q'], data['z_q']
        print(f'features loaded: (p) {z_p.shape}, (q) {z_q.shape}')
        print(f'labels   loaded: (p) {y_p.shape}, (q) {y_q.shape}')
        if len(z_p) != len(y_p) or len(z_q) != len(y_q):
            raise RuntimeError

        z_all = np.append(z_p, z_q, 0)
        scaler = StandardScaler().fit(z_all)
        z_all, z_p, z_q = map(scaler.transform, (z_all, z_p, z_q))

        DEFAULT_SEED = 0
        DEFAULT_SAMPLE_SIZE = 10000

        print('computing KDE for importance sampling')
        sampling_pdf = gaussian_kde(z_all.T)
        points = sampling_pdf.resample(DEFAULT_SAMPLE_SIZE, seed=DEFAULT_SEED)
        probs = sampling_pdf(points)

        print('computing KDE for p and q')
        p = gaussian_kde(z_p.T)(points)
        q = gaussian_kde(z_q.T)(points)

        # print('computing diversity shift')
        # div = compute_div(p, q, probs, 1e-12)
        # if np.isnan(div) or np.isinf(div):
        #     raise RuntimeError
        # print(f'div: {div}')

        print('computing correlation shift')
        cor = compute_cor(y_p, z_p, y_q, z_q, p, q, probs, points, 5e-4,
                        strict=False)
        if np.isnan(cor) or np.isinf(cor):
            raise RuntimeError
        print(f'cor: {cor}')
        
        with open("metrics_output_mnist_double.txt","a") as file:
            file.write(f"{p_value1} {p_value2} {q_value1} {q_value2} {cor} \n")
        
    
