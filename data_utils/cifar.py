import os.path as osp
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
from data_utils import transform as T

from data_utils.randaugment import RandomAugment
from data_utils.sampler import RandomSampler, BatchSampler
from data_utils.transform import GaussianBlur


def load_sub_data_train(inds_x, inds_u, dataset='CIFAR10', dspth='/home/liu/datasets'):
    if dataset == 'CIFAR10':
        datalist = [
            osp.join(dspth, 'cifar10', 'cifar-10-batches-py', 'data_batch_{}'.format(i + 1))
            for i in range(5)
        ]
        n_class = 10
    elif dataset == 'CIFAR100':
        datalist = [
            osp.join(dspth, 'cifar100', 'cifar-100-python', 'train')]
        n_class = 100

    data, labels = [], []
    for data_batch in datalist:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr, encoding='latin1')
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data.append(entry['data'])
            labels.append(lbs)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    data_x, label_x, data_u, label_u = [], [], [], []

    data_x += [
        data[i].reshape(3, 32, 32).transpose(1, 2, 0)
        for i in inds_x
    ]
    label_x += [labels[i] for i in inds_x]

    data_u += [
        data[i].reshape(3, 32, 32).transpose(1, 2, 0)
        for i in inds_u
    ]
    label_u += [labels[i] for i in inds_u]

    # n_labels = L // n_class
    # data_x, label_x, data_u, label_u = [], [], [], []
    # for i in range(n_class):
    #     indices = np.where(labels == i)[0]
    #     np.random.shuffle(indices)
    #     inds_x, inds_u = indices[:n_labels], indices[n_labels:]
    #     data_x += [
    #         data[i].reshape(3, 32, 32).transpose(1, 2, 0)
    #         for i in inds_x
    #     ]
    #     label_x += [labels[i] for i in inds_x]
    #     data_u += [
    #         data[i].reshape(3, 32, 32).transpose(1, 2, 0)
    #         for i in inds_u
    #     ]
    #     label_u += [labels[i] for i in inds_u]
    return data_x, label_x, data_u, label_u


def get_sub_train_loader(dataset, batch_size, ind_x, ind_u):
    data_x, label_x, data_u, label_u = load_sub_data_train(inds_x=ind_x,
                                                           inds_u=ind_u,
                                                           dataset=dataset)

    ds_x = Cifar(dataset=dataset, data=data_x, labels=label_x, is_train=True)
    ds_u = Cifar(dataset=dataset, data=data_u, labels=label_u, is_train=False)

    dl_x = DataLoader(dataset=ds_x, batch_size=batch_size, shuffle=True, num_workers=2)

    dl_u = DataLoader(dataset=ds_u, batch_size=batch_size, shuffle=False, num_workers=2)
    return dl_x, dl_u

def load_semi_loader(inds_x, inds_u, batch_size, mu, n_iters_per_epoch, dataset='CIFAR10', dspth='/home/ytx/桌面/datasets/cifar10'):
    data_x, label_x, data_u, label_u = load_sub_data_train(inds_x, inds_u, dataset=dataset)

    ds_x = Cifar(
        dataset=dataset,
        data=data_x,
        labels=label_x,
        is_train=True
    )  # return an iter of num_samples length (all indices of samples)
    sampler_x = RandomSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size)
    batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)  # yield a batch of samples one time
    dl_x = torch.utils.data.DataLoader(
        ds_x,
        batch_sampler=batch_sampler_x,
        num_workers=2,
        pin_memory=True
    )
    ds_u = Cifar(
        dataset=dataset,
        data=data_u,
        labels=label_u,
        is_train=True
    )
    sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)
    batch_sampler_u = BatchSampler(sampler_u, batch_size * mu, drop_last=True)
    dl_u = torch.utils.data.DataLoader(
        ds_u,
        batch_sampler=batch_sampler_u,
        num_workers=2,
        pin_memory=True
    )
    return dl_x, dl_u



def get_pesudo_train_loader(dataset, batch_size, ind_x, ind_u):
    data_x, label_x, data_u, label_u = load_sub_data_train(inds_x=ind_x,
                                                           inds_u=ind_u,
                                                           dataset=dataset)

    ds_x = Cifar(dataset=dataset, data=data_x, labels=label_x, is_train=True)
    ds_u = Cifar(dataset=dataset, data=data_u, labels=label_u, is_train=False)

    dl_x = DataLoader(dataset=ds_x, batch_size=batch_size, shuffle=True, num_workers=2)

    dl_u = DataLoader(dataset=ds_u, batch_size=batch_size, shuffle=False, num_workers=2)
    return dl_x, dl_u

def get_combine_train_loader(dataset, batch_size, ind_x, ind_g, lbs_g):
    if ind_g is None:
        ind_g = np.array([])
    data_x, label_x, data_g, label_g = load_sub_data_train(inds_x=ind_x,
                                                           inds_u=ind_g,
                                                           dataset=dataset)

    if lbs_g is None:
        lbs_g = np.array([])
    ds_x = Cifar(dataset=dataset, data=data_x, labels=label_x, is_train=True)
    ds_g = Cifar(dataset=dataset, data=data_x + data_g, labels=label_x + lbs_g.flatten().tolist(), is_train=True)

    dl_x = DataLoader(dataset=ds_x, batch_size=batch_size, shuffle=True, num_workers=2)
    dl_g = DataLoader(dataset=ds_g, batch_size=batch_size, shuffle=True, num_workers=2)
    return dl_x, dl_g

def get_guess_train_loader(dataset, batch_size, ind_x, ind_g, lbs_g):
    if ind_g is None:
        ind_g = np.array([])
    data_x, label_x, data_g, label_g = load_sub_data_train(inds_x=ind_x,
                                                           inds_u=ind_g,
                                                           dataset=dataset)

    if lbs_g is None:
        lbs_g = np.array([])
    ds_x = Cifar(dataset=dataset, data=data_x, labels=label_x, is_train=True)
    ds_g = Cifar(dataset=dataset, data=data_g, labels=lbs_g.flatten().tolist(), is_train=True)

    dl_x = DataLoader(dataset=ds_x, batch_size=batch_size, shuffle=True, num_workers=2)
    dl_g = DataLoader(dataset=ds_g, batch_size=batch_size, shuffle=True, num_workers=2)
    return dl_x, dl_g




class Cifar(Dataset):
    def __init__(self, dataset, data, labels, is_train=True):
        super(Cifar, self).__init__()
        self.data, self.labels = data, labels
        self.is_train = is_train
        assert len(self.data) == len(self.labels)
        if dataset == 'CIFAR10':
            mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        elif dataset == 'CIFAR100':
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

        if is_train:
            self.trans_weak = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
            self.trans_strong = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
        else:
            self.trans = T.Compose([
                T.Resize((32, 32)),
                T.Normalize(mean, std),
                # GaussianBlur(kernel_size=int(13)),
                T.ToTensor(),
            ])

    def __getitem__(self, idx):
        im, lb = self.data[idx], self.labels[idx]
        if self.is_train:
            return self.trans_weak(im), self.trans_strong(im), lb
        else:
            return self.trans(im), lb

    def __len__(self):
        leng = len(self.data)
        return leng

def load_data_val(dataset, dspth='/home/liu/datasets'):
    if dataset == 'CIFAR10':
        datalist = [
            osp.join(dspth, 'cifar10', 'cifar-10-batches-py', 'test_batch')
        ]
    elif dataset == 'CIFAR100':
        datalist = [
            osp.join(dspth, 'cifar100', 'cifar-100-python', 'test')
        ]

    data, labels = [], []
    for data_batch in datalist:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr, encoding='latin1')
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data.append(entry['data'])
            labels.append(lbs)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    data = [
        el.reshape(3, 32, 32).transpose(1, 2, 0)
        for el in data
    ]
    return data, labels

def get_val_loader(dataset, batch_size, num_workers, pin_memory=True):
    data, labels = load_data_val(dataset)
    ds = Cifar(
        dataset=dataset,
        data=data,
        labels=labels,
        is_train=False
    )
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dl