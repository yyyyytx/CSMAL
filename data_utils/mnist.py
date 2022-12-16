from torchvision import transforms
from PIL import Image
import os
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
import numpy as np
from torchvision.datasets import mnist

def get_train_dataset():
    train_datasets = mnist.MNIST(root='/home/liu/桌面/datasets/mnist', download=True,train=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]))
    return train_datasets

def get_test_dataset():
    test_datasets = mnist.MNIST(root='/home/liu/桌面/datasets/mnist', download=True,train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]))
    return test_datasets

def get_unlabel_dataset():
    train_datasets = mnist.MNIST(root='/home/liu/桌面/datasets/mnist', download=True,train=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]))
    return train_datasets

def get_dataloader(train_datasets, train_batch_size, shuffle=True):
    train_dataloader = DataLoader(dataset=train_datasets,
                                  batch_size=train_batch_size,
                                  shuffle=shuffle,
                                  num_workers=4)
    return train_dataloader


if __name__ == '__main__':
    d = get_train_dataset()
    print(len(d))
    d = get_test_dataset()
    print(len(d))