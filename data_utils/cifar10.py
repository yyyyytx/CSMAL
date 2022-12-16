from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import cv2
import numpy as np
from data_utils.transform import GaussianBlur


transform_train = transforms.Compose([

    transforms.RandomCrop(size=32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # GaussianBlur(kernel_size=int(13)),

])


def get_train_dataset():
    train_datasets = CIFAR10(root='/home/liu/datasets/cifar10', train=True, transform=transform_train)
    return train_datasets

def get_unlabel_dataset():
    train_datasets = CIFAR10(root='/home/liu/datasets/cifar10', train=True, transform=transform_test)
    return train_datasets

def get_test_dataset():
    test_datasets = CIFAR10(root='/home/liu/datasets/cifar10', train=False, transform=transform_test)
    return test_datasets

def get_dataloader(dataset, batch_size, shuffle=True):
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=2)
    return dataloader