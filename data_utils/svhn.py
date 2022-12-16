from torchvision import transforms
from PIL import Image
import os
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
import numpy as np
from torchvision.datasets import svhn

transform_train = transforms.Compose([

    transforms.RandomCrop(size=32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])



transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def get_train_dataset():
    train_datasets = svhn.SVHN(root='/home/ytx/桌面/datasets/svhn', download=True, split='train', transform=transform_train)
    return train_datasets

def get_test_dataset():
    test_datasets = svhn.SVHN(root='/home/ytx/桌面/datasets/svhn', download=True, split='test', transform=transform_train)

    return test_datasets

def get_unlabel_dataset():
    train_datasets = svhn.SVHN(root='/home/ytx/桌面/datasets/svhn', download=True, split='train', transform=transform_test)

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