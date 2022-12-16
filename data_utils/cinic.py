from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import cv2
import numpy as np


cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]

transform_train = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=cinic_mean, std=cinic_std),
])



transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=cinic_mean, std=cinic_std),
])




def get_train_dataset():
    train_datasets = ImageFolder('./datasets/cinic10/train', transform=transform_train)
    return train_datasets

def get_unlabel_dataset():
    train_datasets = ImageFolder('./datasets/cinic10/train', transform=transform_test)
    return train_datasets

def get_test_dataset():
    test_datasets = ImageFolder('./datasets/cinic10/test', transform=transform_test)
    return test_datasets

def get_dataloader(dataset, batch_size, shuffle=True):
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=2)
    return dataloader


if __name__ == '__main__':
    ds_train = get_train_dataset()
    print(len(ds_train))
    ds_test = get_test_dataset()
    print(len(ds_test))