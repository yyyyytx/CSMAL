from torchvision import transforms
from PIL import Image
import os
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
import numpy as np
import torchvision.datasets as datasets
import torch
from imagenet.folder2lmdb import ImageFolderLMDB




normalize = transforms.Normalize(mean=[0.482, 0.458, 0.408],
                                 std=[0.269, 0.261, 0.276])

train_trans = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
#     transforms.Compose([
#
#         transforms.Resize((128, 128)),
#         transforms.RandomCrop(128, padding=4),
#         transforms.RandomHorizontalFlip(),
#             # transforms.RandomVerticalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
#         )
# ])
test_trans = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    normalize
])


def get_train_dataset():
    # return datasets.ImageFolder(
    #     '/home/liu/桌面/datasets/imagenet64/train',
    #     train_trans)
    return ImageFolderLMDB('/home/liu/桌面/datasets/imagenet64/train.lmdb',
        train_trans)

def get_test_dataset():
    # return datasets.ImageFolder(
    #     '/home/liu/桌面/datasets/imagenet64/val',
    #     test_trans)
    return ImageFolderLMDB('/home/liu/桌面/datasets/imagenet64/val.lmdb',
            test_trans)

def get_unlabel_dataset():
    # return datasets.ImageFolder(
    #     '/home/liu/桌面/datasets/imagenet64/train',
    #     test_trans)
    return ImageFolderLMDB('/home/liu/桌面/datasets/imagenet64/train.lmdb',
                           test_trans)

def get_dataloader(train_datasets, train_batch_size, shuffle=True):
    train_dataloader = DataLoader(dataset=train_datasets,
                                  batch_size=train_batch_size,
                                  shuffle=shuffle,
                                  num_workers=8,
                                  pin_memory=True)
    return train_dataloader

if __name__ == '__main__':
    train_ds = get_train_dataset()
    print(len(train_ds))
    test_ds = get_test_dataset()
    print(len(test_ds))