from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms
import cv2




transform_train = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomCrop(size=32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])



transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])



def get_train_dataset():
    train_datasets = CIFAR100(root='/home/liu/datasets/cifar100', train=True, transform=transform_train)
    return train_datasets

def get_unlabel_dataset():
    train_datasets = CIFAR100(root='/home/liu/datasets/cifar100', train=True, transform=transform_test)
    return train_datasets

def get_test_dataset():
    test_datasets = CIFAR100(root='/home/liu/datasets/cifar100', train=False, transform=transform_test)
    return test_datasets