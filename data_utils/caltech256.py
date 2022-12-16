from torchvision import transforms
from PIL import Image
import os
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
import numpy as np


from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import numpy as np



transform_train = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),


])



transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),


])


def get_train_dataset():
    train_datasets = Caltech256(root='./datasets', is_train=True, transform=transform_train)
    return train_datasets

def get_unlabel_dataset():
    train_datasets = Caltech256(root='./datasets', is_train=True, transform=transform_test)
    return train_datasets

def get_test_dataset():
    test_datasets = Caltech256(root='./datasets', is_train=False, transform=transform_test)
    return test_datasets

def get_dataloader(dataset, batch_size, shuffle=True):
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=2)
    return dataloader

class Caltech256(VisionDataset):
    """`Caltech 256 <http://www.vision.caltech.edu/Image_Datasets/Caltech256/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``caltech256`` exists or will be saved to if download is set to True.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, root, transform=None, target_transform=None, download=False, is_train=False):
        super(Caltech256, self).__init__(os.path.join(root, 'caltech256'),
                                         transform=transform,
                                         target_transform=target_transform)




        self.categories = sorted(os.listdir(os.path.join(self.root, "256_ObjectCategories")))
        self.index = []
        self.y = []
        self.is_train = is_train
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "256_ObjectCategories", c)))

            test_ind = np.array(range(1, n + 1, 4))
            train_ind = np.setdiff1d(np.array(range(1, n + 1, 1)), test_ind)

            if self.is_train is True:
                self.index.extend(train_ind.tolist())
                self.y.extend(len(train_ind.tolist()) * [i])
            else:
                self.index.extend(test_ind.tolist())
                self.y.extend(len(test_ind.tolist()) * [i])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(os.path.join(self.root,
                                      "256_ObjectCategories",
                                      self.categories[self.y[index]],
                                      "{:03d}_{:04d}.jpg".format(self.y[index] + 1, self.index[index]))).convert('RGB')

        target = self.y[index]

        # print(img.size)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self):
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "256_ObjectCategories"))

    def __len__(self):
        return len(self.index)


if __name__ == '__main__':
    ds = Caltech256(root='/home/ytx/桌面/reaearch/metric_active/datasets')
    print(len(ds))
    ds = Caltech256(root='/home/ytx/桌面/reaearch/metric_active/datasets', is_train=True)
    print(len(ds))
