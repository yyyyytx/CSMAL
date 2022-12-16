from torchvision import transforms
from PIL import Image
import os
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
import numpy as np

import  torchvision



train_trans = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomCrop(128, padding=4),
        transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
])
test_trans = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
])


def get_train_dataset():
    train_datasets = Caltech101(root='/home/liu/桌面/datasets', is_train=True, transform=train_trans)
    return train_datasets

def get_test_dataset():
    test_datasets = Caltech101(root='/home/liu/桌面/datasets', is_train=False, transform=test_trans)
    return test_datasets

def get_unlabel_dataset():
    train_datasets = Caltech101(root='/home/liu/桌面/datasets', is_train=True, transform=test_trans)
    return train_datasets


def get_dataloader(train_datasets, train_batch_size, shuffle=True):
    train_dataloader = DataLoader(dataset=train_datasets,
                                  batch_size=train_batch_size,
                                  shuffle=shuffle,
                                  num_workers=4)
    return train_dataloader


class Caltech101(VisionDataset):
    """`Caltech 101 <http://www.vision.caltech.edu/Image_Datasets/Caltech101/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``caltech101`` exists or will be saved to if download is set to True.
        target_type (string or list, optional): Type of target to use, ``category`` or
        ``annotation``. Can also be a list to output a tuple with all specified target types.
        ``category`` represents the target class, and ``annotation`` is a list of points
        from a hand-generated outline. Defaults to ``category``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, root, target_type="category", transform=None,
                 target_transform=None, train_idx=None, test_idx = None, is_train=False):
        super(Caltech101, self).__init__(os.path.join(root, 'caltech101'),
                                         transform=transform,
                                         target_transform=target_transform)
        # makedir_exist_ok(self.root)
        if not isinstance(target_type, list):
            target_type = [target_type]
        # self.target_type = [verify_str_arg(t, "target_type", ("category", "annotation"))
        #                     for t in target_type]
        self.taget_type = target_type

        self.is_train = is_train
        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

        self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        self.categories.remove("BACKGROUND_Google")  # this is not a real class

        # print(self.categories)
        # For some reason, the category names in "101_ObjectCategories" and
        # "Annotations" do not always match. This is a manual map between the
        # two. Defaults to using same name, since most names are fine.
        name_map = {"Faces": "Faces_2",
                    "Faces_easy": "Faces_3",
                    "Motorbikes": "Motorbikes_16",
                    "airplanes": "Airplanes_Side_2"}
        self.annotation_categories = list(map(lambda x: name_map[x] if x in name_map else x, self.categories))

        self.index = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
            # test_ind = range(1, n+1, 3)
            # train_ind = list(range(n)) - test_ind
            # print('c type len:', c, n)
            test_ind = np.array(range(1, n+1, 4))
            train_ind = np.setdiff1d(np.array(range(1, n+1 , 1)), test_ind)


            if self.is_train is True:
                self.index.extend(train_ind.tolist())
                self.y.extend(len(train_ind.tolist()) * [i])
            else:
                self.index.extend(test_ind.tolist())
                self.y.extend(len(test_ind.tolist()) * [i])

            # print(self.index)
            # break
            # print('total len:',len(self.y))

        # print(self.y)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        import scipy.io
        # print('index',index, len(self.y))
        # print('y',self.y[index])
        img_path = os.path.join(self.root,
                                      "101_ObjectCategories",
                                      self.categories[self.y[index]],
                                      "image_{:04d}.jpg".format(self.index[index]))
        img = Image.open(img_path).convert('RGB')

        target = []
        target.append(self.y[index])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            image = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target, img_path


    def __len__(self):
        return len(self.index)

    def classes(self):
        return self.annotation_categories

if __name__ == '__main__':
    d = get_train_dataset()
    print(len(d))
    d = get_test_dataset()
    print(len(d))
