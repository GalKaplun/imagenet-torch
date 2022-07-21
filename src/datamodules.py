import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import os

from src.cifar5m import CIFAR5m
from src.data_utils import CustomDataset
import torchvision.datasets as datasets

def get_all_cinic():
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std)
        
    ])
    return datasets.ImageFolder(root='/galk/ensemble/data/cinic-10/all/', transform=transform)

def get_cinic(is_train, root_dir='/galk/ensemble/data/cinic-10/'):
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std)
    ])

    if is_train:
        train_dataset = datasets.ImageFolder(root=os.path.join(root_dir, 'train'), transform=transform)
        return train_dataset
    else:
        test_dataset = datasets.ImageFolder(root=os.path.join(root_dir, 'test'), transform=transform)
        return test_dataset


def get_cifar5m():

    mean = (0.4555, 0.4362, 0.3415)
    std = (0.2284, 0.2167, 0.2165)
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return CIFAR5m(data_dir=None, transform=transform)


def get_cifar(data_path, train, no_transform=False):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    if train:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
    if no_transform:
        transform = None
    return CIFAR10(data_path, train=train, transform=transform, download=True)


def get_noisy_cifar(data_path):
    reg_cifar = get_cifar(data_path, train=True)

    # targets = np.fromfile('/mobileye/algo_Research_05/gal/code/repos/ensembling/files/noisy_labels.npy',
                        #   dtype='int64')
    targets = np.random.randint(0, 10, 50000)
    return CustomDataset(data_path, reg_cifar, targets=targets, original_targets=reg_cifar.targets)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = None,
        batch_size: int = 256,
        num_workers: int = 4,
        add_noise: bool = False,
        dataset: str = 'cifar10'
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.add_noise = add_noise
        self.dataset = dataset

    def setup(self, stage=None):
        if self.dataset.lower() == 'cifar10':
            if self.add_noise:
                self.train_set = get_noisy_cifar(data_path=self.data_dir)
            else:
                self.train_set = get_cifar(data_path=self.data_dir, train=True)
        elif self.dataset.lower() == 'cifar5m':
            self.train_set = get_cifar5m()
        if self.dataset.lower() == 'cinic':
            self.train_set = get_all_cinic()
            if self.add_noise:
                self.train_set = CustomDataset(root='/galk/ensemble/data/cinic-10/all/', data=self.train_set,
                                targets=np.random.randint(0, 10, len(self.train_set)))
        self.val_set = get_cifar(data_path=self.data_dir, train=False)

    def train_dataloader(self):
        print(self.batch_size)
        print(self.num_workers)
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=not self.add_noise,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2
        )

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          prefetch_factor=2)
