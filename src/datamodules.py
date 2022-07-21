import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import os

from src.cifar5m import CIFAR5m
from src.data_utils import CustomDataset
import torchvision.datasets as datasets


def get_imagenet(data_path, train, no_transform=False):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if train:
        transform = transforms.Compose(
            [   
                transforms.Resize(256),
                transforms.RandomCrop(224, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        transform = transforms.Compose(
            [   
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]
        )
    if no_transform:
        transform = None
    
    data_path = os.path.join(data_path, 'imagenet', 'train' if train else 'val')
    return datasets.ImageFolder(root=data_path, transform=transform)

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


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = None,
        batch_size: int = 256,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        dataset: str = 'imagenet'
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.prefetch_factor = prefetch_factor

    def setup(self, stage=None):
        if self.dataset.lower() == 'cifar10':
            self.train_set = get_cifar(data_path=self.data_dir, train=True)
            self.val_set = get_cifar(data_path=self.data_dir, train=False)
        elif self.dataset.lower() == 'cifar5m':
            self.train_set = get_cifar5m()
            self.val_set = get_cifar(data_path=self.data_dir, train=False)
        elif self.dataset.lower() == 'cinic':
            self.train_set = get_all_cinic()
            self.val_set = get_cifar(data_path=self.data_dir, train=False)
        elif self.dataset.lower() == 'imagenet':
            self.train_set = get_imagenet(data_path=self.data_dir, train=True)
            self.val_set = get_imagenet(data_path=self.data_dir, train=False)
        

    def train_dataloader(self):
        print(self.batch_size)
        print(self.num_workers)
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2 #todo check if this is best
        )

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True,
                          prefetch_factor=self.prefetch_factor) # todo check if this is best
