from typing import Callable, Optional
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os


class CIFAR5m(Dataset):
    def __init__(self, data_dir: str = None, transform: Optional[Callable] = None) -> None:
        self.transform = transform

        if data_dir is None:
            data_dir = "/ml/imagenet/data/cifar5m"
        print(f"reading cifar 5m data from {data_dir}")

        class_files = []
        class_labels = []

        num_classes = 10
        min_len = 509716

        for i in range(num_classes):
            print("reading class %d" % i)
            file_name = f"{data_dir}/class{i}.npy"
            curr_data = np.load(file_name)
            class_files.append(curr_data[:min_len, ...])
            class_labels.append(i * np.ones(min_len, dtype=int))

        self.data_x = np.concatenate(class_files)
        self.data_y = np.concatenate(class_labels)

    def __getitem__(self, index):
        img, target = self.data_x[index], self.data_y[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.data_x.shape[0]

    def extract_subset(self, data_dir: str, idx: np.ndarray, subset_name: str):
        data_x = self.data_x[idx, ...]
        data_y = self.data_y[idx]

        np.save(os.path.join(data_dir, subset_name + "_data.npy"), data_x)
        np.save(os.path.join(data_dir, subset_name + "_labels.npy"), data_y)
