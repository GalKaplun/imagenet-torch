from torchvision.datasets import VisionDataset
import os

osj = os.path.join
cifar10_label_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class CustomDataset(VisionDataset):
    def __init__(self, root, data, targets, original_targets=None, transform=None):
        super(CustomDataset, self).__init__(root, transform=None, target_transform=None)
        self.data, self.targets = data, targets
        self.original_targets = original_targets
        self.my_transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index][0], int(self.targets[index])
        # if self.my_transform is not None:
            # img = self.my_transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

    def get_original_label(self, index):
        return int(self.original_targets[index])

    def get_raw_image(self, index):
        return self.data[index][0]


def get_unorm(dataset):
    if dataset.startswith("CIFAR10"):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
    elif dataset.lower() == 'cifar100':
        mean = (0.5074, 0.4867, 0.4411)
        std = (0.2011, 0.1987, 0.2025)
    elif dataset.lower() == 'cifar5m':
        mean = (0.4555, 0.4362, 0.3415)
        std = (0.2284, 0.2167, 0.2165)
    else:
        raise NotImplementedError()
    return UnNormalize(mean, std)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
