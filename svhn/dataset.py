import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np

def get(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, show=False, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building SVHN data loader with {} workers".format(num_workers))

    def target_transform(target):
        # print(target)
        return int(target) - 1

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    # 0.4914    0.4824    0.4467
    # 0.2471    0.2435    0.2616
    if train:
        transform = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transform
        ])

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='train', download=True,
                transform=transform,
                # target_transform=lambda x: int(x) - 1,
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='test', download=True,
                transform=transform,
                # target_transform=target_transform
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    if show:
        show_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='test', download=True,
                transform=transform,
                # target_transform=target_transform
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(show_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

