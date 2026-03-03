import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder

from utils.distributed import get_world_size

def build_ssl_loader_stl10(
    data_dir: str,
    transform,
    batch_size: int,
    num_workers: int,
    is_distributed: bool,
):
    """
    STL10 SSL loader for your ImageFolder-formatted directory:

      datas/stl10/
        ├── train/
        ├── unlabeled/
        └── val/

    This loader uses: datas/stl10/unlabeled
    """
    dataset = ImageFolder(root=os.path.join(data_dir, "unlabeled"), transform=transform)

    sampler = None
    if is_distributed and get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return loader, sampler

def build_ssl_loader(
    data_dir: str,
    transform,
    batch_size: int,
    num_workers: int,
    is_distributed: bool,
    # seed: int = 0,
):
    dataset = ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)

    sampler = None
    if is_distributed and get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=True, 
                                    #  seed=seed, 
                                     drop_last=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return loader, sampler


def build_eval_loader(
    data_dir: str,
    transform,
    batch_size: int,
    num_workers: int,
    split: str = "val",
):
    dataset = ImageFolder(root=os.path.join(data_dir, split), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return loader

def build_linear_eval_loaders(
    data_dir: str,
    transform,
    batch_size: int,
    num_workers: int,
):

    train_set = ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
    val_set = ImageFolder(root=os.path.join(data_dir, "val"), transform=transform)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return train_loader, val_loader, len(train_set.classes)
