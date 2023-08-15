# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import str_to_pil_interp

from .dataset import *
from .samplers import *


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    (dataset_val, dataset_test), _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
#     if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
#         indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
#         sampler_train = SubsetRandomSampler(indices)
#     else:
#         sampler_train = torch.utils.data.DistributedSampler(
#             dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
#         )
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    if dataset_test:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False
        )
    else:
        data_loader_test = data_loader_val

   
    return dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'cheX':
        dataset = GetCheXData(config=config, transform=transform, isTrain=is_train)
        nb_classes = 13
    elif config.DATA.DATASET == 'nih':
        dataset = GetNIHData(config=config, transform=transform, isTrain=is_train)
        nb_classes = 14
    else:
        raise NotImplementedError("dataset error.")

    return dataset, nb_classes


def build_transform(is_train, config):
    t = []
    t.append(transforms.Resize(256, interpolation=str_to_pil_interp(config.DATA.INTERPOLATION)))
    t.append(transforms.CenterCrop(224))
    if is_train:
        t.append(transforms.RandomHorizontalFlip(p=0.3))
        t.append(transforms.RandomVerticalFlip(p=0.2))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
