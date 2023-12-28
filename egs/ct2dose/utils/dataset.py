# coding = utf-8
# @Time    : 2023-12-26  14:34:11
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: DOSE PREDICTION.

import argparse
import glob
import logging
import os
from pathlib import Path
import sys
from datetime import datetime
import numpy as np
import torch
from monai.config import print_config
from monai.data import ArrayDataset, DataLoader, partition_dataset, CacheDataset
from monai.networks.nets import UNet
from monai.transforms import (Compose, LoadImaged, LoadImage, EnsureChannelFirstd, ScaleIntensity, ScaleIntensityd)
from monai.transforms import (RandAffine, RandAffined, EnsureChannelFirst, EnsureChannelFirst, RandSpatialCrop, RandSpatialCropd)
from monai.transforms import (Lambda, RandFlip, RandFlipd, RandRotate90, RandRotate90d, Resize, Resized, ConcatItemsd)
from monai.transforms import (DeleteItemsd)
from monai.handlers import StatsHandler, TensorBoardStatsHandler

def ct2dose_dataset(images, labels, masks):
    # Ensure the lengths of images, labels, and masks match
    assert len(images) == len(labels) == len(masks), "Mismatch in dataset lengths"
    # Split dataset into training and validation sets
    random_seed = 123
    train_frac = 0.8
    # Check filename consistency
    for i in range(len(images)):
        assert images[i].split('/')[-2] == labels[i].split('/')[-2], "Mismatch in dataset filenames"


    data_dicts = [{'image': img, 'y': lbl, 'mask': msk} for img, lbl, msk in zip(images, labels, masks)]
    transforms = Compose([
        LoadImaged(keys=['image', 'y', 'mask']),
        EnsureChannelFirstd(keys=['image', 'y', 'mask']),
        ScaleIntensityd(keys=['image', 'y', 'mask']),
        Resized(keys=['image', 'y', 'mask'], spatial_size=(96, 96, 96)),
        RandAffined(keys=['image', 'y', 'mask'], prob=0.15, translate_range=(10, 10, 10), 
                    rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36), scale_range=(0.15, 0.15, 0.15), padding_mode="zeros"),
        RandSpatialCropd(keys=['image', 'y', 'mask'], roi_size=(96, 96, 96), random_size=False),
        RandFlipd(keys=['image', 'y', 'mask'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image', 'y', 'mask'], prob=0.5, spatial_axis=1),
        RandFlipd(keys=['image', 'y', 'mask'], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=['image', 'y', 'mask'], prob=0.5, max_k=3),
        ConcatItemsd(keys=['image', 'mask'], name='x', dim=0),
        # remove the original image and mask
        DeleteItemsd(keys=['image', 'mask']),
        Lambda(lambda x: (x['x'], x['y']))
    ])

    num_train = int(train_frac * len(data_dicts))
    train_data_dicts = data_dicts[:num_train]
    val_data_dicts = data_dicts[num_train:]

    train_ds = CacheDataset(data=train_data_dicts, transform=transforms, cache_rate=1.0, num_workers=4)
    val_ds = CacheDataset(data=val_data_dicts, transform=transforms, cache_rate=1.0, num_workers=4)
    # train_ds = ArrayDataset(train_data_dicts, transforms)
    # val_ds = ArrayDataset(val_data_dicts, transforms)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=8, num_workers=8, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader




