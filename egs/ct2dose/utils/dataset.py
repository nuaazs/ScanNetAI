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
from torch.nn import MSELoss
from monai.config import print_config
from monai.data import ArrayDataset, DataLoader, partition_dataset
from monai.networks.nets import UNet
from monai.transforms import (Compose, LoadImage, ScaleIntensity, Resize, RandAffine, EnsureChannelFirst, RandSpatialCrop)
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import MeanSquaredError
from ignite.handlers import ModelCheckpoint, EarlyStopping
from monai.handlers import StatsHandler, TensorBoardStatsHandler

def ct2dose_dataset(images,labels):
    assert len(images) == len(labels), "Mismatch in dataset lengths"
    # Split dataset into training and validation sets
    random_seed = 123
    train_frac = 0.8
    # Set the random seed and shuffle images and labels in the same order
    
    np.random.seed(random_seed)
    np.random.shuffle(images)
    np.random.seed(random_seed)
    np.random.shuffle(labels)
    assert len(images) == len(labels), "Mismatch in dataset lengths"
    for i in range(len(images)):
        # print(f"Image: {images[i]}")
        # print(f"Label: {labels[i]}")
        assert images[i].split('/')[-2] == images[i].split('/')[-2], "Mismatch in dataset filenames"

    # Split into training and validation sets
    num_train = int(train_frac * len(images))
    train_images = images[:num_train]
    train_labels = labels[:num_train]
    val_images = images[num_train:]
    val_labels = labels[num_train:]


    def debug_transform(img):
        print("Current shape:", img.shape)
        return img

    def add_channel_dim(img):
        if img.ndim == 3:  # Check if the image is 3D without channel dimension
            img = img[np.newaxis, ...]  # Add a channel dimension
        return img

    imtrans = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        # debug_transform,
        ScaleIntensity(),
        Resize((96, 96, 96)),
        RandAffine(prob=0.15, translate_range=(10, 10, 10), rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
                scale_range=(0.15, 0.15, 0.15), padding_mode="zeros"),
        # debug_transform,  # Debugging transform after EnsureChannelFirst
        RandSpatialCrop((96, 96, 96), random_size=False)
    ])

    # Create MONAI datasets and data loaders
    train_ds = ArrayDataset(train_images, imtrans, train_labels, imtrans)
    val_ds = ArrayDataset(val_images, imtrans, val_labels, imtrans)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=8, num_workers=4, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader