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

from utils.dataset import ct2dose_dataset

# Initialize and parse command line arguments
parser = argparse.ArgumentParser(description='3D Dose Prediction Project')
parser.add_argument('--exp_name', type=str, default='unetr', help='Experiment name')
parser.add_argument('--data_dir', type=str, default='common/niis', help='Directory for input data')
parser.add_argument('--eval_interval', type=int, default=5, help='Interval of epochs to perform evaluation and visualization')
parser.add_argument('--gpus', type=str, default='1', help='Comma-separated list of GPU IDs to use for training')
args = parser.parse_args()

# Configure GPUs for training
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
# Define device (use CUDA if GPUs are available and specified)
device = torch.device("cuda" if torch.cuda.is_available() and args.gpus != '' else "cpu")


# Create experiment directory
exp_dir = os.path.join('./exp', args.exp_name)
os.makedirs(exp_dir, exist_ok=True)

# Configure logging
log_filename = os.path.join(exp_dir, f'train_{datetime.now().strftime("%Y%m%d-%H%M%S")}.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.info("Starting 3D Dose Prediction Project...")

# Load dataset and split
images = sorted(glob.glob(os.path.join(args.data_dir, "*/ct.nii.gz")))
labels = sorted(glob.glob(os.path.join(args.data_dir, "*/dose.nii.gz")))
ct2dose_dataset(images,labels)

# Split dataset into training and validation sets
random_seed = 123
train_frac = 0.8

# Set the random seed and shuffle images and labels in the same order
np.random.seed(random_seed)
np.random.shuffle(images)
np.random.seed(random_seed)
np.random.shuffle(labels)

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
    debug_transform,  # Debugging transform after loading the image
    # add_channel_dim,  # Manually add channel dimension
    ScaleIntensity(),
    Resize((96, 96, 96)),
    RandAffine(prob=0.15, translate_range=(10, 10, 10), rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
               scale_range=(0.15, 0.15, 0.15), padding_mode="zeros"),
    debug_transform,  # Debugging transform after EnsureChannelFirst
    RandSpatialCrop((96, 96, 96), random_size=False)
])

# Create MONAI datasets and data loaders
train_ds = ArrayDataset(train_images, imtrans, train_labels, imtrans)
val_ds = ArrayDataset(val_images, imtrans, val_labels, imtrans)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=1, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_ds, batch_size=2, num_workers=1, pin_memory=torch.cuda.is_available())

# Define device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2).to(device)

# Setup for multi-GPU training using DataParallel
if torch.cuda.device_count() > 1 and args.gpus != '':
    logging.info(f"Using {torch.cuda.device_count()} GPUs for training")
    model = torch.nn.DataParallel(model)

model.to(device)

# Set up loss function and optimizer
loss_function = MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Create Ignite trainer and evaluator
trainer = create_supervised_trainer(model, optimizer, loss_function, device=device)
evaluator = create_supervised_evaluator(model, metrics={'MSE': MeanSquaredError()}, device=device)

# Add handlers for logging, checkpointing, and tensorboard
checkpoint_handler = ModelCheckpoint(exp_dir, 'checkpoint', n_saved=10, require_empty=False)
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'model': model, 'optimizer': optimizer})

train_stats_handler = StatsHandler(output_transform=lambda x: x)
train_stats_handler.attach(trainer)

train_tensorboard_stats_handler = TensorBoardStatsHandler(
    log_dir=os.path.join(exp_dir, 'logs'),
    output_transform=lambda x: {'loss': x}
)
train_tensorboard_stats_handler.attach(trainer)

# Evaluation and visualization function
def evaluate_and_visualize(engine):
    epoch = engine.state.epoch
    if epoch % args.eval_interval == 0:
        evaluator.run(val_loader)
        # Visualization code here
        # Save visualizations as NIfTI files in exp_dir

# Attach evaluation and visualization to trainer
trainer.add_event_handler(Events.EPOCH_COMPLETED(every=args.eval_interval), evaluate_and_visualize)

# Run the training loop
logging.info("Starting training...")
max_epochs = 10
trainer.run(train_loader, max_epochs=max_epochs)

logging.info("Training completed.")

# Final evaluation at the end of training
logging.info("Running final evaluation...")
evaluator.run(val_loader)