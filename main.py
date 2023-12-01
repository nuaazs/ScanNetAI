import glob
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import nibabel as nib
import numpy as np
from torch.nn import MSELoss
from monai.config import print_config
from monai.data import ArrayDataset, create_test_image_3d, decollate_batch, DataLoader
from monai.handlers import (
    MeanDice,
    MLFlowHandler,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
)
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandAffine,
    RandSpatialCrop,
    Resize,
    ScaleIntensity,
)
from monai.utils import first
import ignite
import torch

# Config
print_config()

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--exp_name', type=str, default='unetr',help='')
parser.add_argument('--input_data_dir', type=str, default='./liuhuan',help='')
parser.add_argument('--output_data_dir', type=str, default='./liuhuan',help='')
args = parser.parse_args()
args.root_dir = os.path.join('./exp', args.exp_name)
os.makedirs(args.root_dir, exist_ok=True)

# Logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load dataset
images = sorted(glob.glob(os.path.join(args.input_data_dir, "*.nii.gz")))
outputs = sorted(glob.glob(os.path.join(args.output_data_dir, "*.nii.gz")))
# Check images and outputs have same length
assert len(images) == len(outputs), f"len(images)={len(images)}, len(outputs)={len(outputs)}"
assert len(images) > 0, "Empty dataset"
# assert all(Path(s).name == Path(i).name for i, s in zip(images, outputs)), "Mismatched images and outputs"
assert all(Path(s).name == Path(i).name for i, s in zip(images, outputs)), "Mismatched images and outputs" 

# Setup transforms, dataset
# Define transforms for input image and output image
imtrans = Compose(
    [
        LoadImage(image_only=True),
        ScaleIntensity(),
        Resize((96, 96, 96)),
        RandAffine(
            prob=0.15,
            translate_range=(10, 10, 10),
            rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
            scale_range=(0.15, 0.15, 0.15),
            padding_mode="zeros",
        ),
        EnsureChannelFirst(),
        RandSpatialCrop((96, 96, 96), random_size=False),
    ]
)

# Define nifti dataset, dataloader
ds = ArrayDataset(images, imtrans, outputs, imtrans)
loader = DataLoader(ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available())
im, out = first(loader)
print(f"image shape: {im.shape}, output shape: {out.shape}")


# Create Model, Loss, Optimizer
device = torch.device("cuda:0")
net = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

loss = MSELoss()
lr = 1e-3
opt = torch.optim.Adam(net.parameters(), lr)


# Create trainer
trainer = ignite.engine.create_supervised_trainer(net, opt, loss, device, False)

# optional section for checkpoint and tensorboard logging
# adding checkpoint handler to save models (network
# params and optimizer stats) during training
log_dir = os.path.join(args.root_dir, "logs")
checkpoint_handler = ignite.handlers.ModelCheckpoint(log_dir, "net", n_saved=10, require_empty=False)
trainer.add_event_handler(
    event_name=ignite.engine.Events.EPOCH_COMPLETED,
    handler=checkpoint_handler,
    to_save={"net": net, "opt": opt},
)

# StatsHandler prints loss at every iteration
# user can also customize print functions and can use output_transform to convert
# engine.state.output if it's not a loss value
train_stats_handler = StatsHandler(name="trainer", output_transform=lambda x: x)
train_stats_handler.attach(trainer)

# TensorBoardStatsHandler plots loss at every iteration
train_tensorboard_stats_handler = TensorBoardStatsHandler(log_dir=log_dir, output_transform=lambda x: x)
train_tensorboard_stats_handler.attach(trainer)

# # MLFlowHandler plots loss at every iteration on MLFlow web UI
# mlflow_dir = os.path.join(log_dir, "mlruns")
# train_mlflow_handler = MLFlowHandler(tracking_uri=Path(mlflow_dir).as_uri(), output_transform=lambda x: x)
# train_mlflow_handler.attach(trainer)


# Add Validation every N epochs
# optional section for model validation during training
validation_every_n_epochs = 1
# Set parameters for validation MSE
metric_name = "MSE"
# add evaluation metric to the evaluator engine
val_metrics = {metric_name: MSELoss()}
# Ignite evaluator expects batch=(img, seg) and
# returns output=(y_pred, y) at every iteration,
# user can add output_transform to return other values
evaluator = ignite.engine.create_supervised_evaluator(
    net,
    val_metrics,
    device,
    True,
    output_transform=None
)

# create a validation data loader
val_imtrans = Compose(
    [
        LoadImage(image_only=True),
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize((96, 96, 96)),
    ]
)

val_ds = ArrayDataset(images[21:], val_imtrans, outputs[21:], val_imtrans)
val_loader = DataLoader(val_ds, batch_size=5, num_workers=8, pin_memory=torch.cuda.is_available())


@trainer.on(ignite.engine.Events.EPOCH_COMPLETED(every=validation_every_n_epochs))
def run_validation(engine):
    evaluator.run(val_loader)


# Add stats event handler to print validation stats via evaluator
val_stats_handler = StatsHandler(
    name="evaluator",
    # no need to print loss value, so disable per iteration output
    output_transform=lambda x: None,
    # fetch global epoch number from trainer
    global_epoch_transform=lambda x: trainer.state.epoch,
)
val_stats_handler.attach(evaluator)

# add handler to record metrics to TensorBoard at every validation epoch
val_tensorboard_stats_handler = TensorBoardStatsHandler(
    log_dir=log_dir,
    # no need to plot loss value, so disable per iteration output
    output_transform=lambda x: None,
    # fetch global epoch number from trainer
    global_epoch_transform=lambda x: trainer.state.epoch,
)
val_tensorboard_stats_handler.attach(evaluator)

# add handler to record metrics to MLFlow at every validation epoch
# val_mlflow_handler = MLFlowHandler(
#     tracking_uri=Path(mlflow_dir).as_uri(),
#     # no need to plot loss value, so disable per iteration output
#     output_transform=lambda x: None,
#     # fetch global epoch number from trainer
#     global_epoch_transform=lambda x: trainer.state.epoch,
# )
# val_mlflow_handler.attach(evaluator)

# add handler to draw the first image and the corresponding
# label and model output in the last batch
# here we draw the 3D output as GIF format along Depth
# axis, at every validation epoch
# val_tensorboard_image_handler = TensorBoardImageHandler(
#     log_dir=log_dir,
#     batch_transform=lambda batch: (batch[0], batch[1]),
#     output_transform=lambda output: output[0],
#     global_iter_transform=lambda x: trainer.state.epoch,
# )
# evaluator.add_event_handler(
#     event_name=ignite.engine.Events.EPOCH_COMPLETED,
#     handler=val_tensorboard_image_handler,
# )

# Run training loop
# create a training data loader
train_ds = ArrayDataset(images[:20], imtrans, outputs[:20], imtrans)
train_loader = DataLoader(
    train_ds,
    batch_size=5,
    shuffle=True,
    num_workers=8,
    pin_memory=torch.cuda.is_available(),
)

max_epochs = 10
state = trainer.run(train_loader, max_epochs)