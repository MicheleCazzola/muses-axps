import argparse

import torch
from enum import Enum

from src.config import DEVICE, NUM_EPOCHS
from src.data.dataloaders import get_dataloaders

# ----------------------------------------------------------
# CATEGORY MAPPING (MUSES IDs -> contiguous labels)
# ----------------------------------------------------------

train_loader, val_loader, test_loader = get_dataloaders()
train_dataset = train_loader.dataset

# Create mapping from MUSES category IDs to contiguous label indices for training and evaluation
dataset_categories = sorted(train_dataset.categories, key=lambda c: c["id"])
stuff_classes_ids = [i for i, c in enumerate(dataset_categories) if c["isthing"] == 0]
id2index = {c["id"]: i for i, c in enumerate(dataset_categories)}
index2id = {i: c["id"] for i, c in enumerate(dataset_categories)}

# Number of classes for panoptic segmentation (including "stuff" and "thing" classes)
num_classes = len(dataset_categories)

# Type of task to perform (training, validation, testing or resource measurement)
class TaskType(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    MEASURE = "measure"


# Convert panoptic segmentation mask with encoded segment IDs into RGB format for visualization
def id2rgb(mask):
    r = mask % 256
    g = (mask // 256) % 256
    b = (mask // (256 * 256)) % 256
    return torch.stack([r, g, b], dim=0)

    
# Get optimizer and scheduler for training depending on the specified LiDAR fusion method
def get_training_optim(model, lidar_mid=False, lidar_early=False):
    
    assert not (lidar_mid and lidar_early), "Cannot use both LiDAR mid-fusion and LiDAR early fusion at the same time"
    
    if lidar_mid:     # LiDAR mid-fusion
        optimizer = torch.optim.AdamW(
            [
                {"params": model.model.parameters(), "lr": 1e-5, "weight_decay": 0.01},
                {"params": model.lidar_encoder.parameters(), "lr": 5e-4, "weight_decay": 0.001},
                {"params": model.fusion.parameters(), "lr": 5e-4, "weight_decay": 0.001}
            ]
        )
    elif lidar_early:       # LiDAR early fusion
        projector_params = list(model.model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection.parameters())
        other_params = [p for n, p in model.named_parameters() if "model.pixel_level_module.encoder.embeddings.patch_embeddings.projection" not in n]
        
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": 1e-5, "weight_decay": 0.01},
                {"params": projector_params, "lr": 5e-4, "weight_decay": 0.001}
            ]
        )
    else:       # Standard RGB-only model
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-5,
            weight_decay=0.01
        )
    
    # Using a cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=20,
    )
    
    return {
        "optimizer": optimizer,
        "scheduler": scheduler
    }
    
    
# Get args from command line for training and evaluation setup
def get_args():
    parser = argparse.ArgumentParser(description="Panoptic segmentation with Mask2Former and LiDAR fusion")
    
    parser.add_argument("--mode", type=TaskType, choices=list(TaskType), required=True, help="Mode to run: train, valid, test or measure")
    parser.add_argument("--backbone", type=str, default="tiny", choices=["tiny", "small", "base"], help="Backbone size for Mask2Former")
    parser.add_argument("--resize", type=int, nargs=2, default=(1080, 1920), help="Image size (height width) to resize to during training and evaluation")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of epochs for training (overrides default)")
    parser.add_argument("--lidar_mid", action="store_true", help="Whether to use LiDAR mid-fusion architecture")
    parser.add_argument("--lidar_early", action="store_true", help="Whether to use LiDAR early fusion architecture")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to pretrained model checkpoint for evaluation or fine-tuning")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device to run on (e.g. 'cuda', 'mps', 'cpu')")
    parser.add_argument("--output_dir", type=str, default="./run_results", help="Directory to save panoptic segmentation results (predicted masks and JSON annotations)")
    parser.add_argument("--reduce_factor", type=int, default=1, help="Factor to reduce the dataset size for faster experimentation")
    args = parser.parse_args()
    
    return args