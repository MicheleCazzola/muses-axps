import torch
from torch import nn
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from src.config import DEVICE
from src.utils.utils import num_classes
from src.modeling.lidar_mid_fusion import Mask2FormerLidarMidFusion
from src.modeling.lidar_early_fusion import Mask2FormerLidarEarlyFusion

# Get model class depending on the specified fusion method
def get_model(model_size="tiny", image_size=(1080, 1920), lidar_mid=False, lidar_early=False):
    
    assert not (lidar_mid and lidar_early), "Cannot use both LiDAR mid-fusion and LiDAR early fusion at the same time"
    
    print(f"Loading model with backbone size '{model_size}' and image size {image_size}")
    
    # Load image processor
    image_processor = AutoImageProcessor.from_pretrained(
        f"facebook/mask2former-swin-{model_size}-cityscapes-panoptic",
        num_labels=num_classes,
        ignore_index=0,
        use_fast=True,
        do_rescale=True,
        do_normalize=True,
        do_resize=True,
        size=image_size,
    )
    
    # Load standard Mask2Former model pretrained on Cityscapes panoptic segmentation
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        f"facebook/mask2former-swin-{model_size}-cityscapes-panoptic",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        ignore_value=0
    )
    
    # Wrap the standard model with the LiDAR mid-fusion or early fusion architecture if specified
    if lidar_mid:
        print("Loading model for LiDAR mid-fusion")
        model = Mask2FormerLidarMidFusion(model)
    elif lidar_early:
        print("Loading model for LiDAR early fusion")
        model = Mask2FormerLidarEarlyFusion(
            config_name=f"facebook/mask2former-swin-{model_size}-cityscapes-panoptic",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            ignore_value=0
        )
    else:
        print("Loading standard RGB-only model")

    model = model.to(DEVICE)
        
    return model, image_processor

# Load pretrained model checkpoint, optionally loading optimizer and scheduler states as well for training resumption
def load_chp(checkpoint_path: str, model: nn.Module, optimizer: torch.optim.Optimizer=None, scheduler:torch.optim.lr_scheduler.LRScheduler=None, device=DEVICE, sd_only=False):
    
    # Load checkpoint and model state dict
    chp = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(chp["model_state_dict"], strict=True)
    
    if not sd_only:
        
        # Load optimizer if state dict is present in checkpoint and if optimizer is provided
        if optimizer is not None and "optimizer_state_dict" in chp:
            optimizer.load_state_dict(chp["optimizer_state_dict"])
            print("Optimizer state loaded from checkpoint")
        elif optimizer is not None and "optimizer_state_dict" not in chp:
            print("WARNING: No optimizer state found in checkpoint, starting with fresh optimizer")
        elif optimizer is None and "optimizer_state_dict" in chp:
            print("WARNING: No optimizer state found in checkpoint, but optimizer provided. This may lead to suboptimal training resumption.")
        
        # Load scheduler if state dict is present in checkpoint and if scheduler is provided
        if scheduler is not None and "scheduler_state_dict" in chp:
            scheduler.load_state_dict(chp["scheduler_state_dict"])
            print("Scheduler state loaded from checkpoint")
        elif scheduler is not None and "scheduler_state_dict" not in chp:
            print("WARNING: No scheduler state found in checkpoint, starting with fresh scheduler")
        elif scheduler is None and "scheduler_state_dict" in chp:
            print("WARNING: No scheduler state found in checkpoint, but scheduler provided. This may lead to suboptimal training resumption.")
        
    print(f"Loaded checkpoint '{checkpoint_path}' at epoch {chp['epoch']}")
    if "pq" in chp:
        print(f"Checkpoint PQ: {chp['pq']['All']['pq']:.4f}")
    else:
        print("No PQ found in checkpoint")

# Save model checkpoint, optionally including optimizer and scheduler states for training resumption
def save_chp(save_path, model, optimizer=None, scheduler=None, epoch=None, pq=None):
    chp = {
        "epoch": epoch,
        "pq": pq,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        chp["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        chp["scheduler_state_dict"] = scheduler.state_dict()
    
    torch.save(chp, save_path)