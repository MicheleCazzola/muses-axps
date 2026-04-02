import argparse
import os
import yaml

import torch
from enum import Enum

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
def get_training_optim(cfg_folder, model, lidar_mid=False, lidar_early=False):
    
    assert not (lidar_mid and lidar_early), "Cannot use both LiDAR mid-fusion and LiDAR early fusion at the same time"
    
    match (lidar_mid, lidar_early):
        # LiDAR mid-fusion
        case (True, False):     
            with open(os.path.join(cfg_folder, "mid_fusion.yaml"), "r") as f:
                cfg = yaml.safe_load(f)
                
            learning_rates, weight_decays = map(lambda x: list(cfg[x].values()), ["learning_rate", "weight_decay"])
            model_groups = [
                model.model.parameters(),
                model.lidar_encoder.parameters(),
                model.fusion.parameters()
            ]
            param_groups = [
                {"params": params, "lr": lr, "weight_decay": wd}
                for params, lr, wd in zip(model_groups, learning_rates, weight_decays)
            ]
            t_max = cfg["t_max"]
            
        # LiDAR early fusion
        case (False, True):    
            with open(os.path.join(cfg_folder, "early_fusion.yaml"), "r") as f:
                cfg = yaml.safe_load(f)
                
            learning_rates, weight_decays = map(lambda x: list(cfg[x].values()), ["learning_rate", "weight_decay"])
            projector_params = list(model.model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection.parameters())
            other_params = [p for n, p in model.named_parameters() if "model.pixel_level_module.encoder.embeddings.patch_embeddings.projection" not in n]
            param_groups = [
                {"params": other_params, "lr": learning_rates[0], "weight_decay": weight_decays[0]},
                {"params": projector_params, "lr": learning_rates[1], "weight_decay": weight_decays[1]}
            ]
            t_max = cfg["t_max"]
            
        # Standard RGB-only model
        case (False, False):   
            with open(os.path.join(cfg_folder, "rgb.yaml"), "r") as f:
                cfg = yaml.safe_load(f)
                
            learning_rate, weight_decay = map(cfg.get, ["learning_rate", "weight_decay"])
            param_groups = [{"params": model.parameters(), "lr": learning_rate, "weight_decay": weight_decay}]
            t_max = cfg["t_max"]
            
        # Invalid combination
        case _:                
            raise ValueError("Invalid combination of LiDAR fusion options")
        
    print(param_groups)
    
    # AdamW optimizer
    optimizer = torch.optim.AdamW(param_groups)
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    
    return {
        "optimizer": optimizer,
        "scheduler": scheduler
    }
    
    
# Get args from command line for training and evaluation setup
def get_args(base_cfg):
    
    pretrained_model, output_dir = map(base_cfg.get, ["pretrained_model", "output_dir"])
    backbone = base_cfg["backbone"]
    resize_model = tuple(map(base_cfg["resize_model"].get, ["width", "height"]))
    num_epochs, batch_size = map(base_cfg.get, ["num_epochs", "batch_size"])
    
    parser = argparse.ArgumentParser(description="Panoptic segmentation with Mask2Former and LiDAR fusion")
    
    parser.add_argument("--mode", type=TaskType, choices=list(TaskType), required=True, help="Mode to run: train, valid, test or measure")
    parser.add_argument("--backbone", type=str, default=backbone, choices=["tiny", "small", "base"], help="Backbone size for Mask2Former")
    parser.add_argument("--resize", type=int, nargs=2, default=resize_model, help="Image size (height width) to resize to during training and evaluation")
    parser.add_argument("--batch-size", type=int, default=batch_size, help="Batch size for training and evaluation")
    parser.add_argument("--num-epochs", type=int, default=num_epochs, help="Number of epochs for training (overrides default)")
    parser.add_argument("--lidar-mid", action="store_true", help="Whether to use LiDAR mid-fusion architecture")
    parser.add_argument("--lidar-early", action="store_true", help="Whether to use LiDAR early fusion architecture")
    parser.add_argument("--pretrained-path", type=str, default=pretrained_model, help="Path to pretrained model checkpoint for evaluation or fine-tuning")
    parser.add_argument("--output-dir", type=str, default=output_dir, help="Directory to save panoptic segmentation results (predicted masks and JSON annotations)")
    parser.add_argument("--reduce-factor", type=int, default=1, help="Factor to reduce the dataset size for faster experimentation")
    args = parser.parse_args()
    
    return args