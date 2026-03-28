import torch
import torch.nn as nn
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

def get_early_fusion_m2f(config_name, num_labels, ignore_mismatched_sizes, ignore_value):

    # Import and adapt config
    config = Mask2FormerConfig.from_pretrained(
        config_name, 
        num_labels=num_labels,
        ignore_value=ignore_value
    )

    config.backbone_config.num_channels = 7 

    # Create new model with fresh weights
    model = Mask2FormerForUniversalSegmentation(config)
    
    # Load pretrained Mask2Former model
    pretrained_rgb = Mask2FormerForUniversalSegmentation.from_pretrained(
        config_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=ignore_mismatched_sizes
    )
    pretrained_dict = pretrained_rgb.state_dict()
    new_state_dict = model.state_dict()
    
    # Adapt the first convolutional layer weights to accommodate the 7-channel input (RGB + LiDAR range + LiDAR mask)
    # Copy the RGB weights and initialize the new LiDAR channels with zeros
    patch_key = "model.pixel_level_module.encoder.embeddings.patch_embeddings.projection.weight"
    for key in pretrained_dict.keys():
        if key == patch_key:
            rgb_weights = pretrained_dict[key] 
            
            new_weights = torch.zeros(
                (rgb_weights.shape[0], 7, rgb_weights.shape[2], rgb_weights.shape[3]),
                dtype=rgb_weights.dtype,
                device=rgb_weights.device
            )
            
            with torch.no_grad():
                new_weights[:, :3, :, :] = rgb_weights
            
            new_state_dict[key] = new_weights
        else:
            new_state_dict[key] = pretrained_dict[key]

    # Load the adapted state dict into the new model
    model.load_state_dict(new_state_dict)
    
    return model

class Mask2FormerLidarEarlyFusion(nn.Module):
    def __init__(self, config_name, num_labels, ignore_mismatched_sizes, ignore_value):
        super().__init__()
        self.model = get_early_fusion_m2f(config_name, num_labels, ignore_mismatched_sizes, ignore_value)

    def forward(self, pixel_values=None, lidar_values=None, lidar_mask=None, **kwargs):
        assert lidar_values is not None and lidar_mask is not None, "LiDAR values and mask must be provided for early fusion"
        
        pixel_values = torch.cat([pixel_values, lidar_values, lidar_mask], dim=1)
        return self.model(pixel_values=pixel_values, **kwargs)