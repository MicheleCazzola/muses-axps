import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Mask2FormerForUniversalSegmentation
from transformers.models.mask2former.modeling_mask2former import Mask2FormerForUniversalSegmentationOutput

# Lightweight LiDAR encoder to extract multi-scale features
class LidarEncoder(nn.Module):

    def __init__(self, channels):
        super().__init__()

        # Equivalent to Swin-T patch embedding, but with LiDAR input (range + mask) instead of RGB
        self.stage1 = nn.Sequential(
            nn.Conv2d(4, channels[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU()
        )

        # Stride 8
        self.stage2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU()
        )

        # Stride 16
        self.stage3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU()
        )

        # Stride 32
        self.stage4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU()
        )

    def forward(self, x):

        f1 = self.stage1(x)        # stride 4
        f2 = self.stage2(f1)       # stride 8
        f3 = self.stage3(f2)       # stride 16
        f4 = self.stage4(f3)       # stride 32

        return [f1, f2, f3, f4]
    
# Gated feature modulation for mid-level fusion of RGB and LiDAR features
class GatedFeatureModulation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        # Learnable modulation parameters conditioned on LiDAR features
        self.gamma = nn.Conv2d(channels, channels, kernel_size=1)
        self.beta = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate = nn.Conv2d(channels, channels, kernel_size=1)

        # Zero initialization to start with RGB-only features
        # and let the model learn how to use LiDAR modulation
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, rgb_feat, lidar_feat):
        gamma = self.gamma(lidar_feat)
        beta = self.beta(lidar_feat)
        gate = torch.sigmoid(self.gate(lidar_feat)) # gate in [0,1]

        # Gated fusion: 
        # - if gate is close to 0, rely mostly on RGB
        # - if gate is close to 1, apply full modulation
        lidar_contrib = gate * (gamma * rgb_feat + beta)
        fused = rgb_feat + lidar_contrib
        
        return fused
    
# Multi-scale fusion module that applies gated modulation at each feature level
class MultiScaleFusion(nn.Module):

    def __init__(self,channels):

        super().__init__()

        self.blocks = nn.ModuleList([
            GatedFeatureModulation(c) for c in channels
        ])

    def forward(self, rgb_feats, lidar_feats):

        fused = [
            self.blocks[i](rgb_feats[i], lidar_feats[i]) 
            for i in range(len(rgb_feats))
        ]

        return fused
    
# Main model that integrates the LiDAR encoder and multi-scale fusion into the Mask2Former architecture
class Mask2FormerLidarMidFusion(nn.Module):

    def __init__(self, rgb_model):

        super().__init__()

        self.model: Mask2FormerForUniversalSegmentation = rgb_model

        backbone_channels = [96, 192, 384, 768]

        # LiDAR encoder to extract multi-scale features from the 4-channel LiDAR input (range + mask)
        self.lidar_encoder = LidarEncoder(backbone_channels)
        
        # Multi-scale fusion module to combine RGB and LiDAR features at each level of the backbone
        self.fusion = MultiScaleFusion(backbone_channels)

    def forward(self, pixel_values, lidar_values, lidar_mask, class_labels=None, mask_labels=None):
        
        # Concatenate LiDAR range values and mask to create a 4-channel input for the LiDAR encoder
        lidar = torch.cat([lidar_values, lidar_mask], dim=1).contiguous()

        backbone = self.model.model.pixel_level_module.encoder

        # Extract multi-scale features from both RGB and LiDAR branches
        rgb_feats = list(backbone(pixel_values).feature_maps)
        lidar_feats = self.lidar_encoder(lidar)
            
        # Align LiDAR features to RGB feature map sizes
        # Necessary because of different handling of padding and downsampling in the two branches
        for i in range(len(rgb_feats)):
            if rgb_feats[i].shape != lidar_feats[i].shape:
                lidar_feats[i] = F.interpolate(
                    lidar_feats[i], 
                    size=rgb_feats[i].shape[2:], 
                    mode='bilinear', 
                    align_corners=False
            )
            
        # Multi-scale fusion of RGB and LiDAR features using gated modulation
        fused_feats = self.fusion(rgb_feats, lidar_feats)

        # Pass fused features through the rest of the Mask2Former model (decoder + transformer)
        pixel_decoder = self.model.model.pixel_level_module.decoder
        decoder_out = pixel_decoder(fused_feats)

        transformer_decoder = self.model.model.transformer_module
        transformer_decoder_outputs = transformer_decoder(
            multi_scale_features=decoder_out.multi_scale_features,
            mask_features=decoder_out.mask_features,
            output_hidden_states=True,
        )
        
        class_queries_logits = ()
        for decoder_output in transformer_decoder_outputs.intermediate_hidden_states:
            class_prediction = self.model.class_predictor(decoder_output.transpose(0, 1))
            class_queries_logits += (class_prediction,)
        
        masks_queries_logits = transformer_decoder_outputs.masks_queries_logits

        auxiliary_logits = self.model.get_auxiliary_logits(class_queries_logits, masks_queries_logits)

        # Compute loss if labels are provided (training mode)
        loss = None
        if mask_labels is not None and class_labels is not None:
            loss_dict = self.model.get_loss_dict(
                masks_queries_logits=masks_queries_logits[-1],
                class_queries_logits=class_queries_logits[-1],
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=auxiliary_logits,
            )
            loss = self.model.get_loss(loss_dict)
            
        # Return final output with loss and logits for the last decoder layer
        output = Mask2FormerForUniversalSegmentationOutput(
            loss=loss,
            class_queries_logits=class_queries_logits[-1],
            masks_queries_logits=masks_queries_logits[-1]
        )

        return output