import os
import json
import random
from PIL import Image
import numpy as np
import torch
from torchvision.datasets import VisionDataset

from src.data.muses_sdk import load_lidar_projection, load_muses_calibration_data, load_meta_data

class MUSESPanopticDataset(VisionDataset):
    """
    MUSES dataset loader for panoptic segmentation.
    Reads images and panoptic masks from the JSON annotation file.
    """
    def __init__(self, root: str, image_folder, lidar_folder, gt_folder, split, transform=None, target_transform=None, use_lidar=False, reduce_factor=None):
        """
        Args:
            root: root directory of muses dataset
            image_folder: subfolder name for images (e.g., "images")
            lidar_folder: subfolder name for LiDAR data (e.g., "lidar")
            gt_folder: subfolder name for ground truth (e.g., "gt_panoptic")
            split: "train", "val" or "test"
            transform: optional transform for RGB image
            target_transform: optional transform for panoptic mask
            use_lidar: whether to load and return LiDAR data
            reduce_factor: if not None, reduces dataset size by keeping only every N-th sample
        """
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.image_folder = image_folder
        self.lidar_folder = lidar_folder
        self.gt_folder = gt_folder
        self.calib_data = load_muses_calibration_data(root)
        self.meta_data = load_meta_data(root)
        self.use_lidar = use_lidar
        self.reduce_factor = reduce_factor

        # Path to JSON annotation
        if self.gt_folder is not None:
            self.data_info_path = os.path.join(root, self.gt_folder, f"{split}.json")
        else:
            self.data_info_path = os.path.join(root, "gt_panoptic", f"test_image_info.json")  # Use GT JSON for test set without GT masks

        # Load JSON data
        with open(self.data_info_path, "r") as f:
            self.data_info = json.load(f)
            
        # Create mapping from image_id to annotation for quick lookup
        if self.gt_folder is not None:
            self.image_id_to_ann = {ann["image_id"]: ann for ann in self.data_info["annotations"]}
        else:
            self.image_id_to_ann = {}  # Empty mapping for test set without GT masks

        self.categories = self.data_info["categories"]
        self.images = self.data_info["images"]
        
        # Reduce dataset size if reduce_factor is specified
        if self.reduce_factor is not None and (1 < self.reduce_factor < len(self.images)):
            self.reduce(self.reduce_factor)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image_path = os.path.join(self.root, self.image_folder, img_info["file_name"])
        lidar_path = os.path.join(self.root, self.lidar_folder, img_info["lidar_file_name"])
        
        # Load RGB image
        img = np.array(Image.open(image_path).convert("RGB"))
        
        if self.gt_folder:
            segments_info = {seg_info["id"]: seg_info["category_id"] for seg_info in self.image_id_to_ann[img_info["id"]]["segments_info"]}
            panoptic_path = os.path.join(self.root, self.gt_folder, self.image_id_to_ann[img_info["id"]]["file_name"])
            mask = np.array(Image.open(panoptic_path).convert("RGB"))
        else:
            panoptic_path, segments_info = None, None
            mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)  # Dummy mask for test set without GT
        
        if self.use_lidar:
            # Load LiDAR
            lidar, validity_mask = load_lidar_projection(
                lidar_path=lidar_path,
                calib_data=self.calib_data,
                scene_meta_dict=self.meta_data[img_info["id"]],
                motion_compensation=True,
                muses_root=self.root,
                target_shape=(1920, 1080),
                enlarge_lidar_points=True
            )
            
            lidar_range, lidar_intensity, lidar_height = np.split(lidar, 3, axis=2)  # Separate channels
            lidar_mask = (np.isfinite(lidar_range) & (lidar_range > 0)).astype(np.float32)  # Mask of valid LiDAR point
            
            # Range
            range_max = np.quantile(lidar_range[lidar_mask == 1], 0.99)  # Get 99th percentile of valid points
            lidar_range = np.clip(lidar_range, 0, range_max) / range_max 
            lidar_range = lidar_range * lidar_mask  # Mask out invalid points
            
            # Intensity normalization: normalize to [0, 1] and mask out invalid points
            lidar_intensity = lidar_intensity / 255.0  # Normalize intensity to [0, 1]
            lidar_intensity = lidar_intensity * lidar_mask  # Mask out invalid points
            
            # Height normalization: clip to 1st and 99th percentile of valid points, then normalize to [0, 1]
            h_min = np.quantile(lidar_height[lidar_mask == 1], 0.01)
            h_max = np.quantile(lidar_height[lidar_mask == 1], 0.99)
            lidar_height = np.clip((lidar_height - h_min) / (h_max - h_min), 0, 1)
            lidar_height = lidar_height * lidar_mask 

            lidar_values = np.concatenate([lidar_range, lidar_intensity, lidar_height], axis=2)  # (H, W, 3)
            
            # Mark enlarged points as less reliable (0.5) than original valid points (1.0)
            lidar_mask = 0.5 * lidar_mask + 0.5 * (np.expand_dims(validity_mask, axis=2)).astype(np.float32)  
        else:
            # Create dummy LiDAR data (all zeros) if not using LiDAR, they will be ignored in training and inference
            lidar_values = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
            lidar_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)
        
        if self.transform:
            transformed = self.transform(image=img, mask=mask, lidar_values=lidar_values, lidar_mask=lidar_mask)
            img = transformed["image"]
            mask = transformed["mask"]
            lidar_values = transformed["lidar_values"]
            lidar_mask = transformed["lidar_mask"]

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).permute(2, 0, 1)
        lidar_values = torch.from_numpy(lidar_values).permute(2, 0, 1)
        lidar_mask = torch.from_numpy(lidar_mask).permute(2, 0, 1)
        
        # Convert panoptic RGB mask to 2D array of segment IDs (segment_id = R + G*256 + B*256^2)
        mask = mask[0, :, :].to(torch.int32) + mask[1, :, :].to(torch.int32) * 256 + mask[2, :, :].to(torch.int32) * 256**2
        
        # Return dictionary with image, mask (2D segment IDs), lidar (if used), and other data info
        return img, {
            "mask": mask,
            "lidar_values": lidar_values,
            "lidar_mask": lidar_mask,
            "image_id": img_info["id"],
            "image_file_name": img_info["file_name"],
            "segments_info": segments_info
        }
        
    def reduce(self, factor):
        """
        Reduce dataset size by keeping only every N-th sample.
        """
        if self.reduce_factor is None:
            self.reduce_factor = factor
        
        random.shuffle(self.images)  # Shuffle before reducing
        self.images = self.images[::self.reduce_factor]
        self.image_id_to_ann = (
            {img["id"]: self.image_id_to_ann[img["id"]] for img in self.images}
            if self.gt_folder is not None
            else {}
        )