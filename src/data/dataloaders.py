import torch
from torch.utils.data import DataLoader

import albumentations as A

# ==========================================================
# DATASET
# ==========================================================

from src.data.dataset import MUSESPanopticDataset
from src.config import DATA_ROOT, RESIZE_MODEL, BATCH_SIZE, NUM_WORKERS
from src.utils.setup import set_reproducibility

g, worker_init_fn = set_reproducibility()

def collate_fn(batch):
    images = []
    targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    return torch.stack(images), targets

def define_transforms(splits, resize):
    resize_transform = A.Resize(height=resize[0], width=resize[1])
    
    train_transform = A.Compose(
        [resize_transform], 
        additional_targets={"mask": "mask", "lidar_values": "mask", "lidar_mask": "mask"},
        seed=42
    )
    val_transform = A.Compose(
        [resize_transform],
        seed=42
    )
    
    return dict(zip(splits, [train_transform, val_transform, val_transform]))

def make_dataset(data_root, split, transform, use_lidar, reduce_factor):
    dataset = MUSESPanopticDataset(
        root=data_root,
        image_folder="frame_camera",
        lidar_folder="lidar",
        gt_folder="gt_panoptic" if split != "test" else None,  # No GT folder for test set
        split=split,
        transform=transform,
        use_lidar=use_lidar,
        reduce_factor=reduce_factor
    )
    return dataset

def make_dataloader(dataset, batch_size, shuffle, drop_last):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        generator=g,
        collate_fn=collate_fn,
    )
    return dataloader

def get_dataloaders(data_root=DATA_ROOT, batch_size=BATCH_SIZE, resize=RESIZE_MODEL, reduce_factor=None, lidar=False):
    
    splits = ["train", "val", "test"]

    # Define transforms for each split
    transforms = define_transforms(splits, resize)
    
    # ==========================================================
    # DATASETS
    # ==========================================================
    datasets = {
        split: make_dataset(
            data_root=data_root,
            split=split,
            transform=transforms[split],
            use_lidar=lidar,
            reduce_factor=reduce_factor
        ) for split in splits
    }   

    # ==========================================================
    # DATALOADERS
    # ==========================================================
    
    # Batch sizes for each split -> Use smaller batch size for val/test when using LiDAR to avoid OOM
    batch_sizes = {
        "train": batch_size,
        "val": batch_size if not lidar else 1,
        "test": batch_size if not lidar else 1
    }
    
    # Shuffle and drop_last settings for each split -> True only for training
    shuffle = dict(zip(splits, [True, False, False]))
    drop_last = dict(zip(splits, [True, False, False]))
    
    dataloaders = {
        split: make_dataloader(
            dataset=datasets[split],
            batch_size=batch_sizes[split],
            shuffle=shuffle[split],
            drop_last=drop_last[split]
        ) for split in splits
    }
    
    train_loader, val_loader, test_loader = map(dataloaders.get, splits)

    return train_loader, val_loader, test_loader