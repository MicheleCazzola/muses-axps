import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.utils import id2rgb

@torch.no_grad()
def evaluate(model, image_processor, dataloader, size, stuff_classes_ids, device, lidar=False, log_frequency=3):
    
    model.eval()
    print("Running validation inference...")

    predictions, img_infos = [], []
    num_iter, tot_iter = 0, len(dataloader)
    for images, targets in dataloader:
        
        img_info = [{"image_id": t["image_id"], "image_file_name": t["image_file_name"]} for t in targets]
        
        # Pre-process images
        processed_images = image_processor(images=images, return_tensors="pt", device=device)
        inputs = {"pixel_values": processed_images["pixel_values"].to(device)}
        
        # Add LiDAR data to inputs if specified
        if lidar:
            lidar_values = torch.stack([tgt["lidar_values"] for tgt in targets], dim=0).contiguous().to(device)
            lidar_mask = torch.stack([tgt["lidar_mask"] for tgt in targets], dim=0).contiguous().to(device)
            
            inputs["lidar_values"] = lidar_values
            inputs["lidar_mask"] = lidar_mask
        
        # Forward pass
        outputs = model(**inputs)
        
        # Post-process model outputs to get panoptic segmentation maps
        pred_maps = image_processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[size] * images.shape[0], label_ids_to_fuse=stuff_classes_ids
        )
        
        predictions.extend(pred_maps)
        img_infos.extend(img_info)
        
        num_iter += 1
        if num_iter % log_frequency == 0:
            print(f"Validation | Iteration {num_iter}/{tot_iter}")
        
    return predictions, img_infos


# Helper function to save panoptic segmentation results (both JSON annotations and RGB masks) to disk
def save_submission(folder, file_name, result, rgb_masks):
    if folder is not None and file_name is not None:
        os.makedirs(folder, exist_ok=True)
        for record, rgb_mask in tqdm(zip(result, rgb_masks)):
            
            # Save the mask as a PNG file
            rgb_mask_pil = Image.fromarray(rgb_mask)
            rgb_mask_pil.save(os.path.join(folder, f"{record['file_name']}"))
        
        # Save the JSON file with annotations
        json_result = json.dumps({"annotations": result}, indent=4)
        with open(os.path.join(folder, f"{file_name}.json"), "w") as f:
            f.write(json_result)


# Generate panoptic results from model predictions and save them to disk in the format required for PQ evaluation
def generate_submission(predictions, img_infos, index2id, save_to="panoptic", file_name="panoptic"):
    
    print(f"Saving predictions to '{save_to}'")
    
    result, masks, rgb_masks = [], [], []
    for pred, img_info in tqdm(zip(predictions, img_infos)):
        mask, segments_info = pred["segmentation"], pred["segments_info"]
        
        # Convert the panoptic segmentation map to RGB format for visualization and saving
        rgb_mask = id2rgb(mask)
        
        # Create a record for the JSON annotation file
        record = {
            "image_id": img_info["image_id"],
            "file_name": img_info["image_id"] + "_frame_camera.png",
            "segments_info": [{"id": seg["id"], "category_id": index2id[seg["label_id"]]} for seg in segments_info]
        }
        
        result.append(record)
        rgb_masks.append(rgb_mask.cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
        masks.append(mask.cpu().numpy().astype(np.uint32))
       
    # Save the results to disk (both JSON annotations and RGB masks) 
    save_submission(save_to, file_name, result, rgb_masks)
    
    return {
        "result": result,
        "masks": masks,
        "rgb_masks": rgb_masks
    }