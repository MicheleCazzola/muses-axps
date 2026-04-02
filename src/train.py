import os
from time import time
import torch
import numpy as np

from panopticapi.evaluation import pq_compute

from src.modeling.utils import save_chp
from src.evaluate import evaluate, generate_submission, save_submission

    
# Helper function to preprocess the panoptic segmentation targets into the format required for Mask2Former training
def preprocess_targets(targets, id2index):
    masks, classes = [], []
    for tgt in targets:
        panoptic_mask, segment_info = tgt["mask"], tgt["segments_info"]
        segment_ids = np.unique(panoptic_mask.numpy())
        segment_ids = segment_ids[segment_ids != 0]  # background ID is 0
        
        image_masks, image_classes = [], []
        for seg_id in segment_ids:
            binary_mask = (panoptic_mask == seg_id).float()

            if binary_mask.sum() == 0:
                continue

            image_masks.append(binary_mask)
            image_classes.append(id2index[segment_info[int(seg_id)]])
    
        masks.append(torch.stack(image_masks))
        classes.append(torch.tensor(image_classes))
    
    return masks, classes

def validate(model, image_processor, dataloader, size, device, lidar, stuff_classes_ids, index2id):
    
    pred_folder, file_name = "./temp/panoptic", "panoptic"
    
    # Run validation inference to get predictions and image info
    predictions, img_infos = evaluate(
        model, image_processor, dataloader, size, stuff_classes_ids, device, lidar
    )
    
    # Generate panoptic results to be compatible with the PQ evaluation code
    valid_result = generate_submission(predictions, img_infos, index2id, save_to=pred_folder, file_name=file_name)
    
    # Compute PQ metric on the current validation predictions
    pq_result = pq_compute(
        gt_json_file="./data/muses/gt_panoptic/val.json",
        pred_json_file=os.path.join(pred_folder, f"{file_name}.json"),
        gt_folder="./data/muses/gt_panoptic",
        pred_folder=pred_folder
    )
    
    return valid_result, pq_result


def train(
    model, image_processor, train_loader, val_loader, optimizer, scheduler,
    num_epochs, size, device, lidar, out_folder, stuff_classes_ids, index2id, id2index, log_frequency=5
):
    
    out_folder_panoptic = os.path.join(out_folder, "panoptic_valid")
    
    model = model.to(device, memory_format=torch.contiguous_format)
    
    best_pq_result, best_result = None, None
    train_losses, pq_results = [], []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_iter, tot_iter, num_samples = 0, len(train_loader), 0
        
        start = time()
        for images, targets in train_loader:
            
            num_samples += images.size(0)
            
            # Preprocess images and targets
            processed_images = image_processor(images=images, return_tensors="pt", device=device)
            masks, classes = preprocess_targets(targets, id2index)
            
            inputs = {
                "pixel_values": processed_images["pixel_values"].to(device),
                "mask_labels": [mask.to(device) for mask in masks],
                "class_labels": [cls.to(device) for cls in classes]
            }
            
            # Add LiDAR inputs if using LiDAR mid-fusion or early fusion
            if lidar:
                lidar_values = torch.stack([tgt["lidar_values"] for tgt in targets], dim=0).contiguous().to(device)
                lidar_mask = torch.stack([tgt["lidar_mask"] for tgt in targets], dim=0).contiguous().to(device)
                
                inputs["lidar_values"] = lidar_values
                inputs["lidar_mask"] = lidar_mask
            
            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss
            
            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            num_iter += 1
            
            if num_iter % log_frequency == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Iteration {num_iter}/{tot_iter} | Loss: {total_loss / num_samples:.4f}")

        # Step the scheduler at the end of the epoch
        if scheduler is not None:
            scheduler.step()

        # Compute average loss for the epoch
        epoch_loss = total_loss / num_samples
        train_losses.append(epoch_loss)
        
        # Validation 
        valid_result, pq_result = validate(model, image_processor, val_loader, size, device, lidar, stuff_classes_ids, index2id)
        pq_results.append(pq_result)
        
        # Save best checkpoint based on PQ metric
        if best_pq_result is None or pq_result["All"]["pq"] > best_pq_result["All"]["pq"]:
            best_pq_result = pq_result
            best_result = valid_result
            
            save_chp(os.path.join(out_folder, "best_model.pth.tar"), model, optimizer, scheduler, epoch + 1, best_pq_result)
            save_submission(out_folder_panoptic, "panoptic", best_result["result"], best_result["rgb_masks"])
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f}")
        print(f"Validation PQ: {pq_result['All']['pq']:.4f}, RQ: {pq_result['All']['rq']:.4f}, SQ: {pq_result['All']['sq']:.4f}")
        print(f"Elapsed time: {time() - start:.2f} seconds")
    
    # Save last checkpoint at the end of training
    save_chp(os.path.join(out_folder, "last_model.pth.tar"), model, optimizer, scheduler, epoch + 1, pq_results[-1])
        
    return train_losses, pq_results, best_result, best_pq_result