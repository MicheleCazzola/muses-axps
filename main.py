import os
import json

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from src.config import DATA_ROOT, BASE_SIZE
from src.data.dataloaders import get_dataloaders
from src.evaluate import evaluate, generate_submission
from panopticapi.evaluation import pq_compute
from src.modeling.utils import get_model, load_chp
from src.utils.resources import compute_resource_consumption
from src.train import train
from src.utils.utils import index2id, stuff_classes_ids, TaskType, get_training_optim, get_args

if __name__ == "__main__":
    
    args = get_args()
    
    if args.lidar_mid and args.lidar_early:
        raise ValueError("Cannot use more than one LiDAR fusion method at the same time. Choose only one of --lidar-mid or --lidar-early.")
    
    print("Using device:", args.device)
    
    use_lidar = args.lidar_mid or args.lidar_early
    train_loader, val_loader, test_loader = get_dataloaders(
        DATA_ROOT, args.batch_size, args.resize, args.reduce_factor, use_lidar
    )
    
    model, image_processor = get_model(args.backbone, args.resize, args.lidar_mid, args.lidar_early)
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: '{args.output_dir}'")
    
    if args.mode == TaskType.VALID:
        
        if args.pretrained_path is not None:
            load_chp(args.pretrained_path, model, args.device, sd_only=True)
        
        predictions, img_infos = evaluate(
            model, image_processor, val_loader, BASE_SIZE, stuff_classes_ids, args.device, use_lidar
        )
        
        pred_folder = os.path.join(args.output_dir, "panoptic_valid")
        pred_file = "panoptic"
        valid_result = generate_submission(predictions, img_infos, index2id, save_to=pred_folder, file_name=pred_file)
        
        pq_result = pq_compute(
            gt_json_file=os.path.join(DATA_ROOT, "gt_panoptic", "val.json"),
            pred_json_file=os.path.join(pred_folder, f"{pred_file}.json"),
            gt_folder=os.path.join(DATA_ROOT, "gt_panoptic"),
            pred_folder=pred_folder
        )
        json.dump(pq_result, open(os.path.join(args.output_dir, "pq_results.json"), "w"), indent=4)
        
    elif args.mode == TaskType.TEST:

        if args.pretrained_path is not None:
            load_chp(args.pretrained_path, model, args.device, sd_only=True)
        
        predictions, img_infos = evaluate(
            model, image_processor, test_loader, BASE_SIZE, stuff_classes_ids, args.device, use_lidar
        )
        
        pred_folder = os.path.join(args.output_dir, "panoptic_test")
        pred_file = "panoptic"
        generate_submission(predictions, img_infos, index2id, save_to=pred_folder, file_name=pred_file)
        
    elif args.mode == TaskType.TRAIN:
        
        train_opt = get_training_optim(model, args.lidar_mid, args.lidar_early)
        optimizer, scheduler = map(train_opt.get, ["optimizer", "scheduler"])
        
        if args.pretrained_path is not None:
            print(f"Loading pretrained model from: '{args.pretrained_path}'")
            load_chp(args.pretrained_path, model, optimizer, scheduler, args.device, sd_only=False)
        else:
            print("No pretrained model specified")
                    
        losses, pq_results, best_result, best_pq_result = train(
            model, image_processor, train_loader, val_loader, optimizer, scheduler, args.num_epochs,
            BASE_SIZE, args.device, use_lidar, args.output_dir, log_frequency=5
        )
        
        with open(os.path.join(args.output_dir, "pq_results.json"), "w") as f:
            json.dump(pq_results, f, indent=4)
        with open(os.path.join(args.output_dir, "train_losses.json"), "w") as f:
            json.dump({"train_losses": losses}, f, indent=4)

    elif args.mode == TaskType.MEASURE:
        result = compute_resource_consumption(
            model, use_lidar, args.resize, args.batch_size,
            args.device, num_runs=100, save_to=args.output_dir
        )
    else:
        print("Invalid mode specified. Use --mode valid, test, train or measure.")