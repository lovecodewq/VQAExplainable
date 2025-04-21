# train_bottomup_attention.py

import os
import json
import argparse
from PIL import Image
import logging
from datasets.vg_collator import VGCollator
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import time
from datetime import datetime, timedelta

from models.bottom_up_attention import BottomUpAttention
from datasets.vg import vg  # Import the Visual Genome dataset class
from fast_rcnn.config import cfg  # Import Fast R-CNN config

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

def get_available_device():
    """Check and print all available devices for training"""
    available_devices = []
    device_name = "cpu"
    available_devices.append("cpu")
    
    if torch.cuda.is_available():
        device_name = "cuda"
        available_devices.append("cuda")
        print(f"CUDA is available. Found {torch.cuda.device_count()} CUDA device(s).")
        for i in range(torch.cuda.device_count()):
            print(f"  CUDA:{i} - {torch.cuda.get_device_name(i)}")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_name = "mps"
        available_devices.append("mps")
        print("MPS (Metal Performance Shaders) is available.")
    
    print(f"Available devices: {', '.join(available_devices)}")
    print(f"Default device: {device_name}")
    
    return device_name, available_devices

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vg-version',    default='1600-400-20',
                   help='Visual Genome version (vocabulary size)')
    p.add_argument('--split',         default='train',
                   help='Dataset split to use: train/val/test/minival/minitrain')
    p.add_argument('--val-split',     default='val',
                   help='Validation split to use for evaluation')
    p.add_argument('--out-checkpoint',default='bottomup_vg.pth',
                   help='Where to save the trained model weights')
    p.add_argument('--device',        type=str,   default=None,
                   help='Device to use for training: cpu, cuda, mps (default: auto-detect)')
    p.add_argument('--log-level',     type=str,   default='ERROR',
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                   help='Logging level')
    p.add_argument('--num-workers',   type=int,   default=8,
                   help='Number of workers for data loading')
    p.add_argument('--eval-dir',      default='eval_output',
                   help='Directory to save evaluation results')
    return p.parse_args()

def setup_logging(log_level):
    """Configure logging settings"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def format_time(seconds):
    """Convert seconds to human readable string"""
    return str(timedelta(seconds=int(seconds)))

def evaluate(model, val_loader, device, eval_dir):
    """Evaluate the model on validation set"""
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation")
    
    eval_start_time = time.time()
    model.eval()
    all_boxes = [[[] for _ in range(len(val_loader.dataset))]
                 for _ in range(len(val_loader.dataset._classes))]
    
    inference_time = 0
    with torch.no_grad():
        for i, (imgs, im_info, _) in enumerate(val_loader):
            batch_start = time.time()
            if i % 10 == 0:
                logger.info(f'Processing batch {i}/{len(val_loader)}')
            
            imgs = imgs.to(device)
            im_info = im_info.to(device)
            
            # Forward pass without ground truth (inference mode)
            outputs = model(imgs, im_info)
            inference_time += time.time() - batch_start
            
            # Process detection outputs
            cls_scores = outputs['cls_score'].cpu().numpy()
            bbox_preds = outputs['bbox_pred'].cpu().numpy()
            
            # Store detections
            for j in range(len(imgs)):
                for c in range(1, len(val_loader.dataset._classes)):  # Skip background
                    inds = np.where(cls_scores[j, :] > 0.0)[0]
                    if len(inds) > 0:
                        cls_scores_c = cls_scores[j, inds, c]
                        bbox_preds_c = bbox_preds[j, inds, c*4:(c+1)*4]
                        dets = np.hstack((bbox_preds_c, cls_scores_c[:, np.newaxis]))
                        all_boxes[c][i*cfg.TRAIN.BATCH_SIZE + j] = dets
    
    # Evaluate detections
    os.makedirs(eval_dir, exist_ok=True)
    eval_start = time.time()
    val_loader.dataset.evaluate_detections(all_boxes, eval_dir)
    eval_time = time.time() - eval_start
    
    total_eval_time = time.time() - eval_start_time
    logger.info(f"Evaluation completed in {format_time(total_eval_time)}")
    logger.info(f"  - Inference time: {format_time(inference_time)}")
    logger.info(f"  - Metric computation time: {format_time(eval_time)}")
    
    model.train()
    return all_boxes, total_eval_time

def main():
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Record start time
    training_start_time = time.time()
    
    # Check available devices and set the device
    default_device, available_devices = get_available_device()
    device = args.device if args.device else default_device
    
    # Validate device choice
    if device not in available_devices:
        logger.warning(f"Selected device '{device}' is not available. Using {default_device} instead.")
        device = default_device
    
    logger.info(f"Using device: {device}")
    
    # Initialize datasets
    train_dataset = vg(args.vg_version, args.split)
    val_dataset = vg(args.vg_version, args.val_split)
    
    # Create data loaders
    train_indices = list(range(len(train_dataset.image_index)))
    val_indices = list(range(len(val_dataset.image_index)))
    
    train_collator = VGCollator(train_dataset, device=device)
    val_collator = VGCollator(val_dataset, device=device)
    
    
    batch_size = 8 
    epochs = 10
    # batch_size = cfg.TRAIN.BATCH_SIZE
    # Print dataset and iteration info
    print(f"\nDataset size: {len(train_indices)}")
    print(f"Batch size: {batch_size}")
    print(f"Iterations per epoch: {len(train_indices) // batch_size}")
    
    logger.info("Initializing dataloaders")
    train_loader = DataLoader(
        train_indices,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=args.num_workers,
        pin_memory=(device != 'cuda')
    )
    
    val_loader = DataLoader(
        val_indices,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_collator,
        num_workers=args.num_workers,
        pin_memory=(device != 'cuda')
    )
    logger.info("Dataloaders initialized")

    logger.info("Initializing model")
    # Model & optimizer
    model = BottomUpAttention(
        num_objects=len(train_dataset._classes),
        num_attributes=len(train_dataset._attributes)
    )
    model.train().to(device)
    logger.info("Model initialized")

    # Use Fast R-CNN learning rate settings
    opt = optim.SGD(
        model.parameters(), 
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005
    )

    best_performance = float('inf')  # Track best model performance
    epoch_times = []  # Track time per epoch
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        print(f"\nStarting epoch {epoch}/{epochs}")
        running = {}
        
        # Training phase
        train_start_time = time.time()
        model.train()
        for i, (imgs, im_info, gt_targets) in enumerate(train_loader, 1):
            batch_start_time = time.time()
            
            if i == 1:  # Print only for first batch of each epoch
                print(f"Epoch {epoch}: Processing batch {i}/{len(train_indices) // batch_size}")
            
            imgs = imgs.to(device)
            im_info = im_info.to(device)

            # Forward pass + losses
            losses = model(imgs, im_info, gt_targets)
            total = sum(losses.values())

            opt.zero_grad()
            total.backward()
            opt.step()

            # Accumulate stats
            for k, v in losses.items():
                running[k] = running.get(k, 0.) + v.item()
            
            batch_time = time.time() - batch_start_time
            if i % 5 == 0:
                avg = {k: running[k]/5 for k in running}
                print(f"Epoch {epoch} | iter {i}/{len(train_indices) // batch_size} | " +
                      " | ".join([f"{k}:{avg[k]:.4f}" for k in avg]) +
                      f" | batch_time: {batch_time:.2f}s")
                running.clear()
        
        train_time = time.time() - train_start_time
        logger.info(f"Training epoch {epoch} completed in {format_time(train_time)}")
        
        # Evaluation phase
        logger.info(f"Evaluating epoch {epoch}")
        eval_dir = os.path.join(args.eval_dir, f'epoch_{epoch}')
        all_boxes, eval_time = evaluate(model, val_loader, device, eval_dir)
        
        # Calculate total loss from evaluation (you might want to use a different metric)
        current_performance = sum(running.values()) if running else float('inf')
        
        # Save checkpoint
        checkpoint_path = args.out_checkpoint.replace('.pth', f"_ep{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'performance': current_performance,
        }, checkpoint_path)
        
        # Save best model
        if current_performance < best_performance:
            best_performance = current_performance
            best_checkpoint_path = args.out_checkpoint.replace('.pth', '_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'performance': best_performance,
            }, best_checkpoint_path)
            logger.info(f"New best model saved with performance: {best_performance:.4f}")
        
        # Log timing for this epoch
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        
        logger.info(f"Epoch {epoch} completed in {format_time(epoch_time)}")
        logger.info(f"  - Training time: {format_time(train_time)}")
        logger.info(f"  - Evaluation time: {format_time(eval_time)}")
        logger.info(f"Average epoch time: {format_time(avg_epoch_time)}")
        
        # Estimate remaining time
        remaining_epochs = epochs - epoch
        estimated_remaining_time = avg_epoch_time * remaining_epochs
        logger.info(f"Estimated remaining time: {format_time(estimated_remaining_time)}")
    
    total_training_time = time.time() - training_start_time
    logger.info(f"\nTotal training completed in {format_time(total_training_time)}")
    logger.info(f"Average epoch time: {format_time(sum(epoch_times) / len(epoch_times))}")
    print("=== Done training bottom-up attention ===")

if __name__=="__main__":
    main()