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
import torch.nn.functional as F

from models.bottom_up_attention import BottomUpAttention
from datasets.vg import vg  # Import the Visual Genome dataset class
from fast_rcnn.config import cfg  # Import Fast R-CNN config


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

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model and optimizer state from checkpoint"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return None, 0, float('inf')
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Move optimizer states to correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    start_epoch = checkpoint['epoch'] + 1
    best_performance = checkpoint.get('performance', float('inf'))
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Best performance so far: {best_performance:.4f}")
    
    return start_epoch, best_performance

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
    p.add_argument('--log-level',     type=str,   default='INFO',
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                   help='Logging level')
    p.add_argument('--num-workers',   type=int,   default=8,
                   help='Number of workers for data loading')
    p.add_argument('--eval-dir',      default='eval_output',
                   help='Directory to save evaluation results')
    p.add_argument('--resume',        type=str,   default=None,
                   help='Path to checkpoint to resume training from')
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

def evaluate(epoch, model, val_loader, device, eval_dir):
    print("Starting evaluation")
    t0 = time.time()
    model.eval()

    # get the real dataset & num_images
    vg_dataset = val_loader.collate_fn.dataset
    num_images = len(vg_dataset.image_index)
    num_classes= len(vg_dataset._classes)

    # empty detections: [class][image]→array of Nx5
    all_boxes = [[ np.zeros((0,5),dtype=np.float32)
                   for _ in range(num_images) ]
                 for _ in range(num_classes)]

    inference_time = 0.0

    with torch.no_grad():
        for img_idx in range(num_images):
            print(f"Processing image {img_idx} of {num_images}")
            if img_idx > 10:
                break
            # build a batch of size 1
            imgs, im_info, _ = val_loader.collate_fn([img_idx])
            imgs = imgs.to(device)

            t1 = time.time()
            out = model(imgs)   # returns top-K for this one image
            inference_time += time.time() - t1

            # out['cls_score'] is [K, C] (including bg at 0)
            scores = F.softmax(out['cls_score'], dim=1)     # [K, C]
            scores = scores[:,1:].cpu().numpy()          # drop bg → [K,C-1]
            boxes  = out['boxes'].cpu().numpy()                           # [K, 4]
            K = scores.shape[0]
            for c in range(1, num_classes):
                sc = scores[:, c-1]            # [K]
                inds = np.where(sc>0.0)[0]   # already thresholded by model
                if inds.size:
                    dets = np.hstack([
                        boxes[inds], sc[inds, None]
                    ]).astype(np.float32)      # [N,5]
                else:
                    dets = np.zeros((0,5), dtype=np.float32)
                all_boxes[c][img_idx] = dets
            

    # save & evaluate
    os.makedirs(eval_dir, exist_ok=True)
    metrics = vg_dataset.evaluate_detections(all_boxes, eval_dir)
    t_eval = time.time() - t0

    print(f"Eval done in {t_eval:.1f}s (inference {inference_time:.1f}s)")
    print("mAP:", metrics)

    model.train()
    return -metrics if isinstance(metrics, float) else -np.mean(metrics)


def epoch_training(epoch, epochs, batch_size, num_instances, train_loader, model, device, opt):
    print(f"\nStarting epoch {epoch}/{epochs}")
    running = {}
    
    # Training phase
    train_start_time = time.time()
    model.train()
    for i, (imgs, _, gt_targets) in enumerate(train_loader, 1):
        batch_start_time = time.time()
        
        if i == 1:  # Print only for first batch of each epoch
            print(f"Epoch {epoch}: Processing batch {i}/{num_instances // batch_size}")
        
        imgs = imgs.to(device)

        # Forward pass + losses
        losses = model(imgs, gt_targets)
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
            print(f"Epoch {epoch} | iter {i}/{num_instances // batch_size} | " +
                    " | ".join([f"{k}:{avg[k]:.4f}" for k in avg]) +
                    f" | batch_time: {batch_time:.2f}s")
            running.clear()
    
    train_time = time.time() - train_start_time
    print(f"Training epoch {epoch} completed in {format_time(train_time)}")

def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
        
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
    print(f"Using device: {device}")
    
    # Initialize datasets
    train_dataset = vg(args.vg_version, args.split)
    val_dataset = vg(args.vg_version, args.val_split)
    
    # Create data loaders
    train_indices = list(range(len(train_dataset.image_index)))
    val_indices = list(range(len(val_dataset.image_index)))
    
    train_collator = VGCollator(train_dataset, device=device)
    val_collator = VGCollator(val_dataset, device=device)
    
    
    batch_size = 10
    epochs = 5
    # Print dataset and iteration info
    print(f"Validation dataset size: {len(val_indices)}")
    print(f"Train dataset size: {len(train_indices)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Iterations per epoch: {len(train_indices) // batch_size}")
    
    print("Initializing dataloaders")
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
    print("Dataloaders initialized")

    print("Initializing model")
    # Model & optimizer
    model = BottomUpAttention(
        num_objects=len(train_dataset._classes),
        num_attributes=len(train_dataset._attributes)
    )
    model.to(device)
    
    # Use Fast R-CNN learning rate settings
    opt = optim.SGD(
        model.parameters(), 
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Initialize tracking variables
    start_epoch = 1
    best_performance = float('inf')
    
    # Load checkpoint if specified
    if args.resume:
        start_epoch, best_performance = load_checkpoint(model, opt, args.resume, device)
        if start_epoch is None:  # Loading failed
            return
        print(f"Resuming training from epoch {start_epoch}")

    # model = torch.compile(model, backend="inductor") # no gain for cpu
    print("Model initialized")
    
    epoch_times = []  # Track time per epoch
    epoch_start_time = time.time()
    for epoch in range(start_epoch, epochs + 1):
        epoch_training(epoch, epochs, batch_size, len(train_indices), train_loader, model, device, opt)

        # Evaluation phase
        # current_performance = evaluate(epoch, model, val_loader, device, args.eval_dir)
        
        # Save checkpoint
        checkpoint_path = args.out_checkpoint.replace('.pth', f"_ep{epoch}.pth")
        save_checkpoint(model, opt, epoch, checkpoint_path)
        
        # Save best model (note: we use negative mAP so lower is better)
        # if current_performance < best_performance:
        #     best_performance = current_performance
        #     best_checkpoint_path = args.out_checkpoint.replace('.pth', '_best.pth')
        #     save_checkpoint(model, opt, epoch, best_performance, best_checkpoint_path)
        #     print(f"New best model saved with mAP: {-best_performance:.4f}")

        # Log timing for this epoch
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        
        print(f"Epoch {epoch} completed in {format_time(epoch_time)}")
        print(f"Average epoch time: {format_time(avg_epoch_time)}")
        
        # Estimate remaining time
        remaining_epochs = epochs - epoch
        estimated_remaining_time = avg_epoch_time * remaining_epochs
        print(f"Estimated remaining time: {format_time(estimated_remaining_time)}")
    
    total_training_time = time.time() - training_start_time
    print(f"\nTotal training completed in {format_time(total_training_time)}")
    print(f"Average epoch time: {format_time(sum(epoch_times) / len(epoch_times))}")
    print("=== Done training bottom-up attention ===")

if __name__=="__main__":
    main()