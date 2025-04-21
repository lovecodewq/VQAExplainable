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

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vg-version',    default='1600-400-20',
                   help='Visual Genome version (vocabulary size)')
    p.add_argument('--split',         default='train',
                   help='Dataset split to use: train/val/test/minival/minitrain')
    p.add_argument('--out-checkpoint',default='bottomup_vg.pth',
                   help='Where to save the trained model weights')
    p.add_argument('--device',        type=str,   default=None,
                   help='Device to use for training: cpu, cuda, mps (default: auto-detect)')
    p.add_argument('--log-level',     type=str,   default='ERROR',
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                   help='Logging level')
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

def main():
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Check available devices and set the device
    default_device, available_devices = get_available_device()
    device = args.device if args.device else default_device
    
    # Validate device choice
    if device not in available_devices:
        logger.warning(f"Selected device '{device}' is not available. Using {default_device} instead.")
        device = default_device
    
    logger.info(f"Using device: {device}")
    
    # Initialize Visual Genome dataset
    dataset = vg(args.vg_version, args.split)
    
    # Create data loader using indices with a custom collate function
    indices = list(range(len(dataset.image_index)))
    
    # Create a collate function specific to this dataset using our picklable class
    collator = VGCollator(dataset, device=device)
    
    # Adjust num_workers based on device
    if device == 'mps':
        num_workers = 0
        logger.warning("MPS device detected. Setting num_workers to 0 for compatibility.")
    else:
        # Limit number of workers to system suggestion (8)
        num_workers = min(8, cfg.TRAIN.RPN_BATCHSIZE)
        if cfg.TRAIN.RPN_BATCHSIZE > 8:
            logger.warning(f"Reducing number of workers from {cfg.TRAIN.RPN_BATCHSIZE} to 8 to avoid performance issues.")
    
    batch_size = 8 
    epochs = 10
    # batch_size = cfg.TRAIN.BATCH_SIZE
    # Print dataset and iteration info
    print(f"\nDataset size: {len(indices)}")
    print(f"Batch size: {batch_size}")
    print(f"Iterations per epoch: {len(indices) // batch_size}")
    
    logger.info("Initializing dataloader")
    loader = DataLoader(
        indices,
        # batch_size=cfg.TRAIN.BATCH_SIZE,
        batch_size=8,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=(device != 'cpu')
    )
    logger.info("Dataloader initialized")

    logger.info("Initializing model")
    # Model & optimizer
    model = BottomUpAttention(
        num_objects=len(dataset._classes),
        num_attributes=len(dataset._attributes)
    )
    model.train().to(device)
    logger.info("Model initialized")

    # Use Fast R-CNN learning rate settings
    opt = optim.SGD(
        model.parameters(), 
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005  # Standard Fast R-CNN weight decay
    )

    for epoch in range(1, epochs):
        print(f"\nStarting epoch {epoch}/{epochs}")
        running = {}
        for i, (imgs, im_info, gt_targets) in enumerate(loader, 1):
            if i == 1:  # Print only for first batch of each epoch
                print(f"Epoch {epoch}: Processing batch {i}/{len(indices) // batch_size}")
            
            imgs = imgs.to(device)
            im_info = im_info.to(device)

            # Forward pass + losses
            losses = model(imgs, im_info, gt_targets)  # returns loss dict
            total = sum(losses.values())

            opt.zero_grad()
            total.backward()
            opt.step()

            # Accumulate stats
            for k, v in losses.items():
                running[k] = running.get(k, 0.) + v.item()
            if i % 5 == 0:
                avg = {k: running[k]/5 for k in running}
                print(f"Epoch {epoch} | iter {i}/{len(indices) // batch_size} | " +
                      " | ".join([f"{k}:{avg[k]:.4f}" for k in avg]))
                running.clear()
        # Save at end of epoch
        torch.save(model.state_dict(), args.out_checkpoint.replace('.pth', f"_ep{epoch}.pth"))

    # Final checkpoint
    torch.save(model.state_dict(), args.out_checkpoint)
    print("=== Done training bottom-up attention ===")

if __name__=="__main__":
    main()