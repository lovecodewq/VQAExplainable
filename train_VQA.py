import os
import sys

import torch

import argparse
import yaml
from models.VQA_model import *
from models.bottom_up_attention import *

# from models.bottom_up_attention import BottomUpAttention
from datasets.vg import vg
import argparse
import torch
import numpy as np
import os
from datasets.vg_collator import VGCollator
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet101_Weights
from torchvision.models._utils import IntermediateLayerGetter

def train_vqa():
    parser = argparse.ArgumentParser()
    default_config_file = os.path.join(os.path.abspath(os.getcwd()), "./config/VQA_training_config.yaml")
    dataset_path = os.path.join(os.path.abspath(os.getcwd()), "./data/visual_genome/1600-400-20")

    parser.add_argument("--config", type=str, default=default_config_file, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--output-dir", type=str, default="bottom_up_features", required=False)
    parser.add_argument("--feature_type", type=str, default="bottom_up", required=False)
    parser.add_argument("--device", type=str, default="cpu", required=False)
    parser.add_argument("--num-workers", type=int, default=8, required=False)
    parser.add_argument('--vg_version',    default=dataset_path,
                   help='Visual Genome version (vocabulary size)')
    parser.add_argument('--split',         default='train',
                   help='Dataset split to use: train/val/test/minival/minitrain')
    args = parser.parse_args()
    config_file = args.config
    with open(config_file, "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

    # Hyperparamters
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    input_vocab_size = config["input_vocab_size"]
    answer_vocab_size = config["answer_vocab_size"]
    max_token_length = config["max_token_length"]
    image_feature_dim = config["image_feature_dim"]
    k = config["k"] # number of feature regions

    # Create output directory if it doesn't exist
    if args.feature_type == "bottom_up":
        bottom_up_model = BottomUpAttention()
        # model.load_state_dict(torch.load(args.checkpoint))
        # model.eval()
    else:
        raise ValueError(f"Invalid feature type: {args.feature_type}")

    # Load the dataset
    dataset = vg(args.vg_version, args.split)
    print("dataset size: ", len(dataset.image_index))
    indices = list(range(len(dataset.image_index)))
    collator = VGCollator(dataset, device=args.device)
    dataloader = DataLoader(
        indices,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=(args.device != 'cuda')
    )

    vqa_model = VQAModel(input_vocab_size, answer_vocab_size)
    optimizer = torch.optim.Adam(vqa_model.parameters(),lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, (imgs, im_info, gt_targets) in enumerate(dataloader, 1):
            if args.feature_type == "bottom_up":
                image_features = bottom_up_model(imgs, gt_targets)
                print("image feature: ", image_features.shape)
                rand_question_tokens = torch.randint(0, max_token_length, (batch_size, max_token_length))
                rand_targets = torch.randn(batch_size, answer_vocab_size)
                logits = vqa_model(rand_question_tokens, image_features)
                loss = vqa_loss_fn(logits, rand_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"average loss at epoch {epoch} is {total_loss/100}")

if __name__ == "__main__":
    train_vqa()
