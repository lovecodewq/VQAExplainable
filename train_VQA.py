
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.image_list import ImageList

import argparse
import numpy as np
import os
import sys
import yaml

from datasets.vqa_dataset import *
from models.VQA_model import *

from torch.utils.data import DataLoader
import glob


def dataloader(img_feature_dir, questions_file, annotations_file, batch_size):
    # Load the dataset.
    annotations_file_train = 'data/v2_mscoco_train2014_annotations.json'
    annotations_file_val = 'data/v2_mscoco_val2014_annotations.json'

    answer_to_idx, idx_to_answer = create_answer_idx(None, annotations_file_val)
    print("Number of answers in val datasets: ", len(idx_to_answer))

    val_dataset = VQADataset(
        img_feature_dir,
        questions_file,
        annotations_file,
        answer_to_idx,
        # max_length=23
    )
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

def train_vqa():
    parser = argparse.ArgumentParser()
    default_config_file = os.path.join(os.path.abspath(os.getcwd()), "./config/VQA_training_config.yaml")
    parser.add_argument("--config", type=str, default=default_config_file, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--output-dir", type=str, default="bottom_up_features", required=False)
    parser.add_argument("--feature_type", type=str, default="bottom_up", required=False)
    parser.add_argument("--device", type=str, default="cpu", required=False)
    parser.add_argument("--num-workers", type=int, default=8, required=False)

    args = parser.parse_args()
    config_file = args.config
    with open(config_file, "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print("Configs: ", config)

    # Hyperparamters
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    input_vocab_size = config["input_vocab_size"]
    answer_vocab_size = config["answer_vocab_size"]
    max_token_length = config["max_token_length"]
    image_feature_dim = config["image_feature_dim"]
    k = config["k"] # number of feature regions

    img_feature_dir_val = 'data/val2014_first1000'
    questions_file_train = 'data/v2_OpenEnded_mscoco_train2014_questions.json'
    annotations_file_train = 'data/v2_mscoco_train2014_annotations.json'
    questions_file_val = 'data/v2_OpenEnded_mscoco_val2014_questions.json'
    annotations_file_val = 'data/v2_mscoco_val2014_annotations.json'

    val_loader = dataloader(img_feature_dir_val, questions_file_val, annotations_file_val,  batch_size)
    print("training dataset loading finished...")

    # TODO: Add validation step
    # TODO: Check input_vocab_size
    vqa_model = VQAModel(input_vocab_size, answer_vocab_size)
    optimizer = torch.optim.Adam(vqa_model.parameters(),lr=lr)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for image_features, question_embeddings, answers in val_loader:
            # print("image_features: ", image_features.shape)
            # print("question_embeddings: ", question_embeddings.shape)
            # print("answers: ", answers.shape)
            logits = vqa_model(question_embeddings, image_features[:,:,:2048])
            # print("logits: ", logits.shape)
            loss = criterion(logits, answers)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print("loss: ", loss)
        print(f"average loss at epoch {epoch} is {total_loss/100}")

if __name__ == "__main__":
    train_vqa()
