
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
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def dataloader(img_feature_dir, questions_file, annotations_file, answer_to_idx, batch_size, max_token_length):
    data = VQADataset(
        img_feature_dir,
        questions_file,
        annotations_file,
        answer_to_idx,
        max_length=max_token_length
    )
    return DataLoader(data, batch_size=batch_size, shuffle=True)

def save_training_plot(acc_list, loss_list, val_acc_list, val_loss_list, num_epochs, filename, save_dir="./"):
    epochs = list(range(1, num_epochs + 1))
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(epochs, loss_list, label='Train')
    plt.plot(epochs, val_loss_list, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.savefig(os.path.join(save_dir, f"{filename}_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, [acc * 100 for acc in acc_list], label='Train')
    plt.plot(epochs, [acc * 100 for acc in val_acc_list], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.savefig(os.path.join(save_dir, f"{filename}_accuracy.png"))
    plt.close()

def train_and_validate(vqa_model, train_loader, val_loader, optimizer, device, idx_to_answer, num_epochs, update_step = [6, 12]):
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    acc_list = []
    loss_list = []
    val_acc_list = []
    val_loss_list = []
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(train_loader)
        if epoch in update_step:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
                print(f"Current learning rate: {param_group['lr']}")
        else:
            for param_group in optimizer.param_groups:
                print(f"Current learning rate: {param_group['lr']}")
        vqa_model.train()
        for batch_idx, (image_features, question_embeddings, answers_idx, answers) in enumerate(train_loader):
            image_features = image_features.to(device)
            question_embeddings = question_embeddings.to(device)
            answers_idx = answers_idx.to(device)
            logits = vqa_model(question_embeddings, image_features)
            loss = criterion(logits, answers_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            preds = preds.cpu().numpy()
            pred_answers = [idx_to_answer[pred_idx] for pred_idx in preds]
            answers = np.array(answers)
            correct += (pred_answers == answers).sum()
            total += len(pred_answers)

            if (batch_idx % 10 == 0):
                print(f"Epoch {epoch}, Batch {batch_idx + 1}/{total_batches}, Loss: {loss.item():.4f}")

        print(f"Average training loss at epoch {epoch} is {total_loss/total_batches:.4f}")
        print(f"Average training accuracy at epoch {epoch} is {correct/total:.4f}")
        print(f"Time at epoch {epoch}: {time.time() - epoch_start_time:.4f} seconds")

        acc_list.append(correct/total)
        loss_list.append(total_loss/total_batches)
    
        # Validation step
        vqa_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for image_features, question_embeddings, answers_idx, answers in val_loader:
                image_features = image_features.to(device)
                question_embeddings = question_embeddings.to(device)
                answers_idx = answers_idx.to(device)

                logits = vqa_model(question_embeddings, image_features)
                loss = criterion(logits, answers_idx)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                preds = preds.cpu().numpy()
                pred_answers = [idx_to_answer[pred_idx] for pred_idx in preds]
                answers = np.array(answers)
                val_correct += (pred_answers == answers).sum()
                val_total += len(pred_answers)

        print(f"Average validation loss is {val_loss/len(val_loader):.4f}")
        print(f"Average validation accuracy is {val_correct/val_total:.4f}")
        val_acc_list.append(val_correct/val_total)
        val_loss_list.append(val_loss/len(val_loader))
    
    print(f"Total Time: {time.time() - start_time:.4f} seconds")

    return val_acc_list[len(val_acc_list)-1], acc_list, loss_list, val_acc_list, val_loss_list


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
    
    max_token_length = config["max_token_length"]
    image_feature_dim = config["image_feature_dim"]
    question_dim = config["question_dim"]
    hidden_dim = config["hidden_dim"]
    img_feature_dir_train = 'VQA_data/train2014_image_feats'
    img_feature_dir_val = 'VQA_data/val2014_image_feats'
    questions_file_train = 'VQA_data/v2_OpenEnded_mscoco_train2014_questions.json'
    annotations_file_train = 'VQA_data/v2_mscoco_train2014_annotations.json'
    questions_file_val = 'VQA_data/v2_OpenEnded_mscoco_val2014_questions.json'
    annotations_file_val = 'VQA_data/v2_mscoco_val2014_annotations.json'

    answer_to_idx, idx_to_answer = create_answer_idx(annotations_file_train, annotations_file_val)
    print("Number of frequent answers in datasets:", len(idx_to_answer))
    train_loader = dataloader(img_feature_dir_train, questions_file_train, annotations_file_train, answer_to_idx, batch_size, max_token_length)
    val_loader = dataloader(img_feature_dir_val, questions_file_val, annotations_file_val, answer_to_idx, batch_size, max_token_length)
    print("Training and validation dataset loading finished")
    answer_vocab_size = len(answer_to_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training one model
    vqa_model = VQAModel(answer_vocab_size, image_feature_dim=image_feature_dim, question_dim=question_dim, hidden_dim=hidden_dim)
    vqa_model = vqa_model.to(device)
    optimizer = torch.optim.Adam(vqa_model.parameters(),lr=lr)

    val_acc, acc_list, loss_list, val_acc_list, val_loss_list = train_and_validate(vqa_model, train_loader, val_loader, optimizer, device, idx_to_answer, num_epochs)
    filename = f"VQA_{lr}_{question_dim}_{hidden_dim}_{num_epochs}"
    save_training_plot(acc_list, loss_list, val_acc_list, val_loss_list, num_epochs, filename, save_dir="./plots")

    save_dir = "saved_VQA_models/"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(vqa_model.state_dict(), os.path.join(save_dir, f"{filename}.pth"))

if __name__ == "__main__":
    train_vqa()
