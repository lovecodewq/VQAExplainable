import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json

def load_glove_embeddings(file_path, embedding_dim=50):
    word_embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = torch.tensor(vector)
    return word_embeddings

def tokenize_question(question):
    return question.lower().split()

def tokens_to_embeddings(tokens, embedding_model, max_length=23):
    embeddings = []
    for token in tokens:
        embedding = embedding_model.get(token, torch.zeros(50))
        embeddings.append(embedding)

    if len(embeddings) < max_length:
        padding = [torch.zeros(50)] * (max_length - len(embeddings))
        embeddings.extend(padding)
    else:
        embeddings = embeddings[:max_length]

    return torch.stack(embeddings)

def create_answer_idx(train_annotations_file=None, val_annotations_file=None):
    all_answers = set()
    if train_annotations_file is not None:
        with open(train_annotations_file, 'r') as f:
            train_annotations = json.load(f)['annotations']
            for annotation in train_annotations:
                all_answers.add(annotation['multiple_choice_answer'])

    if val_annotations_file is not None:
        with open(val_annotations_file, 'r') as f:
            val_annotations = json.load(f)['annotations']
            filtered_annotations = [ann for ann in val_annotations if ann['image_id'] <= 14226]
            for annotation in filtered_annotations:
                all_answers.add(annotation['multiple_choice_answer'])

    answer_to_idx = {answer: idx for idx, answer in enumerate(sorted(all_answers))}
    idx_to_answer = {idx: answer for idx, answer in enumerate(sorted(all_answers))}
    return answer_to_idx, idx_to_answer


class VQADataset(Dataset):
    def __init__(self, img_feats_dir, questions_file, annotations_file, answer_to_idx, max_length=23):
        self.img_feats_dir = img_feats_dir
        self.max_length = max_length # question tokens max length
        with open(questions_file, 'r') as f:
            questions_data = json.load(f)['questions']
            self.questions_data = [q for q in questions_data if q['image_id'] <= 14226]
        with open(annotations_file, 'r') as f:
            annotations_data = json.load(f)['annotations']
            self.annotations_data = [ann for ann in annotations_data if ann['image_id'] <= 14226]
        self.answer_to_idx = answer_to_idx
        glove_file_path = "./data/glove/glove.6B.50d.txt"
        self.glove = load_glove_embeddings(glove_file_path)

    def __len__(self):
        return len(self.questions_data)

    def __getitem__(self, idx):
        assert idx <= 14226
        img_id = self.questions_data[idx]['image_id']
        if 'train' in self.img_feats_dir:
            img_feat_path = os.path.join(self.img_feats_dir, f"COCO_train2014_{img_id:012d}.npz")
        else:
            img_feat_path = os.path.join(self.img_feats_dir, f"COCO_val2014_{img_id:012d}.npz")
        img_feat = np.load(img_feat_path)['arr_0']

        question = self.questions_data[idx]['question']
        question_tokens = tokenize_question(question)
        question_embeddings = tokens_to_embeddings(question_tokens, self.glove, self.max_length)

        answer = self.annotations_data[idx]['multiple_choice_answer']
        answer_idx = self.answer_to_idx[answer]

        return torch.tensor(img_feat, dtype=torch.float32), question_embeddings, answer_idx
