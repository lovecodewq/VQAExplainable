import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import random

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

def create_answer_idx(train_annotations_file=None, val_annotations_file=None, freq=8):
    from collections import Counter

    answer_counter = Counter()

    if train_annotations_file is not None:
        with open(train_annotations_file, 'r') as f:
            train_annotations = json.load(f)['annotations']
            for annotation in train_annotations:
                answer_counter[annotation['multiple_choice_answer']] += 1

    if val_annotations_file is not None:
        with open(val_annotations_file, 'r') as f:
            val_annotations = json.load(f)['annotations']
            for annotation in val_annotations:
                answer_counter[annotation['multiple_choice_answer']] += 1

    filtered_answers = [answer for answer, count in answer_counter.items() if count > freq]

    answer_to_idx = {answer: idx for idx, answer in enumerate(sorted(filtered_answers))}
    idx_to_answer = {idx: answer for answer, idx in answer_to_idx.items()}

    return answer_to_idx, idx_to_answer


class VQADataset(Dataset):
    def __init__(self, img_feats_dir, questions_file, annotations_file, answer_to_idx, top_k=30, max_length=23):
        self.img_feats_dir = img_feats_dir
        self.max_length = max_length
        self.answer_to_idx = answer_to_idx
        glove_file_path = "glove.6B.50d.txt"
        self.glove = load_glove_embeddings(glove_file_path)
        self.top_k = top_k

        with open(questions_file, 'r') as f:
            questions_data = json.load(f)['questions']
        with open(annotations_file, 'r') as f:
            annotations_data = json.load(f)['annotations']

        self.samples = []
        for q, ann in zip(questions_data, annotations_data):
            img_id = q['image_id']
            if 'train' in self.img_feats_dir:
                img_feat_path = os.path.join(self.img_feats_dir, f"COCO_train2014_{img_id:012d}.npz")
            else:
                img_feat_path = os.path.join(self.img_feats_dir, f"COCO_val2014_{img_id:012d}.npz")
            if os.path.exists(img_feat_path):
                self.samples.append((q, ann))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        q, ann = self.samples[idx]

        img_id = q['image_id']
        if 'train' in self.img_feats_dir:
            img_feat_path = os.path.join(self.img_feats_dir, f"COCO_train2014_{img_id:012d}.npz")
        else:
            img_feat_path = os.path.join(self.img_feats_dir, f"COCO_val2014_{img_id:012d}.npz")
        img_feat = np.load(img_feat_path)['arr_0']
        assert self.top_k >= img_feat.shape[0]
        padded_img_feat = np.zeros((self.top_k, img_feat.shape[1]), dtype=img_feat.dtype)
        padded_img_feat[:img_feat.shape[0], :] = img_feat

        question = q['question']
        question_tokens = tokenize_question(question)
        question_embeddings = tokens_to_embeddings(question_tokens, self.glove, self.max_length)

        answer = ann['multiple_choice_answer']
        if answer in self.answer_to_idx:
            answer_idx = self.answer_to_idx[answer]
        else:
            answer_idx = random.randint(0, len(self.answer_to_idx) - 1)

        return torch.tensor(padded_img_feat, dtype=torch.float32), question_embeddings, answer_idx, answer
