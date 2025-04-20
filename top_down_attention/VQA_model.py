import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import yaml

def vqa_loss_fn(logits, targets):
    """
    logits: (batch_size, answer_vocab_size), raw model outputs
    targets: (batch_size, answer_vocab_size), soft targets between 0 and 1
             where targets[i][j] is the score for answer j in sample i
    """
    return F.binary_cross_entropy_with_logits(logits, targets)


class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        
    def forward(self, question_tokens):
        # (batch_size, 14, 300), 14: num_tokens
        x = self.embedding(question_tokens)
        _, h_n = self.gru(x)  # h_n: (1, batch_size, 512)
        return h_n.squeeze(0)  # (batch_size, 512)


class TopDownAttention(nn.Module):
    def __init__(self, image_feature_dim=2048, question_dim=512, hidden_dim=512):
        super().__init__()
        self.image_linear = nn.Linear(image_feature_dim, hidden_dim)
        self.question_linear = nn.Linear(question_dim, hidden_dim)
        self.attention_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, image_feats, q_features):
        # image_feats: (batch, k, 2048)
        # q_features: (batch, 512)
        img_proj = self.image_linear(image_feats)  # (batch, k, 512)
        q_proj = self.question_linear(q_features).unsqueeze(1)  # (batch, 1, 512)
        
        joint_repr = torch.tanh(img_proj + q_proj)  # (batch, k, 512)
        attn_weights = self.attention_layer(joint_repr).squeeze(-1)  # (batch, k)
        attn_weights = torch.softmax(attn_weights, dim=1)  # (batch, k)

        # Weighted sum of image features
        weighted_img_feats = (attn_weights.unsqueeze(-1) * image_feats).sum(dim=1)  # (batch, 2048)
        return weighted_img_feats


class VQAModel(nn.Module):
    def __init__(self, vocab_size, answer_vocab_size, image_feature_dim=2048, hidden_dim=512):
        super().__init__()
        self.q_encoder = QuestionEncoder(vocab_size)
        self.attention = TopDownAttention(image_feature_dim, hidden_dim)
        # image embedding
        self.img_linear = nn.Linear(image_feature_dim, hidden_dim)
        # question embedding
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, answer_vocab_size)
        )
        
    def forward(self, question_tokens, image_feats):
        # (batch, num_tokens)
        q_features = self.q_encoder(question_tokens)
        # (batch, num_features), 2048 by default
        v_attention = self.attention(image_feats, q_features)
        
        v_embedding = self.img_linear(v_attention)
        q_embedding = self.q_linear(q_features)

        # Element-wise product (batch, 512)
        joint_features = v_embedding * q_embedding
        # (batch, N)
        logits = self.classifier(joint_features)
        return logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_config_file = os.path.join(os.path.abspath(os.getcwd()), "./top_down_attention/VQA_model_config.yaml")
    parser.add_argument("--config", type=str, default=default_config_file, help="Path to config file")

    args = parser.parse_args()
    config_file = args.config
    with open(config_file, "rb") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

    # for training test
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    lr = config["learning_rate"]

    input_vocab_size = config["input_vocab_size"]
    answer_vocab_size = config["answer_vocab_size"]
    max_token_length = config["max_token_length"]
    image_feature_dim = config["image_feature_dim"]
    k = config["k"] # number of feature regions

    # Fake data
    question_tokens = torch.randint(0, max_token_length, (batch_size * 100, max_token_length))
    image_features = torch.randn(batch_size * 100, k, image_feature_dim)
    targets = torch.randn(batch_size * 100, answer_vocab_size)

    model = VQAModel(input_vocab_size, answer_vocab_size)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    for epoch in range(num_epochs):
      total_loss = 0.0
      for i in range(100):
        logits = model(question_tokens[batch_size*i : batch_size*(i+1)], \
          image_features[batch_size*i : batch_size*(i+1)])
        loss = vqa_loss_fn(logits, targets[batch_size*i : batch_size*(i+1)])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
      print(f"average loss at epoch {epoch} is {total_loss/100}")
