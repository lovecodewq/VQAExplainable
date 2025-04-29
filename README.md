# 1 Generating Visual Question Answers with Enhanced Reasoning (VisionQuest)

**Course:** CS 7643 Final Project  
**Team:** VisionQuest  
**Members:** Feiyi Jiang, Minyue Jin, Wenqiang Li, Bochuan Lyu

---

## 2 Project Overview

Visual Question Answering (VQA) combines computer vision and natural language processing to answer questions about images. In this project, we will:

- Implement a **baseline** VQA model (Bottom‑Up & Top‑Down Attention, Anderson et al. 2017) from scratch.  
- Explore **advanced** pre-trained multimodal models (e.g., BLIP, Visual Instruction Tuning) and fine‑tune them using LoRA.  
- Enhance **reasoning** by generating concise textual explanations and visualizing attention maps with Grad‑CAM and Finer‑CAM.  

Our goal is not only to maximize answer accuracy but also to expose the model's decision‑making process for better interpretability.

---

## 3 Repository Structure

    VQA-Explainable/
    ├── bottom_up_train/         # Training scripts for bottom-up attention
    ├── vqa_train/               # VQA model training
    ├── config/                  # Configuration files
    ├── data/                    # Dataset storage
    │   ├── train2014/           # MSCOCO train images
    │   ├── val2014/             # MSCOCO validation images
    │   └── cache/               # Cached processed data
    ├── datasets/                # Dataset loading and processing
    ├── docs/                    # Documentation
    ├── models/                  # Model implementations
    ├── scripts/                 # Utility scripts
    ├── thirdparty/              # Third-party dependencies
    ├── uitls/                   # Utility functions

---

## 4 Setup and Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/<your-org>/VQA-Explainable.git
   cd VQA-Explainable
   git checkout dev
   ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up Python Path**
   To properly import from the `thirdparty` directory, add it to your Python path:
   ```bash
   # Add this to your .bashrc or .zshrc
   export PYTHONPATH=$PYTHONPATH:/path/to/your/VQA-Explainable/thirdparty
   ```
   Or add this at the beginning of your main scripts:
   ```python
   import sys
   import os
   sys.path.append(os.path.join(os.path.dirname(__file__), 'thirdparty'))
   ```

4. **Download and preprocess Visual Genome dataset**
    ```bash
    bash scripts/setup_vg_data.sh
    ```

## 5 Data Setup

1. **Make sure VQA v2 datasets are downloaded, unzipped and stored in ./data/**
```
VQA-Explainable/
    ├── data/
    │   ├── train2014/
    │   ├── val2014/
    │   ├── v2_OpenEnded_mscoco_train2014_questions.json
    |   ├── v2_mscoco_train2014_annotations.json
    |   ├── v2_OpenEnded_mscoco_val2014_questions.json
    |   ├── v2_mscoco_val2014_annotations.json
    |   └── glove/
    |         └── glove.6B.50d.txt
```

## 6 Training

### Bottom-Up Attention
```bash
bash bottom_up_train/run_train_bottom_up.sh
```
**Note**: add -d to specify device, default device is cpu

### Generate Bottom-Up Features
```bash
bash bottom_up_train/run_generate_bottom_up_features.sh
```

### Set up Pretrained Bottom-Up Model
See [pretrained model](thirdparty/Faster-R-CNN-with-model-pretrained-on-Visual-Genome/README.md)

### VQA Model Training
1. **Update hyperparameters**
   Edit `./config/VQA_training_config.yaml` to set your desired parameters

2. **Start training**
   ```python
   python vqa_train/train_VQA.py
   ```
