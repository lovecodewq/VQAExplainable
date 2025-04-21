# 1 Generating Visual Question Answers with Enhanced Reasoning (VisionQuest)

**Course:** CS 7643 Final Project  
**Team:** VisionQuest  
**Members:** Feiyi Jiang, Minyue Jin, Wenqiang Li, Bochuan Lyu

---

## 2 Project Overview

Visual Question Answering (VQA) combines computer vision and natural language processing to answer questions about images. In this project, we will:

- Implement a **baseline** VQA model (Bottom‑Up & Top‑Down Attention, Anderson et al. 2017) from scratch.  
- Explore **advanced** pre-trained multimodal models (e.g., BLIP, Visual Instruction Tuning) and fine‑tune them using LoRA.  
- Enhance **reasoning** by generating concise textual explanations and visualizing attention maps with Grad‑CAM and Finer‑CAM.  

Our goal is not only to maximize answer accuracy but also to expose the model’s decision‑making process for better interpretability.

---

## 3 Repository Structure

    VQA-Explainable/
    ├── data/                    # Data download scripts or small sample files
    ├── notebooks/               # Jupyter notebooks for experiments
    ├── src/                     # Source code
    │   ├── dataset/             # Data loading and preprocessing
    │   ├── models/              # Model definitions
    │   └── utils/               # Helper functions
    ├── tests/                   # Unit tests
    ├── scripts/                 # CLI scripts (train.py, eval.py, infer.py)
    ├── configs/                 # YAML/JSON config files
    ├── docs/                    # Documentation and reports
    ├── Dockerfile               # Container setup
    ├── requirements.txt         # Python dependencies
    ├── LICENSE                  # Project license
    └── README.md                # Project overview and setup instructions

---

## 🚀 Quick Start

1. **Clone the repository**  
   ```bash
   git clone https://github.com/<your-org>/VQA-Explainable.git
   cd VQA-Explainable
   git checkout dev

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3. **Download and preprocess dataset of visual genome**
    ```bash
    bash scripts/setup_vg_data.sh
    ```

## Testing

1. **Test vg dataset by visualization**
    ```python
    python tests/test_vg_dataset_single.py
    ```

