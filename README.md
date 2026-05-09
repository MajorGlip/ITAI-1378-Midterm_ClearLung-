# ClearLung: Pneumonia Detection from Chest X-Rays

**Course:** ITAI 1378 – Computer Vision and AI  
**Student:** Jonathan Ball  
**Tier:** Tier 1 — Binary Image Classification  

---

## Problem Statement

Pneumonia kills over 2.5 million people annually, yet early detection is frequently delayed due to radiologist shortages — especially in under-resourced hospitals and rural clinics. Manual X-ray review is time-consuming, expensive, and error-prone under high workloads. An automated pre-screening tool could flag high-risk cases before a radiologist reviews them, reducing diagnostic bottlenecks at near-zero cost.

---

## Solution Overview

ClearLung is a binary image classification system trained on chest X-ray photographs to predict whether a patient's lungs show signs of pneumonia. The model takes a grayscale X-ray image as input and returns a label — **NORMAL** or **PNEUMONIA** — along with a confidence percentage.

```
[Chest X-Ray Image]
       ↓
[Preprocessing: Resize → Normalize → Augment]
       ↓
[ResNet-18 (Fine-tuned via Transfer Learning)]
       ↓
[Softmax Output]
       ↓
[Prediction: NORMAL or PNEUMONIA + Confidence %]
```

---

## Tier Justification

**Tier 1** — This project uses a well-established public dataset, a pre-trained backbone (ResNet-18), and a straightforward binary classification task. Transfer learning is applied to keep training feasible within the 6-week timeline on free Colab resources.

---

## Technical Approach

| Component | Choice | Reason |
|---|---|---|
| CV Technique | Image Classification | Binary output: Normal vs. Pneumonia |
| Model | ResNet-18 | Lightweight, beginner-friendly, strong published benchmarks |
| Training Strategy | Transfer Learning | Freeze early layers; retrain final FC layer |
| Framework | PyTorch + torchvision | Industry standard, great documentation |
| Demo UI | Gradio | Easy to build, shareable via URL |

---

## Dataset Plan

**Source:** [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Attribute | Detail |
|---|---|
| Total Images | 5,863 |
| Normal | 1,583 |
| Pneumonia | 4,273 (Bacterial + Viral) |
| Format | JPEG grayscale |
| Pre-split | Yes — train / val / test folders included |
| Manual labeling needed? | No |

**Preparation steps:**
- Resize all images to 224×224 px (ResNet-18 input size)
- Normalize pixel values using ImageNet mean and std
- Apply augmentation: random horizontal flips, ±10° rotation
- Handle class imbalance with a weighted CrossEntropyLoss

---

## Success Metrics

| Metric | Type | Target |
|---|---|---|
| Accuracy | Primary | ≥ 88% on test set |
| Recall (Sensitivity) | Primary | ≥ 90% — minimize missed pneumonia |
| Cross-Entropy Loss | Secondary | < 0.25 |
| Inference Speed | Secondary | < 2 seconds per image (Colab GPU) |

> **Note:** Recall is prioritized over raw accuracy. A missed pneumonia case (false negative) is clinically more dangerous than a false alarm.

---

## Week-by-Week Plan

| Week | Dates | Task | Milestone |
|---|---|---|---|
| 10 | Oct 30 | Download Kaggle dataset, set up GitHub repo, explore data | Dataset ready |
| 11 | Nov 6  | Load pre-trained ResNet-18, write training loop, first run | Model trains without errors |
| 12 | Nov 13 | Evaluate accuracy/recall, tune hyperparameters + augmentation | ≥ 85% accuracy |
| 13 | Nov 20 | Build Gradio demo UI, run inference on new X-rays, record demo | Demo working |
| 14 | Nov 27 | Final testing, write README docs, clean codebase | Submission-ready |
| 15 | Dec 4  | Prepare and rehearse 5-minute presentation | 🎉 Presentation Day |

---

## Risks & Mitigation

| Risk | Probability | Mitigation |
|---|---|---|
| Class imbalance (4× more pneumonia) | High | Weighted loss + oversample normal class via augmentation |
| Low accuracy (< 80%) | Medium | Switch to EfficientNet-B0; increase epochs; add dropout |
| Kaggle dataset access issues | Low | Fallback to NIH Chest X-ray Dataset (also free on Kaggle) |
| Colab GPU session timeouts | Medium | Save checkpoints every epoch; use Kaggle Notebooks (30 GPU hrs/wk) |

---

## Resources Needed

| Resource | Plan |
|---|---|
| Compute | Google Colab (free T4 GPU) / Kaggle Notebooks |
| Framework | PyTorch, torchvision, Gradio, Matplotlib |
| Dataset | Kaggle (~1.2 GB download) |
| Estimated Cost | **$0** |

---

## Repository Structure

```
ClearLung/
├── README.md                  ← You are here
├── requirements.txt           ← Python packages
├── notebooks/
│   └── 01_exploration.ipynb   ← EDA + sample visualizations
├── src/
│   ├── train.py               ← Training loop
│   ├── evaluate.py            ← Metrics + confusion matrix
│   └── demo.py                ← Gradio inference app
├── data/
│   └── README.md              ← Dataset download instructions
└── docs/
    └── proposal.pdf           ← Presentation slides
```

---

## AI Usage Log

| Tool | How Used |
|---|---|
| Claude (Anthropic) | Project ideation, slide content, README draft |

*This log will be updated throughout the project as AI tools are used.*

---

## Setup Instructions

```bash
# Clone the repo
git clone https://github.com/MajorGlip/ClearLung.git
cd ClearLung

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# (requires Kaggle API key — see data/README.md)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Run training
python src/train.py

# Launch demo
python src/demo.py
```

---

## requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
gradio>=4.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
kaggle>=1.5.0
Pillow>=9.5.0
numpy>=1.24.0
```
