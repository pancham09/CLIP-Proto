
Copy

# CLIP-Proto: Text-Guided Prototype Learning for Surgical Instrument Segmentation
 
> **CLIP-Proto** extends [SurgicalSAM](https://github.com/wenxi-yue/SurgicalSAM) with persistent CLIP text alignment and per-class frequency weighting, improving prototype discriminability for fine-grained surgical instrument segmentation.
 
---
 
## Overview
 
SurgicalSAM achieves state-of-the-art surgical instrument segmentation by learning class prototypes through contrastive learning. However, two limitations exist:
 
1. **Semantically empty prompts** — class prototypes are initialized from random noise (`N(0,1)`), providing no prior knowledge about instrument appearance or function.
2. **Class imbalance** — rare instrument classes (e.g., Grasping Retractor, Ultrasound Probe) receive fewer gradient signals during contrastive learning, leading to near-zero IoU on those classes.
CLIP-Proto addresses both with a single additional loss term — **zero architectural changes, zero additional trainable parameters in the core model**.
 
---
 
## Method
 
### Core Contribution
 
We add a persistent **CLIP Text Alignment Loss** to SurgicalSAM's training objective:
 
```
L = L_dice + L_contrastive + λ * L_clip_align
```
 
Where:
 
```
L_clip_align = (1/C) * Σ_k [ w_k * (1 - cosine_sim(prototype_k, anchor_k)) ]
```
 
- `prototype_k` — learnable prototype for class k (same as SurgicalSAM)
- `anchor_k` — **frozen** CLIP text embedding of class k descriptions
- `w_k` — inverse frequency weight (higher for rare classes)
- `λ` — alignment strength, cosine-annealed from 0.5 → 0.05 over training
### Why This Works
 
In SurgicalSAM, prototypes start from random noise. The contrastive loss must first separate 7 random vectors before encoding any surgical knowledge — wasted early training. CLIP text anchors provide a persistent pull toward semantically meaningful positions throughout all epochs, creating a tension between:
 
- **Contrastive loss** — pulls prototypes toward actual surgical image features
- **CLIP alignment loss** — pulls prototypes toward text-semantic positions
Prototypes learn a **joint image-text representation** rather than collapsing into pure image space, improving discriminability especially for visually similar instrument pairs.
 
### Per-Class Frequency Weighting
 
Rare classes get stronger text supervision via inverse-frequency weights computed from training data:
 
```python
w_k = (1 / count_k) / mean(1 / count_k)
```
 
This directly targets SurgicalSAM's documented failure on rare classes (SI, CA, GR near 0% IoU).
 
### Lambda Annealing
 
To prevent early collapse of prototypes into CLIP space, lambda follows a cosine schedule:
 
```python
λ(epoch) = λ_min + 0.5 * (λ_max - λ_min) * (1 + cos(π * epoch/total_epochs))
```
 
- Epoch 0: λ ≈ 0.50 (strong semantic grounding)
- Epoch 1000: λ ≈ 0.27 (balanced)
- Epoch 2000: λ ≈ 0.05 (surgical domain specialization)
---
 
## What Changes vs SurgicalSAM
 
| Component | SurgicalSAM | CLIP-Proto |
|---|---|---|
| Prototype initialization | `N(0,1)` random | `N(0,1)` random (unchanged) |
| Training loss | `L_dice + L_contrastive` | `L_dice + L_contrastive + L_clip_align` |
| CLIP text anchors | None | Frozen, built once at startup |
| Class weights | Uniform | Inverse frequency per fold |
| Lambda schedule | N/A | Cosine anneal 0.5 → 0.05 |
| Trainable params | 4,650,984 | 4,650,984 (identical) |
| SAM ViT-H backbone | Frozen | Frozen (identical) |
| Prototype_Prompt_Encoder | Unchanged | Unchanged |
| Mask decoder | Unchanged | Unchanged |
| Augmentation schedule | Unchanged | Unchanged |
 
---
 
## Repository Structure
 
```
SurgicalSAM/
├── surgicalSAM/
│   ├── train_clip_align.py          ← Main training script (this work)
│   ├── clip_text_align_loss.py      ← CLIPTextAlignLoss + class weights + anchors
│   ├── model.py                     ← Learnable_Prototypes, Prototype_Prompt_Encoder
│   ├── model_forward.py             ← Forward function (unchanged)
│   ├── dataset.py                   ← Endovis17/18 dataset classes
│   ├── loss.py                      ← DiceLoss
│   └── utils.py                     ← Evaluation utilities
├── segment_anything/                ← SAM source (unchanged)
├── data/
│   ├── endovis_2017/
│   │   ├── {0..40}/                 ← Augmentation versions
│   │   │   ├── images/
│   │   │   ├── sam_features_h/      ← Precomputed ViT-H embeddings (.npy)
│   │   │   ├── class_embeddings_h/  ← Per-class image features (.npy)
│   │   │   └── binary_annotations/
│   │   └── mappings.json
│   └── endovis_2018/
└── ckp/
    └── sam/
        └── sam_vit_h_4b8939.pth
```
 
---
 
## Installation
 
### 1. Follow SurgicalSAM setup
 
```bash
git clone https://github.com/wenxi-yue/SurgicalSAM
cd SurgicalSAM
pip install -r requirements.txt
```
 
### 2. Install CLIP
 
```bash
pip install git+https://github.com/openai/CLIP.git
```
 
### 3. Verify
 
```bash
python -c "import clip; print('CLIP OK')"
python -c "import torch; print(torch.cuda.is_available())"
```
 
### 4. Place files
 
Copy `train_clip_align.py` and `clip_text_align_loss.py` into `surgicalSAM/`.
 
---
 
## Data Preparation
 
Follow SurgicalSAM's original data preparation for EndoVis2017 and EndoVis2018. Precomputed SAM ViT-H embeddings must exist at:
 
```
data/endovis_2017/{version}/sam_features_h/
data/endovis_2017/{version}/class_embeddings_h/
```
 
CLIP-Proto uses the **same embeddings** as SurgicalSAM — no recomputation needed.
 
---
 
## Training
 
### Full Method (recommended)
 
```bash
# EndoVis2017 — run all 4 folds
python train_clip_align.py --dataset endovis_2017 --fold 0 --anneal y
python train_clip_align.py --dataset endovis_2017 --fold 1 --anneal y
python train_clip_align.py --dataset endovis_2017 --fold 2 --anneal y
python train_clip_align.py --dataset endovis_2017 --fold 3 --anneal y
 
