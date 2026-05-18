# CLIP-Proto: Semantically Grounded Prototype Learning for Surgical Instrument Segmentation

<div align="center">

[![IEEE Access](https://img.shields.io/badge/IEEE%20Access-2025-blue)](https://ieeeaccess.ieee.org)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
 
</div>

---

## Overview

CLIP-Proto extends [SurgicalSAM](https://github.com/wenxi-yue/SurgicalSAM) with a persistent **CLIP text alignment loss** that anchors learnable class prototypes to frozen semantic embeddings throughout training. This directly addresses two root causes of failure in SAM-based surgical instrument segmentation:

1. **Random prototype initialisation** — prototypes start from `N(0,1)` with no surgical prior, requiring hundreds of epochs just to establish basic class separation before learning surgical knowledge.
2. **Gradient starvation for rare classes** — in EndoVis2017, the rarest class (Monopolar Curved Scissors, 228 frames) receives less than one-third the contrastive gradient of the most common class (Prograsp Forceps, 745 frames), causing prototype collapse for underrepresented instruments.

The fix is a single additional loss term — **zero architectural changes, zero inference overhead**.

```
L = L_dice + L_contrastive + λ(e) · L_clip
```

<div align="center">

| Method | Ch-IoU | mcIoU | GR IoU | PF IoU | MCS IoU |
|---|---|---|---|---|---|
| SurgicalSAM | 64.63 | 57.70 | 50.36 | 34.09 | 73.14 |
| **CLIP-Proto (ours)** | **66.36** | **60.54** | **58.71** | **40.59** | **75.54** |
| Δ | **+1.73** | **+2.84** | **+8.35** | **+6.50** | **+2.40** |

*EndoVis2017 Fold 2. Inference cost: 0%. Training overhead: <2%.*

</div>

---

## Key Features

- **Persistent text supervision** — CLIP alignment runs all 2000 epochs, not just as a warm-start. Prototypes balance between image-level discrimination and semantic coherence throughout training.
- **Cosine-annealed λ schedule** — strong semantic grounding early (λ=0.5) when prototypes are random, decaying to a light regulariser late (λ=0.05) to allow surgical domain specialisation.
- **Inverse-frequency class weighting** — rare classes receive proportionally stronger text supervision, directly countering their sparse contrastive gradient.
- **Zero inference overhead** — CLIP encoder, text anchors, and projection head are all discarded after training. The deployed model is identical to SurgicalSAM.
- **Drop-in enhancement** — modifies only the training loss. Compatible with any prototype-based SAM adaptation without architectural changes.
- **Same parameter count** — 4.65M trainable parameters, identical to SurgicalSAM.

---

## Dataset

### EndoVis2017

The **MICCAI 2017 Robotic Instrument Segmentation Challenge** dataset consists of 8 robotic surgery sequences recorded with a da Vinci Xi system at 1280×1024 resolution, with pixel-level annotations for 7 instrument classes.

| Class | ID | Fold 2 Train Frames | Freq. Weight |
|---|---|---|---|
| Bipolar Forceps | 1 | 651 | 0.65 |
| Prograsp Forceps | 2 | 745 | 0.57 |
| Large Needle Driver | 3 | 672 | 0.63 |
| Vessel Sealer | 4 | 386 | 1.10 |
| Grasping Retractor | 5 | 228 | 1.87 |
| Monopolar Curved Scissors | 6 | 351 | 1.21 |
| Ultrasound Probe | 7 | 449 | 0.95 |

**Download:** [EndoVis2017 Challenge Page](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/)  
*Request access and download the instrument segmentation subset.*

### EndoVis2018

The **MICCAI 2018 Robotic Scene Segmentation Challenge** dataset is also supported.

**Download:** [EndoVis2018 Challenge Page](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/)

---

## Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.8+ |
| PyTorch | 1.12+ |
| CUDA | 11.3+ (12.4+ for Blackwell GPUs) |
| GPU VRAM | 16GB minimum, 24GB recommended |

### Step 1 — Clone repository

```bash
git clone https://github.com/pancham09/CLIP-Proto.git
cd CLIP-Proto
```

### Step 2 — Create environment

```bash
conda create -n clipproto python=3.8
conda activate clipproto
```

### Step 3 — Install PyTorch

```bash
# CUDA 11.3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# CUDA 12.4+ (RTX 40xx / Blackwell)
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124
```

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### Step 5 — Install Segment Anything

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Step 6 — Download SAM ViT-H checkpoint

```bash
mkdir -p ckp/sam
wget -P ckp/sam \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### Verify installation

```bash
python -c "
import torch, clip, segment_anything
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
print(f'CLIP:     OK')
print(f'SAM:      OK')
"
```

---

## Dataset Setup

### Directory structure

```
CLIP-Proto/
├── ckp/
│   └── sam/
│       └── sam_vit_h_4b8939.pth
├── data/
│   ├── endovis_2017/
│   │   ├── mappings.json
│   │   └── {0..40}/                  ← augmentation versions
│   │       ├── images/
│   │       ├── binary_annotations/
│   │       ├── sam_features_h/       ← precomputed SAM embeddings
│   │       └── class_embeddings_h/   ← per-class masked features
│   └── endovis_2018/
│       └── {0..N}/
│           └── ...
└── surgicalSAM/
    ├── train_clip_align.py
    ├── clip_text_align_loss.py
    └── ...
```

### Preprocessing

Follow [SurgicalSAM's preprocessing pipeline](https://github.com/wenxi-yue/SurgicalSAM#data-preparation) to generate:
- `binary_annotations/` — per-class binary PNG masks
- `sam_features_h/` — precomputed SAM ViT-H embeddings (`.npy`, shape `64×64×256`)
- `class_embeddings_h/` — masked average foreground features per annotation (`.npy`, shape `256`)

CLIP-Proto uses **the same precomputed embeddings as SurgicalSAM** — no additional preprocessing required.

---

## Quick Start

```bash
cd surgicalSAM

# Train on EndoVis2017 Fold 2 with full CLIP-Proto method
python train_clip_align.py \
    --dataset endovis_2017 \
    --fold 2 \
    --anneal y \
    --lambda-clip 0.5
```

Training logs and checkpoints are saved to:
```
work_dirs_clip_align/endovis_2017/2/anneal/lambda_0.5/
├── log.txt
├── model_ckp.pth              ← best model
├── checkpoint_epoch_1950.pth  ← periodic checkpoint
└── checkpoint_epoch_1900.pth
```

---

## Training

### Full method — all folds in parallel

With sufficient GPU memory (96GB+), run all four folds simultaneously:

```bash
for fold in 0 1 2 3; do
    python train_clip_align.py \
        --dataset endovis_2017 \
        --fold $fold \
        --anneal y \
        --lambda-clip 0.5 &
done
wait
echo "All folds complete"
```

Each fold uses approximately 10GB VRAM.

### EndoVis2018

```bash
python train_clip_align.py \
    --dataset endovis_2018 \
    --anneal y \
    --lambda-clip 0.5
```

### Configuration

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `endovis_2018` | Dataset: `endovis_2017` or `endovis_2018` |
| `--fold` | `0` | Cross-validation fold 0–3 (EndoVis2017 only) |
| `--lambda-clip` | `0.5` | CLIP alignment loss weight. `0.0` disables CLIP |
| `--clip-model` | `ViT-B/32` | CLIP backbone: `ViT-B/32`, `ViT-B/16`, `ViT-L/14` |
| `--anneal` | `None` | Pass any value to enable cosine lambda annealing |
| `--no-class-weights` | `False` | Use uniform class weights (ablation) |
| `--resume` | `None` | Checkpoint path. Auto-detected from save directory |

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1×10⁻⁴ (EndoVis2017), 1×10⁻³ (EndoVis2018) |
| Weight decay | 1×10⁻⁴ |
| Batch size | 32 |
| Epochs | 2000 (2017), 500 (2018) |
| λ_max / λ_min | 0.50 / 0.05 |
| Projection head LR | 1×10⁻⁵ |
| CLIP backbone | ViT-B/32 |
| Random seed | 666 |

### Monitoring training

The log prints three losses per epoch:

```
Epoch 300/1999 | seg=0.0856  cont=2.0941  clip=0.4062
```

| Loss | Epoch 0 | Epoch 500 | If abnormal |
|---|---|---|---|
| `seg` | ~0.9 | ~0.1–0.3 | High → segmentation not converging |
| `cont` | ~3.3 | ~2.0–2.5 | High → prototypes not separating |
| `clip` | ~0.5 | ~0.2–0.3 | `0.0` → annealing not enabled |

Monitor all folds simultaneously:

```bash
watch -n 30 'for f in work_dirs_clip_align/endovis_2017/*/anneal/lambda_0.5/log.txt; do
    echo "=== $f ===";
    grep "Best Challenge" $f | tail -1;
    grep "^Epoch" $f | tail -1;
done'
```

---

## Resuming Training

Training auto-detects the latest checkpoint from the save directory:

```bash
# Resume automatically
python train_clip_align.py \
    --dataset endovis_2017 \
    --fold 2 \
    --anneal y \
    --lambda-clip 0.5
# Prints: "Auto-detected checkpoint: .../checkpoint_epoch_1500.pth"

# Resume from specific checkpoint
python train_clip_align.py \
    --dataset endovis_2017 \
    --fold 2 \
    --anneal y \
    --lambda-clip 0.5 \
    --resume work_dirs_clip_align/endovis_2017/2/anneal/lambda_0.5/checkpoint_epoch_1000.pth
```

The lambda schedule correctly resumes mid-anneal — it is recomputed from `epoch / total_epochs` each run, not saved in the checkpoint.

Checkpoints are saved every 50 epochs. The last 3 periodic checkpoints are kept plus the best model.

---

## Ablation Study

Run all variants for the paper ablation table:

```bash
# A: Pure SurgicalSAM baseline (λ=0)
python train_clip_align.py --dataset endovis_2017 --fold 2 --lambda-clip 0.0

# B: CLIP align, fixed λ, uniform weights
python train_clip_align.py --dataset endovis_2017 --fold 2 \
    --lambda-clip 0.5 --no-class-weights

# C: CLIP align, fixed λ, frequency weights
python train_clip_align.py --dataset endovis_2017 --fold 2 \
    --lambda-clip 0.5

# D: CLIP align, annealed λ, uniform weights
python train_clip_align.py --dataset endovis_2017 --fold 2 \
    --anneal y --lambda-clip 0.5 --no-class-weights

# E: CLIP-Proto full (annealed λ + frequency weights)
python train_clip_align.py --dataset endovis_2017 --fold 2 \
    --anneal y --lambda-clip 0.5
```

---

## Inference & Evaluation

### Evaluate best model

```bash
python evaluate.py \
    --checkpoint work_dirs_clip_align/endovis_2017/2/anneal/lambda_0.5/model_ckp.pth \
    --dataset endovis_2017 \
    --fold 2
```

### Generate qualitative comparison figures

```bash
python generate_qualitative_figures.py \
    --surgicalsam-ckpt ../surgicalSAM/work_dirs/endovis_2017/2/model_ckp.pth \
    --clipproto-ckpt   work_dirs_clip_align/endovis_2017/2/anneal/lambda_0.5/model_ckp.pth \
    --dataset endovis_2017 \
    --fold 2 \
    --auto-select \
    --output-dir figures/
```

Output:
```
figures/
├── row1_prograsp_forceps_gt.png
├── row1_prograsp_forceps_surgicalsam.png
├── row1_prograsp_forceps_clipproto.png
├── ...
└── figure1_combined_hires.png     ← 300 DPI for paper
```

---

## Results

### EndoVis2017 Fold 2

| Method | Ch-IoU | mcIoU | BF | PF | LND | GR | MCS |
|---|---|---|---|---|---|---|---|
| TernausNet | 35.27 | 10.17 | 13.45 | 12.39 | 20.51 | 1.08 | 1.00 |
| ISINet | 55.62 | 28.96 | 38.70 | 38.50 | 50.09 | 2.10 | 28.72 |
| S3Net | 72.54 | 46.55 | 75.08 | 54.32 | 61.84 | 27.47 | 43.23 |
| SurgicalSAM | 64.63 | 57.70 | 60.43 | 34.09 | 70.50 | 50.36 | 73.14 |
| **CLIP-Proto** | **66.36** | **60.54** | **60.68** | **40.59** | 67.19 | **58.71** | **75.54** |
| Δ vs SurgicalSAM | +1.73 | +2.84 | +0.25 | +6.50 | −3.31 | +8.35 | +2.40 |

The mcIoU gain (+2.84%) exceeds the Ch-IoU gain (+1.73%) because mcIoU weights all classes equally — rare-class improvements (GR +8.35%, PF +6.50%) are fully captured. The LND regression (−3.31%) is expected: frequency weighting deliberately redirects alignment budget to rare classes.

---

## Repository Structure

```
CLIP-Proto/
├── surgicalSAM/
│   ├── train_clip_align.py              ← main training script
│   ├── clip_text_align_loss.py          ← CLIPTextAlignLoss + anchors + weights
│   ├── generate_qualitative_figures.py  ← paper figure generation
│   ├── model.py                         ← Learnable_Prototypes, PPE
│   ├── model_forward.py                 ← forward pass
│   ├── dataset.py                       ← Endovis17/18 dataset classes
│   ├── loss.py                          ← DiceLoss
│   └── utils.py                         ← evaluation utilities
├── segment_anything/                    ← SAM source (unchanged)
├── data/                                ← dataset (not tracked)
├── ckp/                                 ← checkpoints (not tracked)
├── requirements.txt
└── README.md
```


---

## Acknowledgements

Built on [SurgicalSAM](https://github.com/wenxi-yue/SurgicalSAM) and
[Segment Anything](https://github.com/facebookresearch/segment-anything).
CLIP text encodings from [OpenAI CLIP](https://github.com/openai/CLIP).
EndoVis datasets from the
[MICCAI Endoscopic Vision Challenge](https://endovis.grand-challenge.org/).

---
