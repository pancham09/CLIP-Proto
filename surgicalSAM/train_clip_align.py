import sys
sys.path.append("..")
import os
import os.path as osp
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import Endovis18Dataset, Endovis17Dataset
from segment_anything import sam_model_registry
from model import Learnable_Prototypes, Prototype_Prompt_Encoder
from utils import (print_log, create_binary_masks, create_endovis_masks,
                   eval_endovis, read_gt_endovis_masks)
from model_forward import model_forward_function
from loss import DiceLoss
from pytorch_metric_learning import losses
from clip_text_align_loss import (CLIPTextAlignLoss,
                                   build_clip_anchors,
                                   compute_class_frequency_weights,
                                   compute_class_frequency_weights_2)


# =============================================================================
# Arguments
# =============================================================================

print("======> Process Arguments")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="endovis_2018",
                    choices=["endovis_2018", "endovis_2017"])
parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--compute', type=str, default=None)
parser.add_argument('--anneal', type=str, default=None)
parser.add_argument('--lambda-clip', type=float, default=0.5,
                    help='Weight for CLIP alignment loss. '
                         '0.0 = no CLIP (ablation). 0.5 = default. '
                         'Range: 0.1–1.0')
parser.add_argument('--clip-model', type=str, default="ViT-B/32",
                    choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
                    help='CLIP variant. ViT-L/14 gives richer 768-dim embeddings.')
parser.add_argument('--no-class-weights', action='store_true',
                    help='Use uniform class weights instead of inverse frequency. '
                         'Ablation for the frequency weighting component.')
args = parser.parse_args()

USE_CLIP       = args.lambda_clip > 0.0
USE_CLS_WEIGHT = not args.no_class_weights

print(f"  CLIP alignment loss : {'ENABLED (λ={})'.format(args.lambda_clip) if USE_CLIP else 'DISABLED (λ=0, ablation)'}")
print(f"  Class freq weights  : {'ENABLED' if USE_CLS_WEIGHT else 'DISABLED (uniform, ablation)'}")


# =============================================================================
# Parameters — identical to original
# =============================================================================

print("======> Set Parameters for Training")
dataset_name  = args.dataset
fold          = args.fold
thr           = 0
seed          = 666
data_root_dir = f"../data/{dataset_name}"
batch_size    = 32
vit_mode      = "h"

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
np.random.seed(seed)

if args.anneal is None:
    anneal = False
else:
    anneal = True
    print("Annealing")


print("======> Load Dataset-Specific Parameters")
if "18" in dataset_name:
    num_tokens = 2
    val_dataset      = Endovis18Dataset(data_root_dir=data_root_dir,
                                        mode="val", vit_mode="h")
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir=data_root_dir,
                                             mode="val")
    num_epochs = 500
    lr         = 0.001
    save_dir   = f"./work_dirs_clip_align/endovis_2018/lambda_{args.lambda_clip}"

elif "17" in dataset_name:
    num_tokens = 4
    val_dataset      = Endovis17Dataset(data_root_dir=data_root_dir,
                                        mode="val", fold=fold,
                                        vit_mode="h", version=0)
    gt_endovis_masks = read_gt_endovis_masks(data_root_dir=data_root_dir,
                                             mode="val", fold=fold)
    num_epochs = 2000
    lr         = 0.0001
    if anneal:
        addit = "anneal_savecp/"
    else:
        addit = ""
    save_dir   = (f"./work_dirs_clip_align/endovis_2017/{fold}/{addit}"
                  f"lambda_{args.lambda_clip}")

val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)


# =============================================================================
# Load SAM — identical to original
# =============================================================================

print("======> Load SAM")
sam_checkpoint = "../ckp/sam/sam_vit_h_4b8939.pth"
model_type     = "vit_h_no_image_encoder"
sam_prompt_encoder, sam_decoder = sam_model_registry[model_type](
    checkpoint=sam_checkpoint)
sam_prompt_encoder.cuda()
sam_decoder.cuda()

for param in sam_prompt_encoder.parameters():
    param.requires_grad = False
for param in sam_decoder.named_parameters():
    param[1].requires_grad = True


# =============================================================================
# Load Prototypes and Prompt Encoder — identical to original
# =============================================================================

print("======> Load Prototypes and Prototype-based Prompt Encoder")
learnable_prototypes_model = Learnable_Prototypes(num_classes=7,
                                                   feat_dim=256).cuda()
protoype_prompt_encoder    = Prototype_Prompt_Encoder(
    feat_dim=256, hidden_dim_dense=128, hidden_dim_sparse=128,
    size=64, num_tokens=num_tokens).cuda()

with open(sam_checkpoint, "rb") as f:
    state_dict = torch.load(f)
    sam_pn_embeddings_weight = {
        k.split("prompt_encoder.point_embeddings.")[-1]: v
        for k, v in state_dict.items()
        if k.startswith("prompt_encoder.point_embeddings")
        and ("0" in k or "1" in k)
    }
    sam_pn_embeddings_weight_ckp = {
        "0.weight": torch.concat(
            [sam_pn_embeddings_weight['0.weight']] * num_tokens, dim=0),
        "1.weight": torch.concat(
            [sam_pn_embeddings_weight['1.weight']] * num_tokens, dim=0),
    }
    protoype_prompt_encoder.pn_cls_embeddings.load_state_dict(
        sam_pn_embeddings_weight_ckp)

for param in learnable_prototypes_model.parameters():
    param.requires_grad = True

for name, param in protoype_prompt_encoder.named_parameters():
    param.requires_grad = (False if "pn_cls_embeddings" in name else True)


# =============================================================================
# CLIP Text Alignment Loss  ← THE NEW COMPONENT
# =============================================================================

if USE_CLIP:
    print("\n======> Build CLIP Text Alignment Loss")

    # Step 1: compute per-class frequency weights from training data
    if USE_CLS_WEIGHT:
        if args.compute == "cw":
            print("\n=====> Using CW")
            class_weights = compute_class_frequency_weights_2(
                data_root_dir=data_root_dir,
                dataset_name=dataset_name,
                fold=fold,
                num_classes=7,
            ).cuda()
        else:
            class_weights = compute_class_frequency_weights(
                data_root_dir=data_root_dir,
                dataset_name=dataset_name,
                fold=fold,
                num_classes=7,
            ).cuda()
    else:
        class_weights = torch.ones(7).cuda()
        print("  Using uniform class weights (ablation)")

    # Step 2: build frozen CLIP text anchors
    clip_anchors = build_clip_anchors(
        dataset_name=dataset_name,
        feat_dim=256,
        clip_model_name=args.clip_model,
        device="cuda",
    ).cuda()
    print(f"  clip_anchors shape: {clip_anchors.shape}")

    # Step 3: instantiate loss — clip_anchors and class_weights are buffers,
    # they move with the module but are NOT trainable parameters
    clip_align_loss_model = CLIPTextAlignLoss(
        clip_anchors=clip_anchors,
        class_weights=class_weights,
        lambda_clip=args.lambda_clip,
    ).cuda()

    print(f"\n  {clip_align_loss_model}")
    print(f"  CLIP anchors shape: {clip_anchors.shape}")
    print(f"  Class weights     : {[f'{w:.2f}' for w in class_weights.cpu().tolist()]}")

else:
    clip_align_loss_model = None
    print("\n======> CLIP alignment DISABLED (lambda=0.0, running pure SurgicalSAM)")


# =============================================================================
# Optimizer and Loss — identical to original
# =============================================================================

print("\n======> Define Optimizer and Loss")
seg_loss_model         = DiceLoss().cuda()
contrastive_loss_model = losses.NTXentLoss(temperature=0.07).cuda()

# Note: clip_align_loss_model has no trainable parameters (only buffers)
# so it does NOT go into the optimizer. Only the three original param groups.
optimiser_params = [
    {'params': learnable_prototypes_model.parameters()},
    {'params': protoype_prompt_encoder.parameters()},
    {'params': sam_decoder.parameters()},
]
if clip_align_loss_model is not None:
    optimiser_params.append(
        {'params': clip_align_loss_model.proto_proj.parameters(), 'lr': lr * 0.1}
)	
optimiser = torch.optim.Adam(optimiser_params, lr=lr, weight_decay=0.0001)

total_trainable = (
    sum(p.numel() for p in learnable_prototypes_model.parameters()
        if p.requires_grad) +
    sum(p.numel() for p in protoype_prompt_encoder.parameters()
        if p.requires_grad) +
    sum(p.numel() for p in sam_decoder.parameters() if p.requires_grad)
)
print(f"  Trainable parameters: {total_trainable:,}")
print(f"  (Same as original SurgicalSAM — CLIP adds zero trainable params)")


# =============================================================================
# Saving, Logging, Resume
# =============================================================================

os.makedirs(save_dir, exist_ok=True)
log_file = osp.join(save_dir, "log.txt")
print_log(str(args), log_file)
print_log(
    f"Method: SurgicalSAM + CLIP text align | "
    f"lambda_clip={args.lambda_clip} | "
    f"clip_model={args.clip_model} | "
    f"class_weights={'freq' if USE_CLS_WEIGHT else 'uniform'} | "
    f"trainable_params={total_trainable:,}",
    log_file
)

start_epoch            = 0
best_challenge_iou_val = -100.0

# Auto-detect checkpoint
if args.resume is None:
    existing_ckps = [f for f in os.listdir(save_dir)
                     if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    if existing_ckps:
        latest      = max(existing_ckps,
                          key=lambda x: int(x.replace("checkpoint_epoch_", "")
                                            .replace(".pth", "")))
        args.resume = osp.join(save_dir, latest)
        print(f"======> Auto-detected checkpoint: {args.resume}")

if args.resume is not None and osp.isfile(args.resume):
    checkpoint = torch.load(args.resume)
    protoype_prompt_encoder.load_state_dict(
        checkpoint['prototype_prompt_encoder_state_dict'])
    sam_decoder.load_state_dict(checkpoint['sam_decoder_state_dict'])
    learnable_prototypes_model.load_state_dict(checkpoint['prototypes_state_dict'])
    ckp_n_groups = len(checkpoint['optimizer_state_dict']['param_groups'])
    cur_n_groups = len(optimiser.param_groups)
    if ckp_n_groups == cur_n_groups:
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  Optimizer state restored ({cur_n_groups} param groups)")
    else:
        print(f"  Optimizer param groups mismatch "
              f"(checkpoint={ckp_n_groups}, current={cur_n_groups}) "
              f"- starting with fresh optimizer (model weights preserved)")
    
    # ADD THIS:
    if (clip_align_loss_model is not None and
            'clip_align_loss_state_dict' in checkpoint and
            checkpoint['clip_align_loss_state_dict'] is not None):
        clip_align_loss_model.load_state_dict(
            checkpoint['clip_align_loss_state_dict'],
            strict=False)   # strict=False because buffers may differ in size
        print("  Restored clip_align_loss proto_proj weights")
    
    start_epoch            = checkpoint['epoch'] + 1
    best_challenge_iou_val = checkpoint['best_challenge_iou_val']

def save_checkpoint(epoch: int, is_best: bool = False):
    ckp = {
        'epoch':                               epoch,
        'best_challenge_iou_val':              best_challenge_iou_val,
        'prototype_prompt_encoder_state_dict': protoype_prompt_encoder.state_dict(),
        'sam_decoder_state_dict':              sam_decoder.state_dict(),
        'prototypes_state_dict':               learnable_prototypes_model.state_dict(),
        'optimizer_state_dict':                optimiser.state_dict(),
        'lambda_clip':                         args.lambda_clip,
        'clip_model':                          args.clip_model,
        'class_weights':                       'freq' if USE_CLS_WEIGHT else 'uniform',
        'clip_align_loss_state_dict':          clip_align_loss_model.state_dict()
                                               if clip_align_loss_model is not None
                                               else None,
    }
    if (epoch + 1) % 50 == 0:
        path = osp.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save(ckp, path)
        print(f"  [Checkpoint] Saved: {path}")
        existing = sorted(
            [f for f in os.listdir(save_dir)
             if f.startswith("checkpoint_epoch_") and f.endswith(".pth")],
            key=lambda x: int(x.replace("checkpoint_epoch_", "").replace(".pth", ""))
        )
        while len(existing) > 3:
            old = osp.join(save_dir, existing.pop(0))
            os.remove(old)
    if is_best:
        torch.save(ckp, osp.join(save_dir, 'model_ckp.pth'))
        print(f"  [Checkpoint] Best: epoch {epoch}, "
              f"IoU={best_challenge_iou_val:.4f}")


# =============================================================================
# Training Loop
# =============================================================================

print("======> Start Training and Validation")
# best_challenge_iou_val = -100.0

for epoch in range(start_epoch, num_epochs):

    if anneal and clip_align_loss_model is not None:
        lambda_max = 0.5
        lambda_min = 0.05
        progress   = epoch / num_epochs
        clip_align_loss_model.lambda_clip = (
            lambda_min + 0.5 * (lambda_max - lambda_min) *
            (1 + np.cos(np.pi * progress))
        )

    version = 0 if epoch % 2 == 0 else int((epoch % 80 + 1) / 2)

    if "18" in dataset_name:
        train_dataset = Endovis18Dataset(data_root_dir=data_root_dir,
                                         mode="train", vit_mode=vit_mode,
                                         version=version)
    elif "17" in dataset_name:
        train_dataset = Endovis17Dataset(data_root_dir=data_root_dir,
                                         mode="train", fold=fold,
                                         vit_mode=vit_mode, version=version)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=4)

    protoype_prompt_encoder.train()
    sam_decoder.train()
    learnable_prototypes_model.train()

    epoch_seg   = 0.0
    epoch_cont  = 0.0
    epoch_clip  = 0.0
    n_batches   = 0

    for sam_feats, _, cls_ids, masks, class_embeddings in train_dataloader:

        sam_feats        = sam_feats.cuda()
        cls_ids          = cls_ids.cuda()
        masks            = masks.cuda()
        class_embeddings = class_embeddings.cuda()

        prototypes = learnable_prototypes_model()

        preds, _ = model_forward_function(
            protoype_prompt_encoder, sam_prompt_encoder,
            sam_decoder, sam_feats, prototypes, cls_ids)

        seg_loss         = seg_loss_model(preds, masks / 255)
        contrastive_loss = contrastive_loss_model(
            prototypes,
            torch.tensor([i for i in range(1, prototypes.size()[0] + 1)]).cuda(),
            ref_emb=class_embeddings, ref_labels=cls_ids)

        # ── CLIP alignment loss — the new term ────────────────────────────
        if clip_align_loss_model is not None:
            clip_loss = clip_align_loss_model(prototypes)
        else:
            clip_loss = torch.tensor(0.0, device="cuda")
        # ──────────────────────────────────────────────────────────────────

        loss = seg_loss + contrastive_loss + clip_loss

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        epoch_seg   += seg_loss.item()
        epoch_cont  += contrastive_loss.item()
        epoch_clip  += clip_loss.item()
        n_batches   += 1

    avg_seg  = epoch_seg  / n_batches
    avg_cont = epoch_cont / n_batches
    avg_clip = epoch_clip / n_batches

    print_log(
        f"Epoch {epoch}/{num_epochs - 1} | "
        f"seg={avg_seg:.4f} "
        f"cont={avg_cont:.4f} "
        f"clip={avg_clip:.4f}",
        log_file
    )

    # ── Validation — identical to original ───────────────────────────────────
    binary_masks = dict()
    protoype_prompt_encoder.eval()
    sam_decoder.eval()
    learnable_prototypes_model.eval()

    with torch.no_grad():
        prototypes = learnable_prototypes_model()

        for sam_feats, mask_names, cls_ids, _, _ in val_dataloader:
            sam_feats = sam_feats.cuda()
            cls_ids   = cls_ids.cuda()
            preds, preds_quality = model_forward_function(
                protoype_prompt_encoder, sam_prompt_encoder,
                sam_decoder, sam_feats, prototypes, cls_ids)
            binary_masks = create_binary_masks(
                binary_masks, preds, preds_quality, mask_names, thr)

    endovis_masks   = create_endovis_masks(binary_masks, 1024, 1280)
    endovis_results = eval_endovis(endovis_masks, gt_endovis_masks)

    print_log(
        f"Validation - Epoch: {epoch}/{num_epochs - 1}; "
        f"IoU_Results: {endovis_results}",
        log_file
    )

    is_best = endovis_results["challengIoU"] > best_challenge_iou_val
    if is_best:
        best_challenge_iou_val = endovis_results["challengIoU"]
        print_log(
            f"Best Challenge IoU: {best_challenge_iou_val:.4f} at Epoch {epoch}",
            log_file
        )

    save_checkpoint(epoch, is_best=is_best)
