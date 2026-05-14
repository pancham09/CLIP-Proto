import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

INSTRUMENT_DESCRIPTIONS = {
    "bipolar_forceps": [
        "bipolar forceps surgical instrument for tissue coagulation",
        "bipolar forceps with two metallic jaw tips and insulated shaft",
        "grasping and coagulating bipolar forceps used in laparoscopic surgery",
        "silver metallic bipolar forceps with curved jaw and cable attachment",
    ],
    "prograsp_forceps": [
        "prograsp forceps robotic surgical grasping instrument",
        "prograsp forceps with serrated jaw tips for tissue manipulation",
        "intuitive prograsp forceps with articulated wrist for robotic surgery",
        "robotic prograsp forceps with fenestrated jaw design",
    ],

#    "large_needle_driver": [
#        "large needle driver for suturing in robotic surgery",
#        "needle driver with narrow serrated jaw tip for needle grasping",
#        "robotic large needle driver with tungsten carbide jaw insert",
#        "large needle driver instrument with curved shaft and needle holding tip",
#    ],
    "large_needle_driver": [
        "large needle driver with narrow elongated jaw for grasping suture needles",
        "robotic needle holder with thin cylindrical shaft and tungsten carbide tip",
        "needle driver instrument designed specifically for passing curved needles through tissue",
        "slender jaw needle driver with locking ratchet mechanism for needle control",
    ],

    "monopolar_curved_scissors": [
        "monopolar curved scissors for cutting tissue in laparoscopic surgery",
        "curved scissors with monopolar electrosurgical capability",
        "monopolar scissors with curved blades and insulated shaft",
        "silver curved scissors surgical instrument with cutting blades",
    ],
#    "vessel_sealer": [
#        "vessel sealer for sealing and dividing blood vessels",
#        "vessel sealer with broad flat jaw for vascular surgery",
#        "advanced bipolar vessel sealing instrument with wide jaw",
#        "vessel sealing instrument with flat paddle-shaped jaw tips",
#    ],
    "vessel_sealer": [
        "vessel sealer with wide flat paddle jaw for simultaneous sealing and cutting vessels",
        "advanced bipolar vessel sealing device with broad blunt jaw tips",
        "vessel sealing instrument with large surface area jaw for vascular hemostasis",
        "wide jaw vessel sealer that fuses tissue layers using bipolar energy",
    ],
    "grasping_retractor": [
        "grasping retractor for tissue retraction in robotic surgery",
        "retractor with multiple finger-like projections for tissue holding",
        "robotic grasping retractor with fan-shaped tip for retraction",
        "large grasping retractor instrument for organ displacement",
    ],
    "ultrasound_probe": [
        "laparoscopic ultrasound probe for intraoperative imaging",
        "ultrasound probe with flat scanning tip for tissue assessment",
        "intraoperative ultrasound probe with cable and scanning head",
        "linear array ultrasound probe for surgical use",
    ],
    "suction_instrument": [
        "suction irrigation instrument for fluid removal in surgery",
        "suction cannula with hollow tube tip for aspirating fluids",
        "laparoscopic suction irrigator with metal tip and flexible tube",
        "surgical suction instrument with cylindrical nozzle and thin shaft",
    ],
    "clip_applier": [
        "clip applier for placing surgical clips on vessels",
        "clip applying instrument with spring-loaded jaw mechanism",
        "laparoscopic clip applier with metallic jaw for hemostasis",
        "surgical clip applier with locking mechanism and clip magazine",
    ],
}

# Class order must match cls_ids in dataset annotations
ENDOVIS_2017_CLASSES = [
    "bipolar_forceps", "prograsp_forceps", "large_needle_driver",
    "vessel_sealer", "grasping_retractor", "monopolar_curved_scissors",
    "ultrasound_probe",
]
ENDOVIS_2018_CLASSES = [
    "bipolar_forceps", "prograsp_forceps", "large_needle_driver",
    "suction_instrument", "clip_applier", "monopolar_curved_scissors",
    "ultrasound_probe",
]


# =============================================================================
# Class frequency weights from training data
# =============================================================================

def compute_class_frequency_weights_2(
    data_root_dir: str,
    dataset_name: str,
    fold: int = 0,
    num_classes: int = 7,
    smoothing: float = 0.1,
) -> torch.Tensor:

    counts = np.zeros(num_classes, dtype=np.float32) + smoothing

    # Read the mappings.json to find which sequences belong to training
    # for this fold � same logic the dataset class uses internally
    import json
    mappings_path = os.path.join(data_root_dir, "mappings.json")

    train_seqs = None
    if os.path.exists(mappings_path):
        with open(mappings_path) as f:
            mappings = json.load(f)
        # mappings likely has fold->train_seqs or similar structure
        # print it so we can verify the key structure
        fold_key = str(fold)
        if fold_key in mappings:
            train_seqs = mappings[fold_key].get("train", None)
            print(f"  [ClassWeights] Fold {fold} train seqs: {train_seqs}")

    cls_emb_dir = os.path.join(data_root_dir, "0", "class_embeddings_h")
    if not os.path.exists(cls_emb_dir):
        print(f"  [ClassWeights] {cls_emb_dir} not found - using uniform weights")
        return torch.ones(num_classes)

    mask_files = glob.glob(os.path.join(cls_emb_dir, "**", "*.npy"),
                           recursive=True)

    for fpath in mask_files:
        # filter to training sequences only if we know which they are
        if train_seqs is not None:
            seq_name = os.path.basename(os.path.dirname(fpath))  # e.g. "seq3"
            if seq_name not in train_seqs and seq_name.replace("seq","") not in train_seqs:
                continue

        fname = os.path.basename(fpath)
        if "_class" in fname:
            try:
                cls_idx = int(fname.split("_class")[-1].replace(".npy", "")) - 1
                if 0 <= cls_idx < num_classes:
                    counts[cls_idx] += 1
            except ValueError:
                continue

    inv_freq = 1.0 / counts
    weights  = inv_freq / inv_freq.mean()
    weights  = torch.from_numpy(weights).float()

    print(f"  [ClassWeights] Per-class counts (fold {fold} train only): "
          f"{counts.astype(int).tolist()}")
    print(f"  [ClassWeights] Inverse-frequency weights: "
          f"{[f'{w:.2f}' for w in weights.tolist()]}")

    return weights


def compute_class_frequency_weights(
    data_root_dir: str,
    dataset_name: str,
    fold: int = 0,
    num_classes: int = 7,
    smoothing: float = 0.1,
) -> torch.Tensor:

    # Count class_embeddings files per class — these exist for every
    # annotated frame so they proxy for training sample count per class
    # class_embeddings_h/ has filenames like frame000_BipolarForceps.npy
    # The class index is encoded in cls_ids during dataset loading.
    # We count annotation masks in binary_annotations/ instead — more reliable.

    counts = np.zeros(num_classes, dtype=np.float32) + smoothing

    # Version 0 is the base (unaugmented) training data
    # Use only version 0 to avoid counting augmented copies multiple times
    if "17" in dataset_name:
        # EndoVis2017: training sequences depend on fold
        # Folds 0-3 each hold out 2 sequences. Standard split from SurgicalSAM.
        # Count masks in version 0 binary_annotations
        ann_dir = os.path.join(data_root_dir, "0", "binary_annotations")
    else:
        ann_dir = os.path.join(data_root_dir, "0", "binary_annotations")

    if not os.path.exists(ann_dir):
        print(f"  [ClassWeights] binary_annotations not found at {ann_dir}")
        print(f"  Using uniform weights (all 1.0)")
        return torch.ones(num_classes)

    # Count mask files per class
    # Filenames are like: frame000_BipolarForceps.png, frame000_PrograspForceps.png
    # We use the class_embeddings_h directory which has one file per mask
    cls_emb_dir = os.path.join(data_root_dir, "0", "class_embeddings_h")
    if not os.path.exists(cls_emb_dir):
        cls_emb_dir = ann_dir

    mask_files = glob.glob(os.path.join(cls_emb_dir, "**", "*.npy"),
                           recursive=True)
    if not mask_files:
        mask_files = glob.glob(os.path.join(ann_dir, "**", "*.png"),
                               recursive=True)

    if "17" in dataset_name:
        class_keywords = [
            ["bipolar", "BipolarForceps", "Bipolar_Forceps"],
            ["prograsp", "PrograspForceps", "Prograsp_Forceps"],
            ["needle", "LargeNeedleDriver", "Large_Needle_Driver"],
            ["vessel", "VesselSealer", "Vessel_Sealer"],
            ["retractor", "GraspingRetractor", "Grasping_Retractor"],
            ["scissors", "MonopolarCurvedScissors", "Monopolar_Curved_Scissors"],
            ["ultrasound", "UltrasoundProbe", "Ultrasound_Probe"],
        ]
    else:
        class_keywords = [
            ["bipolar", "BipolarForceps", "Bipolar_Forceps"],
            ["prograsp", "PrograspForceps", "Prograsp_Forceps"],
            ["needle", "LargeNeedleDriver", "Large_Needle_Driver"],
            ["suction", "SuctionInstrument", "Suction_Instrument"],
            ["clip", "ClipApplier", "Clip_Applier"],
            ["scissors", "MonopolarCurvedScissors", "Monopolar_Curved_Scissors"],
            ["ultrasound", "UltrasoundProbe", "Ultrasound_Probe"],
        ]

    for fpath in mask_files:
        fname = os.path.basename(fpath).lower()
        for cls_idx, keywords in enumerate(class_keywords):
            if any(kw.lower() in fname for kw in keywords):
                counts[cls_idx] += 1
                break

    # Inverse frequency: rare classes get higher weight
    inv_freq = 1.0 / counts
    # Normalize: mean weight = 1.0 so overall loss scale is preserved
    weights = inv_freq / inv_freq.mean()
    weights = torch.from_numpy(weights).float()

    print(f"  [ClassWeights] Per-class sample counts: {counts.astype(int).tolist()}")
    print(f"  [ClassWeights] Inverse-frequency weights: "
          f"{[f'{w:.2f}' for w in weights.tolist()]}")

    return weights


# =============================================================================
# CLIP text anchors
# =============================================================================
def build_clip_anchors(
    dataset_name: str,
    feat_dim: int = 256,   # kept for API compatibility, no longer used for projection
    clip_model_name: str = "ViT-B/32",
    device: str = "cuda",
) -> torch.Tensor:
    try:
        import clip
    except ImportError:
        raise ImportError(
            "CLIP not installed. Run:\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        )

    if "17" in dataset_name:
        class_list = ENDOVIS_2017_CLASSES
    else:
        class_list = ENDOVIS_2018_CLASSES

    print(f"\n  [CLIPAnchors] Encoding {len(class_list)} classes with {clip_model_name}")

    clip_encoder, _ = clip.load(clip_model_name, device=device)
    clip_encoder.eval()
    clip_dim = clip_encoder.text_projection.shape[1]

    all_embeddings = []
    with torch.no_grad():
        for cls_name in class_list:
            descs     = INSTRUMENT_DESCRIPTIONS[cls_name]
            desc_embs = []
            for desc in descs:
                tokens = clip.tokenize([desc]).to(device)
                feat   = clip_encoder.encode_text(tokens)  # (1, clip_dim)
                feat   = feat / feat.norm(dim=-1, keepdim=True)
                desc_embs.append(feat.squeeze(0))
            avg = torch.stack(desc_embs).float().mean(0)
            avg = avg / (avg.norm() + 1e-8)
            all_embeddings.append(avg)

    text_embs = torch.stack(all_embeddings)  # (num_classes, clip_dim)

    # Project clip_dim → feat_dim via SVD to preserve inter-class distances
    anchors = text_embs.float()   # keep native 512-dim, no projection
    # if clip_dim != feat_dim:
    #     text_embs = text_embs.float()                         # (7, 512)
        
    #     # SVD fails here because num_classes=7 < feat_dim=256
    #     # Use random orthogonal projection instead:
    #     # Generate a fixed random matrix (512 → 256), same seed every run
    #     torch.manual_seed(42)
    #     R = torch.randn(clip_dim, feat_dim, device=text_embs.device)
    #     # Orthogonalize via QR for better distance preservation
    #     R, _ = torch.linalg.qr(R)                            # (512, 256)
    #     anchors = text_embs @ R                               # (7, 256)
        
    #     print(f"  [CLIPAnchors] Random projection: {clip_dim} → {feat_dim}, "
    #         f"anchors shape: {anchors.shape}")
    # else:
    #     anchors = text_embs.float()

    anchors = F.normalize(anchors, dim=1)
    # assert anchors.shape == (len(class_list), feat_dim), \
    #     f"Expected ({len(class_list)}, {feat_dim}), got {anchors.shape}"

    # Print inter-class cosine similarities to verify separation
    sim = anchors @ anchors.T
    off = sim.clone().fill_diagonal_(-1)
    max_sim  = off.max().item()
    max_pair = (off == off.max()).nonzero(as_tuple=False)[0].tolist()
    print(f"  [CLIPAnchors] Most similar pair: "
          f"{class_list[max_pair[0]]} ↔ {class_list[max_pair[1]]} "
          f"(cosine={max_sim:.3f})")
    if max_sim > 0.85:
        print(f"  [CLIPAnchors] WARNING: high similarity {max_sim:.3f}. "
              f"CLIP text may not separate these classes well. "
              f"Consider adding more distinctive descriptions.")

    del clip_encoder
    torch.cuda.empty_cache()

    return anchors.detach()   # frozen — no gradients ever flow through this


# =============================================================================
# CLIP Text Alignment Loss
# =============================================================================

class CLIPTextAlignLoss(nn.Module):

########
    def __init__(self, clip_anchors, class_weights, lambda_clip=0.5, proto_dim=256):
        super().__init__()
        self.register_buffer("clip_anchors", clip_anchors)   # (7, 512) raw CLIP
        self.register_buffer("class_weights", class_weights)
        self.lambda_clip = lambda_clip
        clip_dim = clip_anchors.shape[1]

        # Project prototypes UP to CLIP space for comparison
        # This is learnable — it learns to map SAM's 256-dim space to CLIP's 512-dim
        # Initialized to approximate identity (via kaiming) so loss starts meaningful
        if clip_dim != proto_dim:
            self.proto_proj = nn.Linear(proto_dim, clip_dim, bias=False)
        else:
            self.proto_proj = nn.Identity()
    # def __init__(
    #     self,
    #     clip_anchors: torch.Tensor,
    #     class_weights: torch.Tensor,
    #     lambda_clip: float = 0.5,
    # ):
    #     super().__init__()

    #     # Register as buffer — moves with .cuda(), saved in state_dict,
    #     # but NOT treated as a trainable parameter
    #     self.register_buffer("clip_anchors", clip_anchors)
    #     self.register_buffer("class_weights", class_weights)
    #     self.lambda_clip = lambda_clip


    def forward(self, prototypes):
        proto_projected = self.proto_proj(prototypes)              # (7, 512)
        proto_norm      = F.normalize(proto_projected, dim=1)
        anchor_norm     = F.normalize(self.clip_anchors, dim=1)
        cos_sim         = (proto_norm * anchor_norm).sum(dim=1)    # (7,)
        per_class_loss  = self.class_weights * (1.0 - cos_sim)
        return self.lambda_clip * per_class_loss.mean()
    # def forward(self, prototypes: torch.Tensor) -> torch.Tensor:
    #     # Normalize both for cosine similarity
    #     proto_norm  = F.normalize(prototypes, dim=1)   # (C, D)
    #     anchor_norm = F.normalize(self.clip_anchors, dim=1)  # (C, D)

    #     # Cosine similarity per class: (C,)
    #     cos_sim = (proto_norm * anchor_norm).sum(dim=1)

    #     # Alignment loss: 1 - cosine_sim, weighted by class frequency
    #     # Higher weight for rare classes → stronger text supervision for SI, CA, GR
    #     per_class_loss = self.class_weights * (1.0 - cos_sim)

    #     return self.lambda_clip * per_class_loss.mean()

    def extra_repr(self) -> str:
        return (f"num_classes={self.clip_anchors.shape[0]}, "
                f"feat_dim={self.clip_anchors.shape[1]}, "
                f"lambda_clip={self.lambda_clip}")