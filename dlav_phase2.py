"""
DLAV Phase 2 — Full Script
==========================
Improvements over Phase 2 baseline:
  1. Dataset       : Phase-1-style augmentation + ImageNet normalisation
                     + depth/segmentation loading
  2. Backbone      : ResNet-18 feature pyramid (4 scales) — no detach,
                     so ALL aux-task gradients flow into the backbone
  3. History enc.  : Transformer encoder with [CLS] token + learnable
                     positional embeddings (replaces 1D-CNN)
  4. Command cond. : Embedding → MLP → FiLM + double-concat at head
                     (Phase 2 baseline had no command at all)
  5. Traj decoder  : Coarse-to-fine polynomial with kinematic prior
                     (coarse K=3 → fine K=5, both supervised)
  6. Aux heads     : Depth (L1) + Segmentation (CrossEntropy), UNet-style
                     decoders from the /32 feature map, gradients flow
  7. Training      : AdamW + dual LR groups + CosineAnnealing + grad clip
                     + best-model tracking on ADE
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. SETUP
# ─────────────────────────────────────────────────────────────────────────────
import os
import copy
import random
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # Data
    train_dir        = "train",
    val_dir          = "val",
    test_dir         = "test_public",
    num_workers      = 2,

    # Model
    degree_coarse    = 3,
    degree_fine      = 5,
    num_timesteps    = 60,
    img_feat_dim     = 512,      # ResNet-18 layer4 output channels
    hist_d_model     = 128,
    hist_nhead       = 4,
    hist_layers      = 2,
    hist_feat_dim    = 128,
    cmd_embed_dim    = 64,
    fusion_dim       = 256,
    dropout          = 0.2,
    freeze_backbone  = 2,        # freeze first N ResNet stages
    velocity_window  = 5,
    num_seg_classes  = 19,       # Cityscapes-style; adjust if different

    # Aux task flags
    use_depth        = True,
    use_seg          = True,     # set False if no seg labels in your data

    # Loss weights
    lambda_coarse    = 0.3,
    lambda_depth     = 0.05,
    lambda_seg       = 0.1,

    # Training
    batch_size       = 32,
    num_epochs       = 50,
    lr_backbone      = 1e-4,
    lr_head          = 1e-3,
    weight_decay     = 1e-4,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────────────────────────────────────
class DrivingDataset(Dataset):
    """
    Loads each .pkl sample and returns:
      camera   : (3, H, W)  float32, ImageNet-normalised
      history  : (21, 3)    float32  [x, y, heading]
      command  : ()         int64    0=forward, 1=left, 2=right
      future   : (60, 3)    float32  [x, y, heading]  — absent for test
      depth    : (H, W, 1)  float32  — absent if not in file
      seg      : (H, W)     int64    — absent if not in file
    """
    COMMAND_TO_IDX = {"forward": 0, "left": 1, "right": 2}

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    MAX_CROP_TOP = 30
    MAX_HSHIFT   = 8

    def __init__(self, file_list, train=False, test=False):
        self.samples = file_list
        self.train   = train
        self.test    = test

        if train:
            self.color_aug = transforms.Compose([
                transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                       saturation=0.15, hue=0.05),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.08),
                                         ratio=(0.3, 3.3), value=0),
            ])
        else:
            self.color_aug = None

    # ── spatial augmentations ───────────────────────────────────────────────
    def _random_crop_top(self, img):
        crop = torch.randint(0, self.MAX_CROP_TOP + 1, (1,)).item()
        if crop == 0:
            return img
        cropped = img[:, crop:, :]
        return F.interpolate(cropped.unsqueeze(0), size=(200, 300),
                             mode="bilinear", align_corners=False).squeeze(0)

    def _random_hshift(self, img):
        shift = torch.randint(-self.MAX_HSHIFT, self.MAX_HSHIFT + 1, (1,)).item()
        if shift == 0:
            return img
        shifted = torch.roll(img, shifts=shift, dims=2)
        if shift > 0:
            shifted[:, :, :shift] = 0
        else:
            shifted[:, :, shift:] = 0
        return shifted

    # ── main ────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        with open(self.samples[idx], "rb") as f:
            data = pickle.load(f)

        # Camera
        camera = torch.FloatTensor(data["camera"]).permute(2, 0, 1) / 255.0
        if self.train:
            camera = self._random_crop_top(camera)
            camera = self._random_hshift(camera)
            camera = self.color_aug(camera)
        camera = (camera - self.IMAGENET_MEAN) / self.IMAGENET_STD

        history = torch.FloatTensor(data["sdc_history_feature"])
        command = torch.tensor(
            self.COMMAND_TO_IDX[data["driving_command"]], dtype=torch.long
        )

        sample = {"camera": camera, "history": history, "command": command}

        if not self.test:
            sample["future"] = torch.FloatTensor(data["sdc_future_feature"])

        # Optional aux labels
        if "depth" in data:
            sample["depth"] = torch.FloatTensor(data["depth"])        # (H,W,1)
        if "seg" in data:
            sample["seg"] = torch.LongTensor(data["seg"].astype(np.int64))  # (H,W)

        return sample


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

# 3-a  Transformer history encoder
class TransformerHistoryEncoder(nn.Module):
    """
    Encodes (B, 21, 3) → (B, out_dim) with a Transformer encoder.

    • Learnable positional embeddings (short sequence → no need for sinusoidal)
    • [CLS] token for aggregation (attends to all steps freely)
    • Pre-LayerNorm (norm_first=True) for stable training
    """
    def __init__(self, in_dim=3, d_model=128, nhead=4, num_layers=2,
                 out_dim=128, dropout=0.1, max_len=21):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_embed  = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LN: more stable for small T
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, history):
        B, T, _ = history.shape
        x   = self.input_proj(history)                         # (B, T, d_model)
        pos = torch.arange(T, device=history.device)
        x   = x + self.pos_embed(pos).unsqueeze(0)            # (B, T, d_model)

        cls = self.cls_token.expand(B, -1, -1)                # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                      # (B, T+1, d_model)
        x   = self.transformer(x)                             # (B, T+1, d_model)

        return self.out_proj(x[:, 0, :])                      # (B, out_dim)


# 3-b  FiLM — feature-wise linear modulation by driving command
class FiLM(nn.Module):
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, feat_dim)
        self.to_beta  = nn.Linear(cond_dim, feat_dim)
        nn.init.zeros_(self.to_gamma.weight); nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight);  nn.init.zeros_(self.to_beta.bias)

    def forward(self, x, cond):
        return self.to_gamma(cond) * x + self.to_beta(cond)


# 3-c  Polynomial trajectory decoder with kinematic prior
class PolynomialTrajectoryDecoder(nn.Module):
    """
    trajectory = linear_prior(history) + polynomial_residual(coeffs)

    The model only predicts the deviation from a constant-velocity
    straight-line extrapolation, removing the need to learn scale.
    """
    def __init__(self, degree=5, num_timesteps=60, velocity_window=5):
        super().__init__()
        self.degree        = degree
        self.num_timesteps = num_timesteps
        self.velocity_window = velocity_window

        t = torch.linspace(0, 1, num_timesteps)
        V = torch.stack([t ** (k + 1) for k in range(degree)], dim=1)  # (T, K)
        self.register_buffer("V", V)

        t_lin = torch.arange(1, num_timesteps + 1, dtype=torch.float32)
        self.register_buffer("t_linear", t_lin)

    def compute_prior(self, history):
        k         = self.velocity_window
        past_pos  = history[:, -1 - k, :2]          # (B, 2)
        vel       = -past_pos / k                    # (B, 2)
        prior     = self.t_linear.view(1, -1, 1) * vel.unsqueeze(1)  # (B, T, 2)
        return prior

    def forward(self, coeffs, history):
        """coeffs: (B, 2, K), history: (B, 21, 3) → (B, T, 2)"""
        prior    = self.compute_prior(history)
        residual = torch.einsum("tk,bak->bta", self.V, coeffs)
        return prior + residual


# 3-d  Coarse-to-fine decoder
class CoarseToFineDecoder(nn.Module):
    """
    Stage 1 (coarse): low-degree polynomial → global direction
    Stage 2 (fine):   high-degree polynomial conditioned on coarse output → local refinement

    Both outputs are supervised; coarse uses a smoothed target.
    """
    def __init__(self, fusion_dim, degree_coarse=3, degree_fine=5,
                 num_timesteps=60, velocity_window=5, dropout=0.1):
        super().__init__()
        self.T = num_timesteps

        # Coarse head
        self.coarse_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 2 * degree_coarse),
        )
        self.coarse_decoder = PolynomialTrajectoryDecoder(
            degree_coarse, num_timesteps, velocity_window)

        # Fine head — takes latent + encoded coarse trajectory
        self.coarse_enc = nn.Sequential(
            nn.Linear(num_timesteps * 2, 64), nn.ReLU(),
        )
        self.fine_head = nn.Sequential(
            nn.Linear(fusion_dim + 64, fusion_dim // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 2 * degree_fine),
        )
        self.fine_decoder = PolynomialTrajectoryDecoder(
            degree_fine, num_timesteps, velocity_window)

        # Zero-init last layers so training starts near prior
        nn.init.zeros_(self.coarse_head[-1].weight)
        nn.init.zeros_(self.coarse_head[-1].bias)
        nn.init.zeros_(self.fine_head[-1].weight)
        nn.init.zeros_(self.fine_head[-1].bias)

    def forward(self, latent, history):
        B = latent.size(0)

        # Stage 1: coarse
        coarse_coeffs = self.coarse_head(latent).view(B, 2, -1)
        coarse_traj   = self.coarse_decoder(coarse_coeffs, history)   # (B, T, 2)

        # Stage 2: fine, conditioned on coarse
        coarse_enc    = self.coarse_enc(coarse_traj.view(B, -1))
        fine_input    = torch.cat([latent, coarse_enc], dim=1)
        fine_coeffs   = self.fine_head(fine_input).view(B, 2, -1)
        fine_residual = self.fine_decoder(fine_coeffs, history)        # (B, T, 2)

        fine_traj = coarse_traj + fine_residual
        return coarse_traj, fine_traj


# 3-e  UNet-style aux decoder (shared structure for depth & seg)
def build_aux_decoder(in_channels, out_channels, target_size=(200, 300)):
    """
    4-stage ConvTranspose upsampling: /32 → full resolution.
    Gradients flow freely into the backbone (no detach).
    """
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, 256, 4, stride=2, padding=1), nn.ReLU(inplace=True),
        nn.ConvTranspose2d(256,         128, 4, stride=2, padding=1), nn.ReLU(inplace=True),
        nn.ConvTranspose2d(128,          64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
        nn.ConvTranspose2d( 64,          32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(32, out_channels, 3, padding=1),
        nn.Upsample(size=target_size, mode="bilinear", align_corners=False),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────
class DrivingPlannerV2(nn.Module):
    """
    Full Phase-2 model with every improvement stacked:

    Image (3,H,W) ──► ResNet-18 feature pyramid ──► GAP ──► 512-d
                              │
                              └──► depth decoder   (aux, UNet)
                              └──► seg   decoder   (aux, UNet)

    History (21,3) ──► Transformer encoder ──► 128-d

    Command ──► Embedding → MLP ──► 64-d (FiLM + concat)

    [512 + 128] ──► Fusion MLP ──► FiLM(cmd) ──► latent
                                                    │
                                                    └──► Coarse-to-Fine decoder
                                                         ├─ coarse traj (K=3)
                                                         └─ fine   traj (K=5)
    """
    def __init__(self, cfg):
        super().__init__()
        self.use_depth = cfg["use_depth"]
        self.use_seg   = cfg["use_seg"]

        # ── Backbone: ResNet-18 split into stages ────────────────────────────
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stage0 = nn.Sequential(*list(resnet.children())[:5])  # /4,   64ch
        self.stage1 = resnet.layer2                                 # /8,  128ch
        self.stage2 = resnet.layer3                                 # /16, 256ch
        self.stage3 = resnet.layer4                                 # /32, 512ch
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self._freeze_stages(cfg["freeze_backbone"])

        # ── History encoder (Transformer) ────────────────────────────────────
        self.history_encoder = TransformerHistoryEncoder(
            in_dim=3,
            d_model=cfg["hist_d_model"],
            nhead=cfg["hist_nhead"],
            num_layers=cfg["hist_layers"],
            out_dim=cfg["hist_feat_dim"],
            dropout=cfg["dropout"],
        )

        # ── Command conditioning ──────────────────────────────────────────────
        ced = cfg["cmd_embed_dim"]
        self.command_embedding = nn.Embedding(3, ced)
        self.command_mlp = nn.Sequential(
            nn.Linear(ced, ced), nn.ReLU(inplace=True), nn.Linear(ced, ced),
        )

        # ── Fusion MLP + FiLM ─────────────────────────────────────────────────
        in_dim = cfg["img_feat_dim"] + cfg["hist_feat_dim"]
        fd     = cfg["fusion_dim"]
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, fd), nn.ReLU(inplace=True), nn.Dropout(cfg["dropout"]),
            nn.Linear(fd, fd),     nn.ReLU(inplace=True), nn.Dropout(cfg["dropout"]),
        )
        self.film = FiLM(cond_dim=ced, feat_dim=fd)

        # ── Coarse-to-fine trajectory decoder ────────────────────────────────
        self.traj_decoder = CoarseToFineDecoder(
            fusion_dim    = fd + ced,                 # latent + cmd concat
            degree_coarse = cfg["degree_coarse"],
            degree_fine   = cfg["degree_fine"],
            num_timesteps = cfg["num_timesteps"],
            velocity_window = cfg["velocity_window"],
            dropout       = cfg["dropout"],
        )

        # ── Aux decoders (from /32 feature map, 512 channels) ────────────────
        if self.use_depth:
            self.depth_decoder = build_aux_decoder(512, 1)
            # Sigmoid applied in forward; loss uses L1 on [0,1]-normalised depth

        if self.use_seg:
            self.seg_decoder = build_aux_decoder(512, cfg["num_seg_classes"])
            # No activation; CrossEntropyLoss expects raw logits

    # ── helpers ──────────────────────────────────────────────────────────────
    def _freeze_stages(self, n):
        boundaries = {1: 5, 2: 6, 3: 7, 4: 8}
        limit = boundaries.get(n, 0)
        for i, child in enumerate(
            [self.stage0, self.stage1, self.stage2, self.stage3]
        ):
            if i < limit:
                for p in child.parameters():
                    p.requires_grad = False

    def get_param_groups(self, lr_backbone, lr_head, weight_decay):
        backbone_params = [
            p for stage in [self.stage0, self.stage1, self.stage2, self.stage3]
            for p in stage.parameters() if p.requires_grad
        ]
        head_modules = [
            self.gap, self.history_encoder,
            self.command_embedding, self.command_mlp,
            self.fusion, self.film, self.traj_decoder,
        ]
        if self.use_depth: head_modules.append(self.depth_decoder)
        if self.use_seg:   head_modules.append(self.seg_decoder)
        head_params = [p for m in head_modules for p in m.parameters()]
        return [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": head_params,     "lr": lr_head,     "weight_decay": weight_decay},
        ]

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, camera, history, command):
        B = camera.size(0)

        # Feature pyramid — gradients flow from ALL tasks into backbone
        f0 = self.stage0(camera)          # (B, 64,  H/4,  W/4)
        f1 = self.stage1(f0)              # (B, 128, H/8,  W/8)
        f2 = self.stage2(f1)              # (B, 256, H/16, W/16)
        f3 = self.stage3(f2)              # (B, 512, H/32, W/32)

        feat_img  = self.gap(f3).flatten(1)                            # (B, 512)
        feat_hist = self.history_encoder(history)                      # (B, 128)
        feat_cmd  = self.command_mlp(self.command_embedding(command))  # (B, 64)

        fused  = self.fusion(torch.cat([feat_img, feat_hist], dim=1))  # (B, fd)
        latent = self.film(fused, feat_cmd)                            # (B, fd)

        # Double-inject command (FiLM + concat at head)
        latent_with_cmd = torch.cat([latent, feat_cmd], dim=1)         # (B, fd+ced)

        coarse_traj, fine_traj = self.traj_decoder(latent_with_cmd, history)

        depth_out = torch.sigmoid(self.depth_decoder(f3)) if self.use_depth else None
        seg_out   = self.seg_decoder(f3)                  if self.use_seg   else None

        return fine_traj, coarse_traj, depth_out, seg_out


# ─────────────────────────────────────────────────────────────────────────────
# 5. LOSS
# ─────────────────────────────────────────────────────────────────────────────
def smooth_trajectory(traj, degree=3, num_timesteps=60):
    """
    Fit a low-degree polynomial to the GT trajectory to create a smooth
    coarse target. Used for supervising the coarse head.
    traj: (B, T, 2)  →  smooth: (B, T, 2)
    """
    B, T, _ = traj.shape
    t     = torch.linspace(0, 1, T, device=traj.device)
    # Build Vandermonde matrix (T, degree+1)
    V = torch.stack([t ** k for k in range(degree + 1)], dim=1)   # (T, K+1)
    # Least-squares fit per batch item per axis
    traj_np    = traj.detach().cpu().numpy()
    t_np       = np.linspace(0, 1, T)
    smooth_np  = np.zeros_like(traj_np)
    for b in range(B):
        for ax in range(2):
            c = np.polyfit(t_np, traj_np[b, :, ax], deg=degree)
            smooth_np[b, :, ax] = np.polyval(c, t_np)
    return torch.tensor(smooth_np, dtype=traj.dtype, device=traj.device)


def compute_loss(fine_traj, coarse_traj, depth_out, seg_out,
                 batch, cfg, device):
    """
    Returns total scalar loss and a dict of individual components for logging.
    """
    gt_future = batch["future"].to(device)[..., :2]    # (B, T, 2)

    # Trajectory losses
    traj_loss   = F.mse_loss(fine_traj,   gt_future)
    smooth_gt   = smooth_trajectory(gt_future, degree=cfg["degree_coarse"])
    coarse_loss = F.mse_loss(coarse_traj, smooth_gt)

    total = traj_loss + cfg["lambda_coarse"] * coarse_loss
    log   = {"traj": traj_loss.item(), "coarse": coarse_loss.item()}

    # Depth aux
    if depth_out is not None and "depth" in batch:
        gt_depth  = batch["depth"].to(device)          # (B, H, W, 1)
        gt_depth  = gt_depth.permute(0, 3, 1, 2)       # (B, 1, H, W)
        # Normalise GT to [0, 1] per sample for L1
        d_max     = gt_depth.flatten(1).max(1)[0].view(-1, 1, 1, 1).clamp(min=1e-3)
        gt_depth  = gt_depth / d_max
        depth_loss = F.l1_loss(depth_out, gt_depth)
        total     = total + cfg["lambda_depth"] * depth_loss
        log["depth"] = depth_loss.item()

    # Segmentation aux
    if seg_out is not None and "seg" in batch:
        gt_seg   = batch["seg"].to(device)             # (B, H, W) — int64
        seg_loss = F.cross_entropy(seg_out, gt_seg, ignore_index=255)
        total    = total + cfg["lambda_seg"] * seg_loss
        log["seg"] = seg_loss.item()

    log["total"] = total.item()
    return total, log


# ─────────────────────────────────────────────────────────────────────────────
# 6. METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_ade_fde(pred, gt):
    """pred, gt: (B, T, 2) → scalars"""
    diff = pred - gt[..., :2]
    dist = torch.norm(diff, p=2, dim=-1)   # (B, T)
    ade  = dist.mean(dim=1).mean().item()
    fde  = dist[:, -1].mean().item()
    return ade, fde


# ─────────────────────────────────────────────────────────────────────────────
# 7. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train(model, train_loader, val_loader, optimizer, scheduler, cfg):
    model = model.to(DEVICE)

    best_ade        = float("inf")
    best_state_dict = None
    best_epoch      = -1
    history_log     = {"train_total": [], "val_ade": [], "val_fde": [], "lr": []}

    for epoch in range(cfg["num_epochs"]):

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            camera  = batch["camera"].to(DEVICE)
            history = batch["history"].to(DEVICE)
            command = batch["command"].to(DEVICE)

            optimizer.zero_grad()
            fine_traj, coarse_traj, depth_out, seg_out = model(camera, history, command)
            loss, _ = compute_loss(fine_traj, coarse_traj, depth_out, seg_out,
                                   batch, cfg, DEVICE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        ade_all, fde_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                camera  = batch["camera"].to(DEVICE)
                history = batch["history"].to(DEVICE)
                command = batch["command"].to(DEVICE)
                future  = batch["future"].to(DEVICE)

                fine_traj, *_ = model(camera, history, command)
                ade, fde = compute_ade_fde(fine_traj, future)
                ade_all.append(ade); fde_all.append(fde)

        mean_ade = float(np.mean(ade_all))
        mean_fde = float(np.mean(fde_all))

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        history_log["train_total"].append(epoch_loss)
        history_log["val_ade"].append(mean_ade)
        history_log["val_fde"].append(mean_fde)
        history_log["lr"].append(current_lr)

        is_best = mean_ade < best_ade
        if is_best:
            best_ade        = mean_ade
            best_epoch      = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())

        marker = "  ⭐" if is_best else ""
        print(
            f"Epoch {epoch+1:3d}/{cfg['num_epochs']} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Val ADE: {mean_ade:.4f} | FDE: {mean_fde:.4f} | "
            f"LR: {current_lr:.1e}{marker}"
        )

    print(f"\n✅ Best ADE: {best_ade:.4f} at epoch {best_epoch}")
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model, history_log, best_ade


# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN_NP = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
IMAGENET_STD_NP  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)


def denorm(img_tensor):
    """(3,H,W) normalised tensor → (H,W,3) uint8 numpy"""
    img = img_tensor.numpy() * IMAGENET_STD_NP + IMAGENET_MEAN_NP
    return np.clip(img.transpose(1, 2, 0), 0, 1)


def visualize(model, val_loader, k=4):
    model.eval()
    batch = next(iter(val_loader))
    camera  = batch["camera"].to(DEVICE)
    history = batch["history"].to(DEVICE)
    command = batch["command"].to(DEVICE)
    future  = batch["future"]

    with torch.no_grad():
        fine_traj, coarse_traj, depth_out, seg_out = model(camera, history, command)

    cam_np     = camera.cpu()
    hist_np    = history.cpu().numpy()
    fut_np     = future.numpy()
    fine_np    = fine_traj.cpu().numpy()
    coarse_np  = coarse_traj.cpu().numpy()

    indices = random.choices(range(len(cam_np)), k=k)

    # ── Camera images ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
    for i, idx in enumerate(indices):
        axes[i].imshow(denorm(cam_np[idx]))
        axes[i].axis("off")
        axes[i].set_title(f"Example {i+1}")
    plt.suptitle("Camera Inputs"); plt.tight_layout(); plt.show()

    # ── Trajectories ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
    for i, idx in enumerate(indices):
        axes[i].plot(hist_np[idx, :, 0],  hist_np[idx, :, 1],
                     "o-", color="gold", markersize=4, label="Past")
        axes[i].plot(fut_np[idx, :, 0],   fut_np[idx, :, 1],
                     "o-", color="green", markersize=4, label="GT")
        axes[i].plot(coarse_np[idx, :, 0], coarse_np[idx, :, 1],
                     "--", color="orange", markersize=3, label="Coarse")
        axes[i].plot(fine_np[idx, :, 0],  fine_np[idx, :, 1],
                     "o-", color="red",  markersize=3, label="Fine")
        axes[i].legend(fontsize=7); axes[i].axis("equal"); axes[i].grid(alpha=0.3)
    plt.suptitle("Trajectory: GT vs Coarse vs Fine"); plt.tight_layout(); plt.show()

    # ── Depth maps ────────────────────────────────────────────────────────
    if depth_out is not None and "depth" in batch:
        dep_pred_np = depth_out.cpu().numpy()               # (B,1,H,W)
        dep_gt_np   = batch["depth"].numpy()                # (B,H,W,1)
        fig, axes   = plt.subplots(2, k, figsize=(4 * k, 6))
        for i, idx in enumerate(indices):
            axes[0, i].imshow(dep_gt_np[idx, :, :, 0],   cmap="viridis")
            axes[0, i].set_title("GT Depth"); axes[0, i].axis("off")
            axes[1, i].imshow(dep_pred_np[idx, 0, :, :], cmap="viridis")
            axes[1, i].set_title("Pred Depth"); axes[1, i].axis("off")
        plt.suptitle("Depth Estimation"); plt.tight_layout(); plt.show()

    # ── Segmentation ──────────────────────────────────────────────────────
    if seg_out is not None and "seg" in batch:
        seg_pred_np = seg_out.argmax(1).cpu().numpy()       # (B,H,W)
        seg_gt_np   = batch["seg"].numpy()                  # (B,H,W)
        fig, axes   = plt.subplots(2, k, figsize=(4 * k, 6))
        for i, idx in enumerate(indices):
            axes[0, i].imshow(seg_gt_np[idx],   cmap="tab20")
            axes[0, i].set_title("GT Seg"); axes[0, i].axis("off")
            axes[1, i].imshow(seg_pred_np[idx], cmap="tab20")
            axes[1, i].set_title("Pred Seg"); axes[1, i].axis("off")
        plt.suptitle("Segmentation"); plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 9. SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
def make_submission(model, cfg, filename="submission_phase2.csv"):
    test_dir   = cfg["test_dir"]
    test_files = [
        os.path.join(test_dir, fn)
        for fn in sorted(
            [f for f in os.listdir(test_dir) if f.endswith(".pkl")],
            key=lambda fn: int(os.path.splitext(fn)[0]),
        )
    ]
    test_dataset = DrivingDataset(test_files, train=False, test=True)
    test_loader  = DataLoader(test_dataset, batch_size=250,
                              num_workers=cfg["num_workers"])

    model.eval()
    all_plans = []
    with torch.no_grad():
        for batch in test_loader:
            camera  = batch["camera"].to(DEVICE)
            history = batch["history"].to(DEVICE)
            command = batch["command"].to(DEVICE)
            fine_traj, *_ = model(camera, history, command)
            all_plans.append(fine_traj.cpu().numpy()[..., :2])

    all_plans = np.concatenate(all_plans, axis=0)          # (N, T, 2)
    N, T, _   = all_plans.shape
    flat      = all_plans.reshape(N, T * 2)

    cols = ["id"] + [f"{ax}_{t+1}" for t in range(T) for ax in ("x", "y")]
    df   = pd.DataFrame(
        np.hstack([np.arange(N).reshape(-1, 1), flat]),
        columns=cols,
    )
    df["id"] = df["id"].astype(int)
    df.to_csv(filename, index=False)
    print(f"✅ Saved {filename}  ({N} samples, {T} timesteps)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Data ──────────────────────────────────────────────────────────────
    train_files = [
        os.path.join(CFG["train_dir"], f)
        for f in os.listdir(CFG["train_dir"]) if f.endswith(".pkl")
    ]
    val_files = [
        os.path.join(CFG["val_dir"], f)
        for f in os.listdir(CFG["val_dir"]) if f.endswith(".pkl")
    ]
    print(f"Train: {len(train_files)} samples | Val: {len(val_files)} samples")

    train_dataset = DrivingDataset(train_files, train=True)
    val_dataset   = DrivingDataset(val_files,   train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"], shuffle=True,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"], shuffle=False,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = DrivingPlannerV2(CFG)
    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params total: {n_total:,} | Trainable: {n_trainable:,}")

    # ── Optimiser (dual LR: backbone vs head) ─────────────────────────────
    param_groups = model.get_param_groups(
        lr_backbone  = CFG["lr_backbone"],
        lr_head      = CFG["lr_head"],
        weight_decay = CFG["weight_decay"],
    )
    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["num_epochs"], eta_min=1e-6,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    model, history_log, best_ade = train(
        model, train_loader, val_loader, optimizer, scheduler, CFG,
    )
    print(f"\n🎯 Final best val ADE: {best_ade:.4f}")

    # ── Training curves ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history_log["train_total"], label="Train Loss"); axes[0].set_title("Train Loss"); axes[0].grid(alpha=0.3)
    axes[1].plot(history_log["val_ade"],     label="Val ADE");   axes[1].set_title("Val ADE");    axes[1].grid(alpha=0.3)
    axes[2].plot(history_log["val_fde"],     label="Val FDE");   axes[2].set_title("Val FDE");    axes[2].grid(alpha=0.3)
    for ax in axes: ax.legend()
    plt.tight_layout(); plt.show()

    # ── Visualise ─────────────────────────────────────────────────────────
    visualize(model, val_loader, k=4)

    # ── Submission ────────────────────────────────────────────────────────
    make_submission(model, CFG, filename="submission_phase2.csv")

    # ── Checkpoint ────────────────────────────────────────────────────────
    torch.save(model.state_dict(), "best_model_phase2.pt")
    print("💾 Checkpoint saved: best_model_phase2.pt")


if __name__ == "__main__":
    main()
