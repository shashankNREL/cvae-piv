"""
PIV Gap-Fill CVAE with 3-Phase Loss Scaling + Anti-Collapse Strategies
======================================================================
Includes four strategies to prevent posterior collapse:

  1. Target KLD loss: replaces β·KLD with β·(KLD_raw − target)²
     Two-sided gradient pushes KLD toward a user-set target, preventing
     the encoder from ignoring the latent code z.

  2. Skip-connection dropout: F.dropout(p=skip_dropout) on every skip
     connection during training. Much more effective than Gaussian noise
     at forcing the decoder to rely on z.

  3. Skip bottleneck: 1×1 convolutions halve the channel width of every
     skip connection before concatenation, reducing information flow
     through the skip path.

  4. beta_warm = 0.0: Phase 1 has zero KLD penalty, letting the encoder
     learn a non-trivial posterior before regularisation begins.

3-Phase schedule:
  Phase 1 (0 – phase1_frac):    beta = 0 (recon only),    gamma = 0
  Phase 2 (phase1_frac – phase2_frac): beta cyclical 0→max_beta, gamma = 0
  Phase 3 (phase2_frac – 100%):  beta = max_beta (frozen),  gamma ramp

Diagnostic logging
------------------
Every epoch : loss components, KLD_raw, target, active units, latent stats
Every 10 ep : per-dimension KLD breakdown
Every 50 ep : checkpoint + validation error + latent health summary

User-tunable parameters:
  --max_beta          weight for target-KLD loss term (default 1e-3)
  --target_kld        desired KLD value (sum over dims, mean over batch; default 5.0)
  --max_gamma         ceiling for vorticity loss weight (default 3e-6)
  --skip_dropout      dropout probability on skip connections (default 0.3)
  --skip_bottleneck_factor  channel reduction factor for 1×1 skip convs (default 2)
  --n_cycles          beta annealing cycles in Phase 2 (default 8)
"""

import os
import math
import datetime
import csv
import hashlib
import json
import numpy as np
from numpy import linalg as LA
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import time
import argparse
from partialconv.models import partialconv2d
from sklearn.preprocessing import RobustScaler
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib
import resource
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
except ImportError:
    ray = None
    tune = None
    ASHAScheduler = None

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

class VariableHandler:
    def __init__(self, device=torch.device("cpu"), dtype=torch.float):
        self.device = device
        self.dtype = dtype

    def tovar(self, input):
        return Variable(torch.as_tensor(input, dtype=self.dtype,
                                        device=self.device))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CVAE(nn.Module):
    def __init__(self, latent_size, num_labels, ImgSizeX, ImgSizeY,
                 leaky_relu_slope=0.2, dropout_rate=0.2,
                 skip_dropout=0.3, skip_bottleneck_factor=2, vh=None):
        super().__init__()
        self.vh = vh if vh is not None else VariableHandler()

        self.latent_size = latent_size
        self.encoder = Encoder(latent_size, num_labels, ImgSizeX, ImgSizeY,
                               leaky_relu_slope, dropout_rate).to(device=self.vh.device, dtype=self.vh.dtype)
        self.decoder = Decoder(latent_size, num_labels,
                               leaky_relu_slope, dropout_rate,
                               skip_dropout=skip_dropout,
                               skip_bottleneck_factor=skip_bottleneck_factor
                               ).to(device=self.vh.device, dtype=self.vh.dtype)

    def forward(self, x, c, mask):
        batch_size = x.size(0)
        means, log_var, skips = self.encoder(x, c, mask)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size], device=self.vh.device, dtype=self.vh.dtype)
        z = eps * std + means
        recon_x = self.decoder(z, c, skips)
        return recon_x, means, log_var, z

    def inference(self, x, c, mask):
        x = self.vh.tovar(x)
        mask = self.vh.tovar(mask)
        c = self.vh.tovar(c)
        means, _, skips = self.encoder(x, c, mask)
        decodedX = self.decoder(means, c, skips)
        return decodedX.cpu().detach().numpy()

    def load(self, fname):
        obj = torch.load(fname, map_location=lambda storage, loc: storage)
        # Accept either a bare state_dict (legacy) or a bundle dict carrying
        # state_dict + model_kwargs + config + metrics (new format).
        if isinstance(obj, dict) and 'state_dict' in obj:
            self.load_state_dict(obj['state_dict'])
        else:
            self.load_state_dict(obj)
        self.eval()


class Encoder(nn.Module):
    def __init__(self, latent_size, nLabels, ImgSizeX, ImgSizeY, leaky_relu_slope=0.2, dropout_rate=0.2):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope
        self.dropout_rate = dropout_rate

        self.EnCodeConv1 = partialconv2d.PartialConv2d(3, 16, 3, stride=2, padding=1, multi_channel=True, return_mask=True)
        self.EnCodeBN1 = nn.BatchNorm2d(16)

        self.EnCodeConv2 = partialconv2d.PartialConv2d(16, 32, 3, stride=1, padding=1, multi_channel=True, return_mask=True)
        self.EnCodeBN2 = nn.BatchNorm2d(32)

        self.EnCodeConv3 = partialconv2d.PartialConv2d(32, 64, 3, stride=2, padding=1, multi_channel=True, return_mask=True)
        self.EnCodeBN3 = nn.BatchNorm2d(64)

        self.EnCodeConv4 = partialconv2d.PartialConv2d(64, 96, 3, stride=1, padding=1, multi_channel=True, return_mask=True)
        self.EnCodeBN4 = nn.BatchNorm2d(96)

        self.EnCodeConv5 = partialconv2d.PartialConv2d(96, 128, 3, stride=2, padding=1, multi_channel=True, return_mask=True)
        self.EnCodeBN5 = nn.BatchNorm2d(128)

        self.EnCodeConv6 = partialconv2d.PartialConv2d(128, 192, 3, stride=2, padding=1, multi_channel=True, return_mask=True)
        self.EnCodeBN6 = nn.BatchNorm2d(192)

        self.EnCodeConv7 = partialconv2d.PartialConv2d(192, 256, 3, stride=1, padding=1, multi_channel=True, return_mask=True)
        self.EnCodeBN7 = nn.BatchNorm2d(256)

        self.inject_label = nn.Linear(256*5*3 + nLabels, latent_size)
        self.linear_means = nn.Linear(latent_size, latent_size)
        self.linear_log_var = nn.Linear(latent_size, latent_size)

    def forward(self, x, c, mask):
        x1, m1 = self.EnCodeConv1(x, mask)
        x1 = F.leaky_relu(self.EnCodeBN1(x1), negative_slope=self.leaky_relu_slope)

        x2, m2 = self.EnCodeConv2(x1, m1)
        x2 = F.leaky_relu(self.EnCodeBN2(x2), negative_slope=self.leaky_relu_slope)

        x3, m3 = self.EnCodeConv3(x2, m2)
        x3 = F.leaky_relu(self.EnCodeBN3(x3), negative_slope=self.leaky_relu_slope)

        x4, m4 = self.EnCodeConv4(x3, m3)
        x4 = F.leaky_relu(self.EnCodeBN4(x4), negative_slope=self.leaky_relu_slope)

        x5, m5 = self.EnCodeConv5(x4, m4)
        x5 = F.leaky_relu(self.EnCodeBN5(x5), negative_slope=self.leaky_relu_slope)

        x6, m6 = self.EnCodeConv6(x5, m5)
        x6 = F.leaky_relu(self.EnCodeBN6(x6), negative_slope=self.leaky_relu_slope)

        x7, m7 = self.EnCodeConv7(x6, m6)
        x7 = F.leaky_relu(self.EnCodeBN7(x7), negative_slope=self.leaky_relu_slope)

        x_flat = x7.view(x7.size(0), -1)
        x_c = torch.cat((x_flat, c), dim=1)

        hidden = F.leaky_relu(self.inject_label(x_c), negative_slope=self.leaky_relu_slope)
        hidden = F.dropout(hidden, p=self.dropout_rate, training=self.training)

        means = self.linear_means(hidden)
        log_vars = self.linear_log_var(hidden)

        skips = {
            'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4,
            'x5': x5, 'x6': x6, 'x7': x7
        }

        return means, log_vars, skips


class Decoder(nn.Module):
    """
    Decoder with two anti-collapse mechanisms:
      - Skip-connection dropout (replaces Gaussian noise)
      - 1×1 conv bottleneck on every skip (halves channels by default)
    """
    def __init__(self, latent_size, num_labels, leaky_relu_slope=0.2, dropout_rate=0.2,
                 skip_dropout=0.3, skip_bottleneck_factor=2):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope
        self.dropout_rate = dropout_rate
        self.skip_dropout = skip_dropout

        f = skip_bottleneck_factor
        # --- 1x1 bottleneck convolutions for each skip connection ---
        # Reduces information flowing through skips, forcing decoder to
        # rely more on the latent code z.
        self.skip_compress_x7 = nn.Sequential(
            nn.Conv2d(256, 256 // f, 1),
            nn.LeakyReLU(leaky_relu_slope, inplace=True))
        self.skip_compress_x6 = nn.Sequential(
            nn.Conv2d(192, 192 // f, 1),
            nn.LeakyReLU(leaky_relu_slope, inplace=True))
        self.skip_compress_x5 = nn.Sequential(
            nn.Conv2d(128, 128 // f, 1),
            nn.LeakyReLU(leaky_relu_slope, inplace=True))
        self.skip_compress_x4 = nn.Sequential(
            nn.Conv2d(96, 96 // f, 1),
            nn.LeakyReLU(leaky_relu_slope, inplace=True))
        self.skip_compress_x3 = nn.Sequential(
            nn.Conv2d(64, 64 // f, 1),
            nn.LeakyReLU(leaky_relu_slope, inplace=True))
        self.skip_compress_x2 = nn.Sequential(
            nn.Conv2d(32, 32 // f, 1),
            nn.LeakyReLU(leaky_relu_slope, inplace=True))
        self.skip_compress_x1 = nn.Sequential(
            nn.Conv2d(16, 16 // f, 1),
            nn.LeakyReLU(leaky_relu_slope, inplace=True))

        # --- Decoder layers (channel sizes adjusted for bottleneck) ---
        self.LD1 = nn.Linear(latent_size + num_labels, 3840)

        self.dec_conv7 = nn.Conv2d(256 + 256 // f, 256, 3, stride=1, padding=1)
        self.dec_bn7 = nn.BatchNorm2d(256)

        self.dec_conv6 = nn.Conv2d(256 + 192 // f, 192, 3, stride=1, padding=1)
        self.dec_bn6 = nn.BatchNorm2d(192)

        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec_conv5 = nn.Conv2d(192 + 128 // f, 128, 3, stride=1, padding=1)
        self.dec_bn5 = nn.BatchNorm2d(128)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec_conv4 = nn.Conv2d(128 + 96 // f, 96, 3, stride=1, padding=1)
        self.dec_bn4 = nn.BatchNorm2d(96)

        self.dec_conv3 = nn.Conv2d(96 + 64 // f, 64, 3, stride=1, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(64)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec_conv2 = nn.Conv2d(64 + 32 // f, 32, 3, stride=1, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(32)

        self.dec_conv1 = nn.Conv2d(32 + 16 // f, 16, 3, stride=1, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(16)

        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec_conv0 = nn.Conv2d(16, 3, 3, stride=1, padding=1)
        self.dec_bn0 = nn.BatchNorm2d(3)

    def _degrade_skip(self, s):
        """Apply dropout to a skip-connection tensor during training.

        Unlike Gaussian noise, dropout completely zeroes random features,
        forcing the decoder to fall back on z to fill the information gaps.
        """
        if self.training and self.skip_dropout > 0.0:
            return F.dropout(s, p=self.skip_dropout, training=True)
        return s

    def forward(self, z, c, skips):
        z_c = torch.cat((z, c), dim=-1)
        x = F.leaky_relu(self.LD1(z_c), negative_slope=self.leaky_relu_slope)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = x.view(-1, 256, 5, 3)

        s7 = self.skip_compress_x7(self._degrade_skip(skips['x7']))
        x = torch.cat((x, s7), dim=1)
        x = F.leaky_relu(self.dec_bn7(self.dec_conv7(x)), negative_slope=self.leaky_relu_slope)

        s6 = self.skip_compress_x6(self._degrade_skip(skips['x6']))
        x = torch.cat((x, s6), dim=1)
        x = F.leaky_relu(self.dec_bn6(self.dec_conv6(x)), negative_slope=self.leaky_relu_slope)

        x = self.up5(x)
        s5 = self.skip_compress_x5(self._degrade_skip(skips['x5']))
        x = torch.cat((x, s5), dim=1)
        x = F.leaky_relu(self.dec_bn5(self.dec_conv5(x)), negative_slope=self.leaky_relu_slope)

        x = self.up4(x)
        s4 = self.skip_compress_x4(self._degrade_skip(skips['x4']))
        x = torch.cat((x, s4), dim=1)
        x = F.leaky_relu(self.dec_bn4(self.dec_conv4(x)), negative_slope=self.leaky_relu_slope)

        s3 = self.skip_compress_x3(self._degrade_skip(skips['x3']))
        x = torch.cat((x, s3), dim=1)
        x = F.leaky_relu(self.dec_bn3(self.dec_conv3(x)), negative_slope=self.leaky_relu_slope)

        x = self.up2(x)
        s2 = self.skip_compress_x2(self._degrade_skip(skips['x2']))
        x = torch.cat((x, s2), dim=1)
        x = F.leaky_relu(self.dec_bn2(self.dec_conv2(x)), negative_slope=self.leaky_relu_slope)

        s1 = self.skip_compress_x1(self._degrade_skip(skips['x1']))
        x = torch.cat((x, s1), dim=1)
        x = F.leaky_relu(self.dec_bn1(self.dec_conv1(x)), negative_slope=self.leaky_relu_slope)

        x = self.up0(x)
        x = torch.sigmoid(self.dec_bn0(self.dec_conv0(x)))

        return x


# ---------------------------------------------------------------------------
# Vorticity helper
# ---------------------------------------------------------------------------

def compute_vorticity_torch(img, tensor_X, tensor_Y):
    """
    Calculate vorticity in PyTorch using central differences.
    img: (N, 3, NX, NY) where channel 1=v, 0=u.
    """
    dv_dx = torch.gradient(img[:, 1, :, :], spacing=(tensor_X,), dim=1)[0]
    du_dy = torch.gradient(img[:, 0, :, :], spacing=(tensor_Y,), dim=2)[0]
    return dv_dx - du_dy


# ---------------------------------------------------------------------------
# Loss function — Target KLD
# ---------------------------------------------------------------------------

def loss_fn(recon_x, x, mean, log_var, mask, batch_clean, tensor_X, tensor_Y,
            beta=0.001, gamma_base=0.0, target_kld=5.0):
    """
    Loss = BCE + beta * (KLD_raw - target_kld)^2 + eff_gamma * VortMSE

    The target-KLD term provides a two-sided gradient:
      - When KLD < target: gradient pushes KLD upward (fights collapse)
      - When KLD > target: gradient pushes KLD downward (regularises)
    
    KLD is computed as sum over latent dims, mean over batch.
    """
    BCE = torch.nn.functional.mse_loss(recon_x * mask, x * mask, reduction='mean')

    # KLD: sum over latent dims, mean over batch
    kld_per_dim = -0.5 * (1.0 + log_var - mean.pow(2) - log_var.exp())   # [B, L]
    KLD_raw = kld_per_dim.sum(dim=1).mean()                               # scalar

    # Target KLD loss (two-sided)
    kld_target_loss = (KLD_raw - target_kld) ** 2

    coverage = mask[:, 0].mean()
    effective_gamma = gamma_base * (1.0 - coverage)

    v_recon = compute_vorticity_torch(recon_x, tensor_X, tensor_Y)
    v_clean = compute_vorticity_torch(batch_clean, tensor_X, tensor_Y)
    v_mask = mask[:, 0] * mask[:, 1]
    VortMSE = torch.mean(((v_recon - v_clean) * v_mask)**2)

    TotalLoss = BCE + beta * kld_target_loss + effective_gamma * VortMSE

    # Per-dim KLD for diagnostics (detached — no gradient cost)
    kld_per_dim_avg = kld_per_dim.mean(dim=0).detach()      # [L]
    active_units = (kld_per_dim_avg > 0.1).sum().item()

    return TotalLoss, BCE, KLD_raw, kld_target_loss, VortMSE, effective_gamma, kld_per_dim_avg, active_units


# ---------------------------------------------------------------------------
# 3-Phase scheduling helpers
# ---------------------------------------------------------------------------

def _sigmoid(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + math.exp(-x))


def compute_phased_coefficients(epoch, total_epochs, max_beta, max_gamma,
                                beta_warm=0.0, sigmoid_k=10.0,
                                phase1_frac=0.20, phase2_frac=0.60,
                                n_cycles=4):
    """
    Return (current_beta, current_gamma_base) for the given epoch according
    to the 3-phase schedule.

    Phase 1  [0,         phase1_frac*N):  beta = beta_warm, gamma = 0
    Phase 2  [phase1_frac*N, phase2_frac*N): beta cyclical annealing → max_beta, gamma = 0
    Phase 3  [phase2_frac*N, N):           beta = max_beta (frozen), gamma linear → max_gamma
    """
    phase1_end = int(phase1_frac * total_epochs)
    phase2_end = int(phase2_frac * total_epochs)

    if epoch < phase1_end:
        # ---- Phase 1: reconstruction only (beta_warm = 0 by default) ----
        current_beta = beta_warm
        current_gamma_base = 0.0
        phase_label = "P1-Recon"

    elif epoch < phase2_end:
        # ---- Phase 2: cyclical beta annealing, gamma still 0 ----
        t = (epoch - phase1_end) / max(1, phase2_end - phase1_end)
        cycle_idx = int(t * n_cycles)
        cycle_t = t * n_cycles - cycle_idx
        if cycle_t < 0.5:
            current_beta = beta_warm + max_beta * (cycle_t / 0.5)
        else:
            current_beta = max_beta
        current_gamma_base = 0.0
        phase_label = "P2-KLD"

    else:
        # ---- Phase 3: beta frozen, linear gamma ramp ----
        phase3_len = total_epochs - phase2_end
        t = (epoch - phase2_end) / max(1, phase3_len - 1)
        current_beta = max_beta
        current_gamma_base = max_gamma * min(t, 1.0)
        phase_label = "P3-Vort"

    return current_beta, current_gamma_base, phase_label


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_preprocess_data(file_path, n_samples, train_holes, n_valid):
    ImgFile = os.path.abspath(file_path)
    Data = np.load(ImgFile)
    Images = Data['ImgScale'][0:n_samples].copy()
    Mask = Data['Mask'][0:n_samples, 0:3].copy()

    Coverage = 1.0 - np.sum(Mask[:n_samples, 0], axis=(1, 2)) / (Mask.shape[2] * Mask.shape[3])
    ValidRows = np.argsort(-Coverage)[:n_valid]

    if train_holes == 1:
        CleanData = np.load(os.path.abspath(file_path.replace("Holes", "Clean")))
        CleanImg = CleanData['ImgScale'][0:n_samples]
        X = CleanData['X']
        Y = CleanData['Y']
        Max = CleanData['Max'][0:n_samples]
        Min = CleanData['Min'][0:n_samples]
    else:
        CleanImg = Images
        X = np.zeros_like(Images[:, 0, :, :])
        Y = np.zeros_like(Images[:, 0, :, :])
        Max = np.ones((n_samples, 3, 1, 1))
        Min = np.zeros((n_samples, 3, 1, 1))

    dtImg = 1.0 / float(n_samples)
    rawLabels = np.zeros((n_samples, 10))
    rawLabels[:, 0] = np.arange(1, n_samples + 1) * dtImg
    rawLabels[:, 1] = Min[:n_samples, 0, 0, 0] if 'Min' in locals() else 0
    rawLabels[:, 2] = Max[:n_samples, 0, 0, 0] if 'Max' in locals() else 0
    rawLabels[:, 3] = Min[:n_samples, 1, 0, 0] if 'Min' in locals() else 0
    rawLabels[:, 4] = Max[:n_samples, 1, 0, 0] if 'Max' in locals() else 0
    rawLabels[:, 5] = Min[:n_samples, 2, 0, 0] if 'Min' in locals() else 0
    rawLabels[:, 6] = Max[:n_samples, 2, 0, 0] if 'Max' in locals() else 0
    rawLabels[:, 7] = np.sum(CleanImg[:n_samples, 0], axis=(1, 2))
    rawLabels[:, 8] = np.sum(CleanImg[:n_samples, 1], axis=(1, 2))
    rawLabels[:, 9] = np.sum(CleanImg[:n_samples, 2], axis=(1, 2))

    transform = RobustScaler()
    clabels = transform.fit_transform(rawLabels)

    ValidationSet = Images[ValidRows]
    ValidationLabels = clabels[ValidRows]
    ValidationCoverage = Coverage[ValidRows]
    ValidMask = Mask[ValidRows]
    CleanValidation = CleanImg[ValidRows]

    return {
        'Images': Images, 'clabels': clabels, 'Mask': Mask, 'CleanImg': CleanImg,
        'ValidationSet': ValidationSet, 'ValidationLabels': ValidationLabels,
        'ValidationCoverage': ValidationCoverage, 'ValidMask': ValidMask,
        'CleanValidation': CleanValidation, 'transform': transform,
        'X': X, 'Y': Y, 'Max': Max, 'Min': Min, 'ValidRows': ValidRows
    }


# ---------------------------------------------------------------------------
# Error metric
# ---------------------------------------------------------------------------

def CalculateErrorImages(fname, ImageInput, ImagePredict, nImages, CleanImage=None, Mask=None, Coverage=None):
    if CleanImage is not None:
        magPred = np.sqrt(np.sum(np.square(ImagePredict[:nImages, :3]), axis=1))
        magClean = np.sqrt(np.sum(np.square(CleanImage[:nImages, :3]), axis=1))

        NormMag = np.abs(magClean - magPred) / np.maximum(magClean, 1e-5)
        MaxMagClean = np.max(magClean, axis=(1, 2))
        SumNormMag = np.sum(NormMag, axis=(1, 2))

        MaxError = np.max(SumNormMag / np.maximum(MaxMagClean, 1e-5))
        return MaxError
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------

CHECKPOINT_ARCH_VERSION = '1.0-skip-dropout-bottleneck'


def save_checkpoint(path, vae, config, epoch, total_epochs, metrics):
    """Save a self-describing checkpoint.

    Bundles state_dict + the exact kwargs needed to reconstruct CVAE +
    the training config + final-epoch metrics + library versions. Reloads
    can call torch.load(path) and have everything needed in one object,
    eliminating the "what hyperparameters was this trained with?" guesswork
    that previously required inspecting state_dict shapes.
    """
    model_kwargs = {
        'latent_size': config['latent_size'],
        'num_labels': 10,
        'ImgSizeX': 80,
        'ImgSizeY': 48,
        'leaky_relu_slope': config['leaky_relu_slope'],
        'dropout_rate': config['dropout_rate'],
        'skip_dropout': config.get('skip_dropout', 0.3),
        'skip_bottleneck_factor': config.get('skip_bottleneck_factor', 2),
    }
    bundle = {
        'arch_version': CHECKPOINT_ARCH_VERSION,
        'state_dict': vae.state_dict(),
        'model_kwargs': model_kwargs,
        'config': dict(config),
        'epoch_completed': int(epoch),
        'total_epochs': int(total_epochs),
        'metrics': metrics,
        'torch_version': torch.__version__,
    }
    torch.save(bundle, path)


def load_checkpoint(path, map_location='cpu'):
    """Load a checkpoint, accepting either the new bundle format or a
    legacy bare state_dict. Returns a dict with keys:
        'state_dict', 'model_kwargs' (or None), 'config' (or None),
        'arch_version' ('legacy' if absent), 'metrics' (or None).
    """
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(obj, dict) and 'state_dict' in obj:
        return {
            'state_dict': obj['state_dict'],
            'model_kwargs': obj.get('model_kwargs'),
            'config': obj.get('config'),
            'arch_version': obj.get('arch_version', 'unknown'),
            'metrics': obj.get('metrics'),
            'epoch_completed': obj.get('epoch_completed'),
        }
    # Legacy: bare state_dict.
    return {
        'state_dict': obj,
        'model_kwargs': None,
        'config': None,
        'arch_version': 'legacy',
        'metrics': None,
        'epoch_completed': None,
    }


# ---------------------------------------------------------------------------
# Standalone post-processing
# ---------------------------------------------------------------------------

POSTPROCESS_COMPONENTS = ('Ux', 'Uy', 'Uz', 'Speed', 'Vorticity')


def _relative_norm_error(clean, prediction):
    denominator = LA.norm(clean.ravel())
    if denominator == 0.0:
        return 0.0 if LA.norm(prediction.ravel()) == 0.0 else np.nan
    return LA.norm((clean - prediction).ravel()) / denominator


def _field_correlation(clean, prediction):
    clean_flat = clean.ravel() - np.mean(clean)
    prediction_flat = prediction.ravel() - np.mean(prediction)
    denominator = LA.norm(clean_flat) * LA.norm(prediction_flat)
    if denominator == 0.0:
        return np.nan
    return np.dot(clean_flat, prediction_flat) / denominator


def _wasserstein_equal_samples(clean, prediction):
    return np.mean(np.abs(np.sort(clean.ravel()) - np.sort(prediction.ravel())))


def _ks_distance(clean, prediction):
    clean_sorted = np.sort(clean.ravel())
    prediction_sorted = np.sort(prediction.ravel())
    combined = np.concatenate((clean_sorted, prediction_sorted))
    clean_cdf = np.searchsorted(clean_sorted, combined, side='right') / clean_sorted.size
    prediction_cdf = np.searchsorted(prediction_sorted, combined, side='right') / prediction_sorted.size
    return np.max(np.abs(clean_cdf - prediction_cdf))


def _spatial_descriptors(values):
    flattened = values.ravel()
    mean = np.mean(flattened)
    std = np.std(flattened)
    if std == 0.0:
        return mean, std, np.nan, np.nan
    standardized = (flattened - mean) / std
    return mean, std, np.mean(standardized**3), np.mean(standardized**4)


def _compute_vorticity_numpy(velocity, X, Y):
    X_m = np.asarray(X, dtype=np.float64) * 1e-3
    Y_m = np.asarray(Y, dtype=np.float64) * 1e-3
    edge_order = 2 if velocity.shape[2] >= 3 and velocity.shape[3] >= 3 else 1
    dv_dx = np.gradient(velocity[:, 1], X_m, axis=1, edge_order=edge_order)
    du_dy = np.gradient(velocity[:, 0], Y_m, axis=2, edge_order=edge_order)
    return dv_dx - du_dy


def _structure_function(field, axis, max_lag, orders):
    values = np.empty((len(orders), max_lag), dtype=np.float64)
    signed_third = np.full(max_lag, np.nan, dtype=np.float64)
    pair_count = np.empty(max_lag, dtype=np.int64)
    for lag in range(1, max_lag + 1):
        if axis == 0:
            increments = field[lag:, :] - field[:-lag, :]
        else:
            increments = field[:, lag:] - field[:, :-lag]
        pair_count[lag - 1] = increments.size
        for order_index, order in enumerate(orders):
            values[order_index, lag - 1] = np.mean(np.abs(increments)**order)
        if 3 in orders:
            signed_third[lag - 1] = np.mean(increments**3)
    return values, signed_third, pair_count


def _select_postprocess_snapshots(requested_indices, mask, automatic_count):
    """Select requested snapshots or both extremes of valid-vector coverage."""
    if automatic_count < 1:
        raise ValueError('--automatic-snapshot-count must be at least 1')
    validity = np.asarray(mask[:, 0], dtype=np.float64)
    coverage = np.mean(np.clip(validity, 0.0, 1.0), axis=(1, 2))
    eliminated_count = np.count_nonzero(validity < 1.0, axis=(1, 2))
    n_snapshots = validity.shape[0]

    if requested_indices is not None:
        selected = []
        for value in requested_indices:
            index = int(value)
            if index not in selected:
                selected.append(index)
        snapshot_indices = np.asarray(selected, dtype=np.int64)
        if snapshot_indices.size == 0:
            raise ValueError('At least one --snapshot-index is required')
        if np.any(snapshot_indices < 0) or np.any(snapshot_indices >= n_snapshots):
            raise ValueError(f'Snapshot indices must be between 0 and {n_snapshots - 1}')
        return snapshot_indices, ['user'] * snapshot_indices.size, coverage, eliminated_count

    count = min(automatic_count, n_snapshots)
    original_indices = np.arange(n_snapshots)
    most_gapped = np.lexsort((original_indices, coverage))[:count]
    least_gapped = np.lexsort((original_indices, -coverage))[:count]
    selected = []
    labels = []
    for label, indices in (('most_gapped', most_gapped), ('least_gapped', least_gapped)):
        for index in indices:
            index = int(index)
            if index in selected:
                labels[selected.index(index)] += f'/{label}'
            else:
                selected.append(index)
                labels.append(label)
    return np.asarray(selected, dtype=np.int64), labels, coverage, eliminated_count


def _plot_vector_relative_snapshot(clean_velocity, prediction_velocity, mask,
                                   X, Y, snapshot_index, output_file,
                                   eliminated_count=None, quiver_scale=325.0):
    """Plot clean, gappy, and reconstructed vectors with relative error."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    clean_velocity = np.asarray(clean_velocity)
    prediction_velocity = np.asarray(prediction_velocity)
    validity = np.asarray(mask)
    if validity.ndim == 3:
        validity = validity[0]
    if clean_velocity.ndim != 3 or prediction_velocity.shape != clean_velocity.shape:
        raise ValueError('Velocity snapshots must have matching [component, nx, ny] shapes')
    if clean_velocity.shape[0] < 2 or validity.shape != clean_velocity.shape[1:]:
        raise ValueError('Mask and velocity grid shapes must match')

    X = np.asarray(X)
    Y = np.asarray(Y)
    if clean_velocity.shape[1:] != (X.size, Y.size):
        raise ValueError('Coordinate lengths must match the velocity grid shape')

    x_grid, y_grid = np.meshgrid(X, Y, indexing='ij')
    missing = validity < 1.0
    gappy_velocity = np.ma.array(
        clean_velocity[:2], mask=np.broadcast_to(missing, clean_velocity[:2].shape)
    )
    difference_magnitude = np.sqrt(np.sum(
        (clean_velocity - prediction_velocity) ** 2, axis=0
    ))
    clean_magnitude = np.sqrt(np.sum(clean_velocity ** 2, axis=0))
    relative_error = difference_magnitude / np.maximum(
        clean_magnitude, np.finfo(np.float64).eps
    )
    finite_errors = relative_error[np.isfinite(relative_error)]
    if finite_errors.size == 0:
        raise ValueError('Relative-error field contains no finite values')
    color_limit = float(np.percentile(finite_errors, 98.0))
    if color_limit <= 0.0:
        color_limit = max(float(np.max(finite_errors)), 1.0)
    color_levels = np.linspace(0.0, color_limit, 21)
    error_colormap = LinearSegmentedColormap.from_list(
        'relative_error_light',
        ('#ffffff', '#e1f0f7', '#b8e0d2', '#fee8a8', '#f3a683'),
    )

    title_font_size = 24
    label_font_size = 22
    tick_font_size = 22
    suptitle_font_size = 26

    figure, axes = plt.subplots(1, 3, figsize=(24, 7.5), constrained_layout=True)
    quiver_style = {
        'scale': quiver_scale,
        'angles': 'xy',
        'pivot': 'mid',
        'width': 0.0040,
        'color': '#111111',
        'zorder': 4,
    }
    reconstructed_quiver_halo_style = {
        **quiver_style,
        'width': 0.0110,
        'color': 'white',
        'zorder': 4,
    }
    reconstructed_quiver_style = {
        **quiver_style,
        'width': 0.0050,
        'color': '#333333',
        'zorder': 5,
    }
    axes[0].quiver(
        x_grid, y_grid, clean_velocity[0], clean_velocity[1], **quiver_style
    )
    axes[0].set_title('Original vector field', fontsize=title_font_size)
    axes[1].quiver(
        x_grid, y_grid, gappy_velocity[0], gappy_velocity[1], **quiver_style
    )
    axes[1].set_title('Gappy vector field', fontsize=title_font_size)

    contour = axes[2].contourf(
        x_grid, y_grid, relative_error, levels=color_levels,
        cmap=error_colormap, norm=Normalize(0.0, color_limit, clip=True),
        extend='max', zorder=1,
    )
    axes[2].quiver(
        x_grid, y_grid, prediction_velocity[0], prediction_velocity[1],
        **reconstructed_quiver_halo_style
    )
    axes[2].quiver(
        x_grid, y_grid, prediction_velocity[0], prediction_velocity[1],
        **reconstructed_quiver_style
    )
    axes[2].set_title('Reconstructed field with relative error', fontsize=title_font_size)
    colorbar = figure.colorbar(contour, ax=axes[2], fraction=0.046, pad=0.04)
    colorbar.set_label('Relative error', fontsize=label_font_size)
    colorbar.ax.tick_params(labelsize=tick_font_size)

    for axis in axes:
        axis.set_xlabel('Transverse dimension [mm]', fontsize=label_font_size)
        axis.tick_params(axis='both', labelsize=tick_font_size)
        axis.set_aspect('equal', adjustable='box')
        axis.grid(alpha=0.15)
    axes[0].set_ylabel('Axial dimension [mm]', fontsize=label_font_size)
    x_limits = axes[0].get_xlim()
    y_limits = axes[0].get_ylim()
    for axis in axes[1:]:
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)

    detail = ''
    if eliminated_count is not None:
        detail = f': {int(eliminated_count)} vectors eliminated'
    figure.suptitle(f'Vector-field reconstruction{detail}', fontsize=suptitle_font_size)
    figure.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close(figure)
    return relative_error


def _plot_gap_region_error_histograms(clean_velocity, prediction_velocity,
                                      clean_vorticity, prediction_vorticity,
                                      mask, output_directory, bins=50):
    """Plot snapshot-level relative-error distributions over eliminated vectors."""
    import matplotlib.pyplot as plt

    validity = np.asarray(mask)
    if validity.ndim != 4 or validity.shape[0] != clean_velocity.shape[0]:
        raise ValueError('Mask must have shape [snapshots, channels, nx, ny]')
    if clean_velocity.shape != prediction_velocity.shape:
        raise ValueError('Clean and predicted velocity arrays must have matching shapes')
    if clean_vorticity.shape != prediction_vorticity.shape \
            or clean_vorticity.shape[0] != clean_velocity.shape[0]:
        raise ValueError('Clean and predicted vorticity arrays must match velocity snapshots')

    snapshot_count = clean_velocity.shape[0]
    velocity_errors = np.full((snapshot_count, 3), np.nan, dtype=np.float64)
    vorticity_errors = np.full(snapshot_count, np.nan, dtype=np.float64)
    for snapshot_index in range(snapshot_count):
        missing = validity[snapshot_index, 0] < 1.0
        if not np.any(missing):
            continue
        for component_index in range(3):
            velocity_errors[snapshot_index, component_index] = _relative_norm_error(
                clean_velocity[snapshot_index, component_index][missing],
                prediction_velocity[snapshot_index, component_index][missing],
            )
        vorticity_errors[snapshot_index] = _relative_norm_error(
            clean_vorticity[snapshot_index][missing],
            prediction_vorticity[snapshot_index][missing],
        )

    valid_velocity = [
        values[np.isfinite(values)] for values in velocity_errors.T
    ]
    valid_vorticity = vorticity_errors[np.isfinite(vorticity_errors)]
    if any(values.size == 0 for values in valid_velocity) or valid_vorticity.size == 0:
        raise ValueError('No finite gap-region relative errors are available to plot')

    os.makedirs(output_directory, exist_ok=True)
    velocity_output = os.path.join(output_directory, 'error_hist_velocity.pdf')
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axis, errors, label, color in zip(
            axes, valid_velocity, ('$U_x$ Error', '$U_y$ Error', '$U_z$ Error'),
            ('steelblue', 'darkorange', 'seagreen')):
        counts, edges = np.histogram(errors, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        axis.bar(
            centers, counts, width=np.diff(edges), color=color, alpha=0.75,
            edgecolor='white', linewidth=0.4,
        )
        axis.plot(centers, counts, color=color, linewidth=1.2)
        mean_error = float(np.mean(errors))
        axis.axvline(
            mean_error, color='black', linestyle='--', linewidth=1.0,
            label=f'Mean = {mean_error:.3f}',
        )
        axis.set_title(label, fontsize=18)
        axis.set_xlabel('Relative Error', fontsize=18)
        axis.set_ylabel('Probability Density', fontsize=18)
        axis.grid(axis='y', linestyle='--', alpha=0.4)
        axis.legend(fontsize=14)
    figure.suptitle(
        'Gap-Region Velocity Error Distributions', fontsize=18, fontweight='bold'
    )
    figure.savefig(velocity_output, format='pdf', bbox_inches='tight')
    plt.close(figure)

    vorticity_output = os.path.join(output_directory, 'error_hist_vorticity.pdf')
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    counts, edges = np.histogram(valid_vorticity, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    axis.bar(
        centers, counts, width=np.diff(edges), color='mediumorchid', alpha=0.75,
        edgecolor='white', linewidth=0.4,
    )
    axis.plot(centers, counts, color='mediumorchid', linewidth=1.2)
    mean_error = float(np.mean(valid_vorticity))
    axis.axvline(
        mean_error, color='black', linestyle='--', linewidth=1.0,
        label=f'Mean = {mean_error:.3f}',
    )
    axis.set_title('Vorticity Error', fontsize=18)
    axis.set_xlabel('Relative Error', fontsize=18)
    axis.set_ylabel('Probability Density', fontsize=18)
    axis.grid(axis='y', linestyle='--', alpha=0.4)
    axis.legend(fontsize=14)
    figure.suptitle(
        'Gap-Region Vorticity Error Distribution', fontsize=13, fontweight='bold'
    )
    figure.savefig(vorticity_output, format='pdf', bbox_inches='tight')
    plt.close(figure)
    return velocity_output, vorticity_output, velocity_errors, vorticity_errors


def _json_compatible(value):
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return {'shape': list(value.shape), 'dtype': str(value.dtype)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_postprocess_inputs(clean_path, prediction_path, holes_path, model_path):
    with np.load(os.path.abspath(prediction_path)) as prediction_data:
        if 'PredictImg' not in prediction_data.files:
            raise ValueError("Prediction file must contain a 'PredictImg' array")
        prediction_velocity = np.asarray(prediction_data['PredictImg'], dtype=np.float64)
        prediction_X = np.asarray(prediction_data['X'], dtype=np.float64) \
            if 'X' in prediction_data.files else None
        prediction_Y = np.asarray(prediction_data['Y'], dtype=np.float64) \
            if 'Y' in prediction_data.files else None

    with np.load(os.path.abspath(holes_path)) as holes_data:
        if 'Mask' not in holes_data.files:
            raise ValueError("Holes file must contain a 'Mask' array")
        mask = np.asarray(holes_data['Mask'], dtype=np.float64)
        holes_X = np.asarray(holes_data['X'], dtype=np.float64) \
            if 'X' in holes_data.files else None
        holes_Y = np.asarray(holes_data['Y'], dtype=np.float64) \
            if 'Y' in holes_data.files else None

    with np.load(os.path.abspath(clean_path)) as clean_data:
        required_clean = {'ImgScale', 'X', 'Y', 'Min', 'Max'}
        missing = required_clean.difference(clean_data.files)
        if missing:
            raise ValueError(f"Clean file is missing arrays: {sorted(missing)}")
        clean_shape = clean_data['ImgScale'].shape
        if len(clean_shape) != 4 or prediction_velocity.ndim != 4 \
                or clean_shape[1:] != prediction_velocity.shape[1:]:
            raise ValueError(
                'Clean and prediction component/spatial shapes must match; got '
                f'{clean_shape} vs {prediction_velocity.shape}'
            )

        clean_snapshot_count = clean_shape[0]
        prediction_snapshot_count = prediction_velocity.shape[0]
        if prediction_snapshot_count > clean_snapshot_count:
            raise ValueError(
                f'Prediction has {prediction_snapshot_count} snapshots but the clean file has only '
                f'{clean_snapshot_count}; prefix alignment is not possible'
            )

        clean_scaled = np.asarray(
            clean_data['ImgScale'][:prediction_snapshot_count], dtype=np.float64
        )
        clean_min = np.asarray(clean_data['Min'])
        clean_max = np.asarray(clean_data['Max'])
        if clean_min.ndim > 0 and clean_min.shape[0] == clean_snapshot_count:
            clean_min = clean_min[:prediction_snapshot_count]
        if clean_max.ndim > 0 and clean_max.shape[0] == clean_snapshot_count:
            clean_max = clean_max[:prediction_snapshot_count]
        clean_velocity = clean_scaled * (clean_max - clean_min) + clean_min
        clean_X = np.asarray(clean_data['X'], dtype=np.float64)
        clean_Y = np.asarray(clean_data['Y'], dtype=np.float64)

    if clean_velocity.ndim != 4 or clean_velocity.shape[1] < 3:
        raise ValueError(
            'Expected clean and prediction arrays shaped [snapshots, 3, nx, ny], '
            f'got {clean_velocity.shape}'
        )
    if mask.ndim != 4 or mask.shape[0] < prediction_snapshot_count \
            or mask.shape[2:] != clean_velocity.shape[2:]:
        raise ValueError(
            'Expected holes Mask shaped [snapshots, channels, nx, ny] with at least '
            f'{prediction_snapshot_count} snapshots; got {mask.shape}'
        )
    mask = mask[:prediction_snapshot_count]

    X = clean_X
    Y = clean_Y
    coordinate_source = 'clean'
    coordinate_mismatches = []
    if holes_X is not None and holes_Y is not None:
        holes_match_grid = holes_X.ndim == 1 and holes_Y.ndim == 1 \
            and clean_velocity.shape[2:] == (holes_X.size, holes_Y.size)
        if holes_match_grid:
            X = holes_X
            Y = holes_Y
            coordinate_source = 'holes'
        else:
            coordinate_mismatches.append('holes_shape')
    elif holes_X is not None or holes_Y is not None:
        coordinate_mismatches.append('holes_incomplete')

    if clean_X.shape != X.shape or not np.allclose(clean_X, X):
        coordinate_mismatches.append('clean_X')
    if clean_Y.shape != Y.shape or not np.allclose(clean_Y, Y):
        coordinate_mismatches.append('clean_Y')
    if prediction_X is not None and (prediction_X.shape != X.shape
                                     or not np.allclose(prediction_X, X)):
        if coordinate_source == 'clean':
            raise ValueError('X coordinates differ between clean and prediction files')
        coordinate_mismatches.append('prediction_X')
    if prediction_Y is not None and (prediction_Y.shape != Y.shape
                                     or not np.allclose(prediction_Y, Y)):
        if coordinate_source == 'clean':
            raise ValueError('Y coordinates differ between clean and prediction files')
        coordinate_mismatches.append('prediction_Y')

    if X.ndim != 1 or Y.ndim != 1 or clean_velocity.shape[2:] != (X.size, Y.size):
        raise ValueError(
            f'Coordinate lengths {(X.size, Y.size)} do not match field shape '
            f'{clean_velocity.shape[2:]}'
        )
    if X.size < 2 or Y.size < 2 or not np.allclose(np.diff(X), np.diff(X)[0]) \
            or not np.allclose(np.diff(Y), np.diff(Y)[0]):
        raise ValueError('Post-processing requires uniformly spaced 1D X and Y coordinates')

    checkpoint = load_checkpoint(os.path.abspath(model_path), map_location='cpu')
    model_metadata = {
        'sha256': _sha256_file(os.path.abspath(model_path)),
        'arch_version': checkpoint['arch_version'],
        'epoch_completed': checkpoint['epoch_completed'],
        'model_kwargs': checkpoint['model_kwargs'],
        'config': checkpoint['config'],
        'metrics': checkpoint['metrics'],
    }
    alignment_metadata = {
        'clean_snapshot_count': clean_snapshot_count,
        'prediction_snapshot_count': prediction_snapshot_count,
        'used_snapshot_count': prediction_snapshot_count,
        'clean_prefix_truncated': clean_snapshot_count > prediction_snapshot_count,
        'holes_snapshot_count': int(mask.shape[0]),
        'assumption': 'Prediction snapshot n corresponds to clean snapshot n.',
        'coordinate_source': coordinate_source,
        'coordinate_mismatches': coordinate_mismatches,
    }
    return (
        clean_velocity[:, :3], prediction_velocity[:, :3], mask, X, Y,
        model_metadata, alignment_metadata
    )


def run_postprocessing(args):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.ticker import LogFormatterSciNotation, LogLocator, MaxNLocator
    except ImportError as exc:
        raise RuntimeError('Post-processing requires matplotlib') from exc

    clean, prediction, mask, X, Y, model_metadata, alignment_metadata = \
        _load_postprocess_inputs(
            args.clean_file, args.prediction_file, args.holes_file, args.model_file
        )
    snapshot_indices, selection_labels, coverage, eliminated_count = \
        _select_postprocess_snapshots(
            args.snapshot_indices, mask, args.automatic_snapshot_count
        )
    selection_mode = 'user' if args.snapshot_indices is not None else 'coverage_extremes'
    print(
        f'[POSTPROCESS] Snapshot selection mode={selection_mode}: '
        + ', '.join(
            f'{index} ({label}, removed={eliminated_count[index]}, '
            f'coverage={coverage[index]:.4f})'
            for index, label in zip(snapshot_indices, selection_labels)
        )
    )
    if alignment_metadata['clean_prefix_truncated']:
        print(
            f"[POSTPROCESS] Using the first {alignment_metadata['used_snapshot_count']} of "
            f"{alignment_metadata['clean_snapshot_count']} clean snapshots to match the prediction."
        )
    n_snapshots, _, nx, ny = clean.shape
    orders = np.asarray(sorted(set(args.structure_orders)), dtype=np.int64)
    if np.any(orders < 1):
        raise ValueError('Structure-function orders must be positive integers')

    max_lag_x = args.max_lag_x if args.max_lag_x is not None else max(1, nx // 4)
    max_lag_y = args.max_lag_y if args.max_lag_y is not None else max(1, ny // 4)
    if not 1 <= max_lag_x < nx or not 1 <= max_lag_y < ny:
        raise ValueError('Maximum structure-function lags must be positive and smaller than the grid')
    if args.sample_rate <= 0.0:
        raise ValueError('--sample-rate must be positive')
    if args.pdf_bins < 2:
        raise ValueError('--pdf-bins must be at least 2')

    os.makedirs(args.postprocess_output_dir, exist_ok=True)
    clean_vorticity = _compute_vorticity_numpy(clean, X, Y)
    prediction_vorticity = _compute_vorticity_numpy(prediction, X, Y)
    clean_speed = np.sqrt(np.sum(clean**2, axis=1))
    prediction_speed = np.sqrt(np.sum(prediction**2, axis=1))
    clean_fields = [clean[:, 0], clean[:, 1], clean[:, 2], clean_speed, clean_vorticity]
    prediction_fields = [
        prediction[:, 0], prediction[:, 1], prediction[:, 2],
        prediction_speed, prediction_vorticity
    ]
    times = np.arange(n_snapshots, dtype=np.float64) / args.sample_rate

    vector_directory = os.path.join(args.postprocess_output_dir, 'vector_plots')
    histogram_directory = os.path.join(args.postprocess_output_dir, 'error_histograms')
    spatial_pdf_directory = os.path.join(args.postprocess_output_dir, 'instantaneous_spatial_pdfs')
    structure_function_directory = os.path.join(
        args.postprocess_output_dir, 'instantaneous_structure_functions'
    )
    os.makedirs(vector_directory, exist_ok=True)
    os.makedirs(histogram_directory, exist_ok=True)
    os.makedirs(spatial_pdf_directory, exist_ok=True)
    os.makedirs(structure_function_directory, exist_ok=True)
    for filename in os.listdir(histogram_directory):
        if filename.endswith(('_velocity_histogram.pdf', '_vorticity_histogram.pdf')):
            os.remove(os.path.join(histogram_directory, filename))
    vector_outputs = []
    for snapshot_index, selection_label in zip(snapshot_indices, selection_labels):
        output_stem = f'{selection_label}_snapshot_{snapshot_index:05d}'
        vector_output = os.path.join(vector_directory, f'{output_stem}_relative_error.pdf')
        _plot_vector_relative_snapshot(
            clean[snapshot_index], prediction[snapshot_index], mask[snapshot_index],
            X, Y, int(snapshot_index), vector_output,
            eliminated_count=eliminated_count[snapshot_index],
        )
        vector_outputs.append(vector_output)

    velocity_histogram, vorticity_histogram, gap_velocity_errors, gap_vorticity_errors = \
        _plot_gap_region_error_histograms(
            clean, prediction, clean_vorticity, prediction_vorticity,
            mask, histogram_directory, bins=args.pdf_bins,
        )

    metric_names = ('mae', 'rmse', 'max_abs', 'relative_l2', 'correlation', 'wasserstein', 'ks')
    time_metrics = np.empty((n_snapshots, len(POSTPROCESS_COMPONENTS), len(metric_names)))
    for snapshot_index in range(n_snapshots):
        for component_index, (clean_field, prediction_field) in enumerate(
                zip(clean_fields, prediction_fields)):
            clean_snapshot = clean_field[snapshot_index]
            prediction_snapshot = prediction_field[snapshot_index]
            difference = clean_snapshot - prediction_snapshot
            time_metrics[snapshot_index, component_index] = (
                np.mean(np.abs(difference)), np.sqrt(np.mean(difference**2)),
                np.max(np.abs(difference)), _relative_norm_error(clean_snapshot, prediction_snapshot),
                _field_correlation(clean_snapshot, prediction_snapshot),
                _wasserstein_equal_samples(clean_snapshot, prediction_snapshot),
                _ks_distance(clean_snapshot, prediction_snapshot),
            )

    cell_area = abs(float(np.diff(X)[0] * np.diff(Y)[0])) * 1e-6
    component_power_clean = 0.5 * np.sum(clean**2, axis=(2, 3)) * cell_area
    component_power_prediction = 0.5 * np.sum(prediction**2, axis=(2, 3)) * cell_area
    enstrophy_clean = 0.5 * np.sum(clean_vorticity**2, axis=(1, 2)) * cell_area
    enstrophy_prediction = 0.5 * np.sum(prediction_vorticity**2, axis=(1, 2)) * cell_area

    selected_count = snapshot_indices.size
    pdf_edges = np.empty((selected_count, len(POSTPROCESS_COMPONENTS), args.pdf_bins + 1))
    pdf_clean = np.empty((selected_count, len(POSTPROCESS_COMPONENTS), args.pdf_bins))
    pdf_prediction = np.empty_like(pdf_clean)
    descriptors_clean = np.empty((selected_count, len(POSTPROCESS_COMPONENTS), 4))
    descriptors_prediction = np.empty_like(descriptors_clean)
    for selected_index, snapshot_index in enumerate(snapshot_indices):
        for component_index, (clean_field, prediction_field) in enumerate(
                zip(clean_fields, prediction_fields)):
            clean_snapshot = clean_field[snapshot_index]
            prediction_snapshot = prediction_field[snapshot_index]
            lower = min(np.min(clean_snapshot), np.min(prediction_snapshot))
            upper = max(np.max(clean_snapshot), np.max(prediction_snapshot))
            if lower == upper:
                lower -= 0.5
                upper += 0.5
            edges = np.linspace(lower, upper, args.pdf_bins + 1)
            pdf_edges[selected_index, component_index] = edges
            pdf_clean[selected_index, component_index] = np.histogram(
                clean_snapshot, bins=edges, density=True
            )[0]
            pdf_prediction[selected_index, component_index] = np.histogram(
                prediction_snapshot, bins=edges, density=True
            )[0]
            descriptors_clean[selected_index, component_index] = _spatial_descriptors(clean_snapshot)
            descriptors_prediction[selected_index, component_index] = _spatial_descriptors(
                prediction_snapshot
            )

    sf_x_clean = np.empty((selected_count, 3, orders.size, max_lag_x))
    sf_x_prediction = np.empty_like(sf_x_clean)
    sf_y_clean = np.empty((selected_count, 3, orders.size, max_lag_y))
    sf_y_prediction = np.empty_like(sf_y_clean)
    sf_x_signed_third_clean = np.full((selected_count, 3, max_lag_x), np.nan)
    sf_x_signed_third_prediction = np.full_like(sf_x_signed_third_clean, np.nan)
    sf_y_signed_third_clean = np.full((selected_count, 3, max_lag_y), np.nan)
    sf_y_signed_third_prediction = np.full_like(sf_y_signed_third_clean, np.nan)
    for selected_index, snapshot_index in enumerate(snapshot_indices):
        for component_index in range(3):
            sf_x_clean[selected_index, component_index], sf_x_signed_third_clean[
                selected_index, component_index
            ], pair_count_x = _structure_function(
                clean[snapshot_index, component_index], 0, max_lag_x, orders
            )
            sf_x_prediction[selected_index, component_index], sf_x_signed_third_prediction[
                selected_index, component_index
            ], _ = _structure_function(
                prediction[snapshot_index, component_index], 0, max_lag_x, orders
            )
            sf_y_clean[selected_index, component_index], sf_y_signed_third_clean[
                selected_index, component_index
            ], pair_count_y = _structure_function(
                clean[snapshot_index, component_index], 1, max_lag_y, orders
            )
            sf_y_prediction[selected_index, component_index], sf_y_signed_third_prediction[
                selected_index, component_index
            ], _ = _structure_function(
                prediction[snapshot_index, component_index], 1, max_lag_y, orders
            )

    separation_x = np.arange(1, max_lag_x + 1) * abs(float(np.diff(X)[0]))
    separation_y = np.arange(1, max_lag_y + 1) * abs(float(np.diff(Y)[0]))
    output_npz = os.path.join(args.postprocess_output_dir, 'turbulence_statistics.npz')
    np.savez_compressed(
        output_npz, sample_rate_hz=args.sample_rate, time_s=times, X_mm=X, Y_mm=Y,
        snapshot_indices=snapshot_indices, component_names=np.asarray(POSTPROCESS_COMPONENTS),
        snapshot_selection_labels=np.asarray(selection_labels),
        mask_coverage=coverage, eliminated_vector_count=eliminated_count,
        gap_region_velocity_relative_error=gap_velocity_errors,
        gap_region_vorticity_relative_error=gap_vorticity_errors,
        metric_names=np.asarray(metric_names), time_metrics=time_metrics,
        component_power_clean=component_power_clean,
        component_power_prediction=component_power_prediction,
        enstrophy_clean=enstrophy_clean, enstrophy_prediction=enstrophy_prediction,
        pdf_edges=pdf_edges, pdf_clean=pdf_clean, pdf_prediction=pdf_prediction,
        descriptor_names=np.asarray(('spatial_mean', 'spatial_std', 'skewness', 'kurtosis')),
        descriptors_clean=descriptors_clean, descriptors_prediction=descriptors_prediction,
        structure_orders=orders, separation_x_mm=separation_x, separation_y_mm=separation_y,
        pair_count_x=pair_count_x, pair_count_y=pair_count_y,
        sf_x_clean=sf_x_clean, sf_x_prediction=sf_x_prediction,
        sf_y_clean=sf_y_clean, sf_y_prediction=sf_y_prediction,
        sf_x_signed_third_clean=sf_x_signed_third_clean,
        sf_x_signed_third_prediction=sf_x_signed_third_prediction,
        sf_y_signed_third_clean=sf_y_signed_third_clean,
        sf_y_signed_third_prediction=sf_y_signed_third_prediction,
    )

    output_csv = os.path.join(args.postprocess_output_dir, 'turbulence_statistics_summary.csv')
    with open(output_csv, 'w', newline='') as csv_file:
        fieldnames = ['snapshot_index', 'time_s']
        for component in POSTPROCESS_COMPONENTS:
            fieldnames.extend(f'{component}_{metric}' for metric in metric_names)
        for component in ('Ux', 'Uy', 'Uz'):
            fieldnames.extend((f'{component}_power_clean', f'{component}_power_prediction'))
        fieldnames.extend(('enstrophy_clean', 'enstrophy_prediction'))
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for snapshot_index in range(n_snapshots):
            row = {'snapshot_index': snapshot_index, 'time_s': times[snapshot_index]}
            for component_index, component in enumerate(POSTPROCESS_COMPONENTS):
                for metric_index, metric in enumerate(metric_names):
                    row[f'{component}_{metric}'] = time_metrics[
                        snapshot_index, component_index, metric_index
                    ]
            for component_index, component in enumerate(('Ux', 'Uy', 'Uz')):
                row[f'{component}_power_clean'] = component_power_clean[snapshot_index, component_index]
                row[f'{component}_power_prediction'] = component_power_prediction[
                    snapshot_index, component_index
                ]
            row['enstrophy_clean'] = enstrophy_clean[snapshot_index]
            row['enstrophy_prediction'] = enstrophy_prediction[snapshot_index]
            writer.writerow(row)

    output_pdf = os.path.join(args.postprocess_output_dir, 'turbulence_statistics.pdf')
    colors = {'clean': '#1f4e79', 'prediction': '#c43c39'}
    clean_line_style = {'color': colors['clean'], 'linestyle': '-', 'linewidth': 2.8}
    prediction_line_style = {
        'color': colors['prediction'], 'linestyle': '--', 'linewidth': 2.8
    }
    title_font_size = 24
    label_font_size = 22
    tick_font_size = 22
    legend_font_size = 22
    suptitle_font_size = 26

    def _format_publication_axis(axis, *, log_ticks=False):
        axis.tick_params(axis='both', which='major', labelsize=tick_font_size,
                         length=7, width=1.2)
        axis.tick_params(axis='both', which='minor', length=4, width=1.0)
        if log_ticks:
            axis.xaxis.set_major_locator(
                LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12)
            )
            axis.yaxis.set_major_locator(
                LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12)
            )
            axis.xaxis.set_major_formatter(
                LogFormatterSciNotation(
                    base=10.0, labelOnlyBase=False,
                    minor_thresholds=(np.inf, np.inf)
                )
            )
            axis.yaxis.set_major_formatter(
                LogFormatterSciNotation(
                    base=10.0, labelOnlyBase=False,
                    minor_thresholds=(np.inf, np.inf)
                )
            )
        else:
            axis.xaxis.set_major_locator(MaxNLocator(nbins=7))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=7))

    spatial_pdf_outputs = []
    structure_function_outputs = []
    with PdfPages(output_pdf) as pdf:
        figure, axes = plt.subplots(3, 2, figsize=(11, 12), constrained_layout=True)
        for component_index, component in enumerate(POSTPROCESS_COMPONENTS):
            axis = axes.flat[component_index]
            axis.plot(times * 1e3, time_metrics[:, component_index, 3], color='#333333')
            axis.set_title(f'{component}: relative L2 error')
            axis.set_xlabel('Time (ms)')
            axis.set_ylabel('Relative error')
            axis.grid(alpha=0.25)
        axes.flat[-1].plot(times * 1e3, enstrophy_clean, color=colors['clean'], label='Clean')
        axes.flat[-1].plot(
            times * 1e3, enstrophy_prediction, color=colors['prediction'], label='Prediction'
        )
        axes.flat[-1].set_title('Instantaneous planar enstrophy-like integral')
        axes.flat[-1].set_xlabel('Time (ms)')
        axes.flat[-1].set_ylabel(r'$0.5\int \omega_z^2\,dA$')
        axes.flat[-1].legend()
        axes.flat[-1].grid(alpha=0.25)
        figure.suptitle('Time-resolved reconstruction comparison (no temporal averaging)')
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(3, 2, figsize=(11, 12), constrained_layout=True)
        for component_index, component in enumerate(POSTPROCESS_COMPONENTS):
            axis = axes.flat[component_index]
            axis.plot(times * 1e3, time_metrics[:, component_index, 4], color='#246b45')
            axis.set_title(f'{component}: instantaneous spatial correlation')
            axis.set_xlabel('Time (ms)')
            axis.set_ylim(-1.05, 1.05)
            axis.grid(alpha=0.25)
        for component_index, component in enumerate(('Ux', 'Uy', 'Uz')):
            axes.flat[-1].plot(
                times * 1e3,
                component_power_prediction[:, component_index] /
                np.maximum(component_power_clean[:, component_index], np.finfo(float).tiny),
                label=component,
            )
        axes.flat[-1].axhline(1.0, color='#333333', linewidth=0.8)
        axes.flat[-1].set_title('Prediction / clean integrated component power')
        axes.flat[-1].set_xlabel('Time (ms)')
        axes.flat[-1].set_ylabel('Power ratio')
        axes.flat[-1].legend()
        axes.flat[-1].grid(alpha=0.25)
        pdf.savefig(figure)
        plt.close(figure)

        for selected_index, snapshot_index in enumerate(snapshot_indices):
            output_stem = f'{selection_labels[selected_index]}_snapshot_{snapshot_index:05d}'
            figure = plt.figure(figsize=(16, 12), constrained_layout=True)
            grid = figure.add_gridspec(2, 2)
            pdf_axes = (
                figure.add_subplot(grid[0, 0]),
                figure.add_subplot(grid[0, 1]),
                figure.add_subplot(grid[1, :]),
            )
            for component_index, (component, axis) in enumerate(zip(('Ux', 'Uy', 'Uz'), pdf_axes)):
                edges = pdf_edges[selected_index, component_index]
                centers = 0.5 * (edges[:-1] + edges[1:])
                axis.plot(centers, pdf_clean[selected_index, component_index],
                          label='Clean', **clean_line_style)
                axis.plot(centers, pdf_prediction[selected_index, component_index],
                          label='Prediction', **prediction_line_style)
                axis.set_title(f'{component} PDF', fontsize=title_font_size)
                axis.set_xlabel(component, fontsize=label_font_size)
                axis.set_ylabel('Probability density', fontsize=label_font_size)
                axis.grid(alpha=0.25)
                axis.legend(fontsize=legend_font_size)
                _format_publication_axis(axis)
            figure.suptitle(
                f'Instantaneous spatial PDFs: snapshot {snapshot_index}, '
                f't={times[snapshot_index] * 1e3:.3f} ms',
                fontsize=suptitle_font_size
            )
            spatial_pdf_output = os.path.join(
                spatial_pdf_directory, f'{output_stem}_velocity_pdfs.pdf'
            )
            figure.savefig(spatial_pdf_output, format='pdf', bbox_inches='tight')
            spatial_pdf_outputs.append(spatial_pdf_output)
            pdf.savefig(figure)
            plt.close(figure)

            for order_index, order in enumerate(orders):
                for component_index, component in enumerate(('Ux', 'Uy', 'Uz')):
                    figure, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
                    for direction_index, (direction, separation, clean_sf, prediction_sf) in enumerate((
                        ('x', separation_x, sf_x_clean, sf_x_prediction),
                        ('y', separation_y, sf_y_clean, sf_y_prediction),
                    )):
                        axis = axes[direction_index]
                        axis.loglog(separation, clean_sf[selected_index, component_index, order_index],
                                    label='Clean', **clean_line_style)
                        axis.loglog(
                            separation, prediction_sf[selected_index, component_index, order_index],
                            label='Prediction', **prediction_line_style
                        )
                        axis.set_title(f'{component}, {direction}-direction')
                        axis.title.set_fontsize(title_font_size)
                        axis.set_xlabel('Separation (mm)', fontsize=label_font_size)
                        axis.set_ylabel(rf'$\langle|\delta u|^{order}\rangle$',
                                        fontsize=label_font_size)
                        axis.grid(alpha=0.25, which='both')
                        axis.legend(fontsize=legend_font_size)
                        _format_publication_axis(axis, log_ticks=True)
                    figure.suptitle(
                        f'{component} absolute order-{order} structure functions: '
                        f'snapshot {snapshot_index}, t={times[snapshot_index] * 1e3:.3f} ms',
                        fontsize=suptitle_font_size
                    )
                    structure_output = os.path.join(
                        structure_function_directory,
                        f'{output_stem}_{component}_absolute_order_{order}_structure.pdf'
                    )
                    figure.savefig(structure_output, format='pdf', bbox_inches='tight')
                    structure_function_outputs.append(structure_output)
                    pdf.savefig(figure)
                    plt.close(figure)

            if 3 in orders:
                for component_index, component in enumerate(('Ux', 'Uy', 'Uz')):
                    figure, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
                    for direction_index, (direction, separation, clean_sf, prediction_sf) in enumerate((
                        ('x', separation_x, sf_x_signed_third_clean, sf_x_signed_third_prediction),
                        ('y', separation_y, sf_y_signed_third_clean, sf_y_signed_third_prediction),
                    )):
                        axis = axes[direction_index]
                        axis.plot(separation, clean_sf[selected_index, component_index],
                                  label='Clean', **clean_line_style)
                        axis.plot(separation, prediction_sf[selected_index, component_index],
                                  label='Prediction', **prediction_line_style)
                        axis.axhline(0.0, color='#333333', linewidth=0.8)
                        axis.set_title(f'{component}, {direction}-direction')
                        axis.title.set_fontsize(title_font_size)
                        axis.set_xlabel('Separation (mm)', fontsize=label_font_size)
                        axis.set_ylabel(r'$\langle(\delta u)^3\rangle$', fontsize=label_font_size)
                        axis.grid(alpha=0.25)
                        axis.legend(fontsize=legend_font_size)
                        _format_publication_axis(axis)
                    figure.suptitle(
                        f'{component} signed third-order structure functions: '
                        f'snapshot {snapshot_index}, t={times[snapshot_index] * 1e3:.3f} ms',
                        fontsize=suptitle_font_size
                    )
                    structure_output = os.path.join(
                        structure_function_directory,
                        f'{output_stem}_{component}_signed_third_structure.pdf'
                    )
                    figure.savefig(structure_output, format='pdf', bbox_inches='tight')
                    structure_function_outputs.append(structure_output)
                    pdf.savefig(figure)
                    plt.close(figure)

    metadata = {
        'clean_file': os.path.abspath(args.clean_file),
        'prediction_file': os.path.abspath(args.prediction_file),
        'holes_file': os.path.abspath(args.holes_file),
        'model_file': os.path.abspath(args.model_file),
        'model': model_metadata,
        'snapshot_alignment': alignment_metadata,
        'sample_rate_hz': args.sample_rate,
        'sample_interval_s': 1.0 / args.sample_rate,
        'snapshot_count': n_snapshots,
        'field_shape': list(clean.shape),
        'snapshot_indices': snapshot_indices.tolist(),
        'snapshot_selection_mode': selection_mode,
        'snapshot_selection_labels': selection_labels,
        'automatic_snapshot_count_per_extreme': args.automatic_snapshot_count,
        'selected_snapshot_coverage': coverage[snapshot_indices].tolist(),
        'selected_eliminated_vector_count': eliminated_count[snapshot_indices].tolist(),
        'structure_orders': orders.tolist(),
        'max_lag_x': max_lag_x,
        'max_lag_y': max_lag_y,
        'pdf_bins': args.pdf_bins,
        'coordinate_units': 'mm',
        'velocity_units': 'as stored after clean Min/Max dimensionalization',
        'notes': [
            'All PDFs and structure functions are instantaneous spatial statistics.',
            'No temporal mean or temporal averaging is used.',
            'Structure functions use in-domain pairs only, without periodic wrapping.',
            'Vorticity is recomputed identically from clean and predicted velocity fields.',
        ],
    }
    output_metadata = os.path.join(args.postprocess_output_dir, 'postprocess_metadata.json')
    with open(output_metadata, 'w') as metadata_file:
        json.dump(_json_compatible(metadata), metadata_file, indent=2, sort_keys=True)

    print(f'[POSTPROCESS] PDF report: {output_pdf}')
    print(f'[POSTPROCESS] Vector PDFs: {vector_directory} ({len(vector_outputs)} files)')
    print(
        f'[POSTPROCESS] Instantaneous spatial PDF plots: '
        f'{spatial_pdf_directory} ({len(spatial_pdf_outputs)} files)'
    )
    print(
        f'[POSTPROCESS] Instantaneous structure function plots: '
        f'{structure_function_directory} ({len(structure_function_outputs)} files)'
    )
    print(f'[POSTPROCESS] Velocity error histogram: {velocity_histogram}')
    print(f'[POSTPROCESS] Vorticity error histogram: {vorticity_histogram}')
    print(f'[POSTPROCESS] Numerical statistics: {output_npz}')
    print(f'[POSTPROCESS] Time-history CSV: {output_csv}')
    print(f'[POSTPROCESS] Metadata: {output_metadata}')


def diagnose_checkpoint_round_trip(vae, checkpoint_path, data_dict, device):
    """Compare in-memory and reloaded weights and isolate inference-mode effects."""
    print("=" * 80)
    print("Checkpoint / inference-mode diagnostics")
    print("=" * 80)

    original_training_mode = vae.training
    try:
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        if checkpoint['model_kwargs'] is None:
            print("[DIAGNOSTIC] checkpoint_round_trip=SKIPPED reason=legacy_checkpoint")
            return

        reloaded = CVAE(
            vh=VariableHandler(device=device, dtype=torch.float),
            **checkpoint['model_kwargs']
        ).to(device=device)
        reloaded.load_state_dict(checkpoint['state_dict'], strict=True)

        state_max_abs_diff = 0.0
        for name, original_tensor in vae.state_dict().items():
            reloaded_tensor = reloaded.state_dict()[name]
            tensor_diff = (original_tensor.detach() - reloaded_tensor.detach()).abs().max().item()
            state_max_abs_diff = max(state_max_abs_diff, float(tensor_diff))

        validation_images = data_dict['ValidationSet']
        validation_labels = data_dict['ValidationLabels']
        validation_mask = data_dict['ValidMask']
        clean_validation = data_dict['CleanValidation']
        validation_coverage = data_dict['ValidationCoverage']
        n_valid = len(validation_images)

        vae.eval()
        reloaded.eval()
        with torch.inference_mode():
            original_eval = vae.inference(validation_images, validation_labels, validation_mask)
            reloaded_eval_1 = reloaded.inference(validation_images, validation_labels, validation_mask)
            reloaded_eval_2 = reloaded.inference(validation_images, validation_labels, validation_mask)

        eval_error = CalculateErrorImages(
            '', validation_images, reloaded_eval_1, n_valid,
            clean_validation, validation_mask, validation_coverage
        )

        # Isolate training-mode BatchNorm statistics from stochastic dropout.
        encoder_dropout_rate = reloaded.encoder.dropout_rate
        decoder_dropout_rate = reloaded.decoder.dropout_rate
        skip_dropout_rate = reloaded.decoder.skip_dropout
        reloaded.encoder.dropout_rate = 0.0
        reloaded.decoder.dropout_rate = 0.0
        reloaded.decoder.skip_dropout = 0.0
        reloaded.train()
        with torch.inference_mode():
            batch_stats_prediction = reloaded.inference(
                validation_images, validation_labels, validation_mask
            )
        batch_stats_error = CalculateErrorImages(
            '', validation_images, batch_stats_prediction, n_valid,
            clean_validation, validation_mask, validation_coverage
        )

        reloaded.encoder.dropout_rate = encoder_dropout_rate
        reloaded.decoder.dropout_rate = decoder_dropout_rate
        reloaded.decoder.skip_dropout = skip_dropout_rate
        with torch.inference_mode():
            train_prediction_1 = reloaded.inference(
                validation_images, validation_labels, validation_mask
            )
            train_prediction_2 = reloaded.inference(
                validation_images, validation_labels, validation_mask
            )
        train_error_1 = CalculateErrorImages(
            '', validation_images, train_prediction_1, n_valid,
            clean_validation, validation_mask, validation_coverage
        )
        train_error_2 = CalculateErrorImages(
            '', validation_images, train_prediction_2, n_valid,
            clean_validation, validation_mask, validation_coverage
        )

        original_reload_diff = float(np.max(np.abs(original_eval - reloaded_eval_1)))
        eval_repeat_diff = float(np.max(np.abs(reloaded_eval_1 - reloaded_eval_2)))
        eval_batch_stats_diff = float(np.max(np.abs(reloaded_eval_1 - batch_stats_prediction)))
        train_repeat_diff = float(np.max(np.abs(train_prediction_1 - train_prediction_2)))

        round_trip_passed = state_max_abs_diff == 0.0 and original_reload_diff == 0.0
        print(
            f"[DIAGNOSTIC] checkpoint_format={checkpoint['arch_version']} "
            f"epoch={checkpoint['epoch_completed']} strict_load=PASS"
        )
        print(
            f"[DIAGNOSTIC] state_max_abs_diff={state_max_abs_diff:.8e} "
            f"round_trip={'PASS' if round_trip_passed else 'FAIL'}"
        )
        print(
            f"[DIAGNOSTIC] prediction_max_abs_diff "
            f"original_eval_vs_reload_eval={original_reload_diff:.8e} "
            f"reload_eval_repeat={eval_repeat_diff:.8e}"
        )
        print(
            f"[DIAGNOSTIC] validation_error eval={eval_error:.6f} "
            f"batch_stats_no_dropout={batch_stats_error:.6f} "
            f"train_dropout_1={train_error_1:.6f} "
            f"train_dropout_2={train_error_2:.6f}"
        )
        print(
            f"[DIAGNOSTIC] mode_prediction_max_abs_diff "
            f"eval_vs_batch_stats={eval_batch_stats_diff:.8e} "
            f"train_repeat={train_repeat_diff:.8e}"
        )
    except Exception as exc:
        print(f"[DIAGNOSTIC] checkpoint_round_trip=FAIL error={type(exc).__name__}: {exc}")
    finally:
        vae.train(original_training_mode)
    print("=" * 80)


# ---------------------------------------------------------------------------
# Training loop with 3-phase loss scheduling + diagnostics
# ---------------------------------------------------------------------------

def train_cvae_phased(config, data_dict, device, outdir, is_ray_tune=False):
    """
    Training function with 3-phase loss coefficient scheduling and
    anti-posterior-collapse mechanisms.
    """
    Images = data_dict['Images']
    clabels = data_dict['clabels']
    Mask = data_dict['Mask']
    CleanImg = data_dict['CleanImg']

    ValidationSet = data_dict['ValidationSet']
    ValidationLabels = data_dict['ValidationLabels']
    ValidationCoverage = data_dict['ValidationCoverage']
    ValidMask = data_dict['ValidMask']
    CleanValidation = data_dict['CleanValidation']

    vh = VariableHandler(device=device, dtype=torch.float)

    vae = CVAE(
        latent_size=config['latent_size'],
        num_labels=10,
        ImgSizeX=80,
        ImgSizeY=48,
        leaky_relu_slope=config['leaky_relu_slope'],
        dropout_rate=config['dropout_rate'],
        skip_dropout=config.get('skip_dropout', 0.3),
        skip_bottleneck_factor=config.get('skip_bottleneck_factor', 2),
        vh=vh
    ).to(device=device)

    optimizer = optim.Adam(vae.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

    tensor_Images = vh.tovar(Images)
    tensor_clabels = vh.tovar(clabels)
    tensor_Mask = vh.tovar(Mask)
    tensor_CleanImg = vh.tovar(CleanImg)

    if 'X' in data_dict and np.any(data_dict['X']):
        tensor_X = torch.as_tensor(data_dict['X'] * 1e-3, device=device, dtype=torch.float)
        tensor_Y = torch.as_tensor(data_dict['Y'] * 1e-3, device=device, dtype=torch.float)
    else:
        tensor_X = torch.linspace(0, 0.080, 80, device=device, dtype=torch.float)
        tensor_Y = torch.linspace(0, 0.048, 48, device=device, dtype=torch.float)

    outputFile = os.path.join(outdir, config['output_file'])
    outValidFile = os.path.join(outdir, "OutputImages_Validation.pdf")

    # Scheduling hyper-parameters
    max_beta = config['max_beta']
    max_gamma = config['max_gamma']
    beta_warm = config.get('beta_warm', 0.0)
    phase1_frac = config.get('phase1_frac', 0.05)
    phase2_frac = config.get('phase2_frac', 0.25)
    sigmoid_k = config.get('sigmoid_k', 10.0)
    n_cycles = config.get('n_cycles', 8)
    target_kld = config.get('target_kld', 5.0)
    total_epochs = config['epochs']
    latent_size = config['latent_size']

    for epoch in range(total_epochs):
        vae.train()

        current_beta, current_gamma, phase_label = compute_phased_coefficients(
            epoch, total_epochs, max_beta, max_gamma,
            beta_warm=beta_warm, sigmoid_k=sigmoid_k,
            phase1_frac=phase1_frac, phase2_frac=phase2_frac,
            n_cycles=n_cycles
        )

        permutation = torch.randperm(Images.shape[0])
        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_kld_raw = 0.0
        epoch_kld_loss = 0.0
        epoch_vort = 0.0
        epoch_eff_gamma = 0.0
        epoch_active_units = 0.0
        # Accumulate per-dim KLD across batches
        epoch_kld_per_dim = torch.zeros(latent_size, device='cpu')
        # Save last batch's latent stats for diagnostics
        last_mean = None
        last_log_var = None
        n_batches = 0

        for i in range(0, Images.shape[0], config['batch_size']):
            indices = permutation[i:i + config['batch_size']]

            batch_x = tensor_Images[indices]
            batch_c = tensor_clabels[indices]
            batch_mask = tensor_Mask[indices]
            batch_clean = tensor_CleanImg[indices]

            recon_x, mean, log_var, z = vae(batch_x, batch_c, batch_mask)

            loss, bce, kld_raw, kld_loss, vort, eff_gamma, kld_per_dim, active_units = loss_fn(
                recon_x, batch_x, mean, log_var,
                mask=batch_mask[:, 0:3],
                batch_clean=batch_clean,
                tensor_X=tensor_X, tensor_Y=tensor_Y,
                beta=current_beta, gamma_base=current_gamma,
                target_kld=target_kld
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_bce += bce.item()
            epoch_kld_raw += kld_raw.item()
            epoch_kld_loss += kld_loss.item()
            epoch_vort += vort.item()
            epoch_eff_gamma += float(eff_gamma)
            epoch_active_units += active_units
            epoch_kld_per_dim += kld_per_dim.cpu()
            last_mean = mean.detach()
            last_log_var = log_var.detach()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        avg_bce = epoch_bce / n_batches
        avg_kld_raw = epoch_kld_raw / n_batches
        avg_kld_loss = epoch_kld_loss / n_batches
        avg_vort = epoch_vort / n_batches
        avg_eff_gamma = epoch_eff_gamma / n_batches
        avg_active_units = epoch_active_units / n_batches
        avg_kld_per_dim = epoch_kld_per_dim / n_batches

        # Latent diagnostics from last batch
        mean_abs_mu = last_mean.abs().mean().item() if last_mean is not None else 0.0
        mean_var = last_log_var.exp().mean().item() if last_log_var is not None else 0.0
        mean_logvar = last_log_var.mean().item() if last_log_var is not None else 0.0

        if not is_ray_tune:
            # ---- Per-epoch summary ----
            print(
                f"Epoch [{epoch + 1:>5d}/{total_epochs}] [{phase_label}] "
                f"Loss: {avg_loss:.4e} | "
                f"BCE: {avg_bce:.4e} | "
                f"β·KLDloss: {current_beta * avg_kld_loss:.4e} "
                f"(β={current_beta:.4e}, KLD_raw={avg_kld_raw:.4e}, tgt={target_kld:.2f}) | "
                f"γ·Vort: {avg_eff_gamma * avg_vort:.4e} "
                f"(γ_eff={avg_eff_gamma:.2e}, VortMSE={avg_vort:.4e}) | "
                f"AU: {avg_active_units:.0f}/{latent_size} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

            # ---- Every 10 epochs: per-dimension KLD + latent stats ----
            if (epoch + 1) % 10 == 0:
                kld_str = ", ".join(f"{v:.4f}" for v in avg_kld_per_dim.numpy())
                print(f"  >> KLD/dim: [{kld_str}]")
                print(
                    f"  >> Latent stats: mean|μ|={mean_abs_mu:.4f}, "
                    f"mean(var)={mean_var:.4f}, mean(logvar)={mean_logvar:.4f}"
                )
                # Collapse warning
                if avg_active_units < 2 and epoch > int(phase1_frac * total_epochs):
                    print(
                        f"  !! WARNING: Only {avg_active_units:.0f} active units — "
                        f"posterior may be collapsing. Consider increasing skip_dropout "
                        f"or decreasing max_beta."
                    )

            # ---- Every 50 epochs: checkpoint + validation + health ----
            if (epoch + 1) % 50 == 0:
                save_checkpoint(
                    outputFile, vae, config,
                    epoch=epoch + 1, total_epochs=total_epochs,
                    metrics={
                        'avg_loss': float(avg_loss),
                        'avg_bce': float(avg_bce),
                        'avg_kld_raw': float(avg_kld_raw),
                        'avg_kld_loss': float(avg_kld_loss),
                        'avg_vort': float(avg_vort),
                        'avg_active_units': float(avg_active_units),
                        'kld_per_dim': avg_kld_per_dim.tolist(),
                        'mean_abs_mu': float(mean_abs_mu),
                        'mean_var': float(mean_var),
                        'current_beta': float(current_beta),
                        'current_gamma': float(current_gamma),
                        'phase_label': phase_label,
                    },
                )
                vae.eval()
                PredictImage = vae.inference(ValidationSet, ValidationLabels, ValidMask)
                ErrMag = CalculateErrorImages(
                    outValidFile, ValidationSet, PredictImage, config['n_valid'],
                    CleanValidation, ValidMask, ValidationCoverage
                )
                print(
                    f"  >> Checkpoint: Val Err: {ErrMag:.4f}"
                )
                # Latent health summary
                dims_above_01 = (avg_kld_per_dim > 0.1).sum().item()
                dims_above_05 = (avg_kld_per_dim > 0.5).sum().item()
                kld_gap = avg_kld_raw - target_kld
                print(
                    f"  >> LatentHealth: active(>0.1)={dims_above_01}/{latent_size}, "
                    f"active(>0.5)={dims_above_05}/{latent_size}, "
                    f"KLD_raw={avg_kld_raw:.4f} (target={target_kld:.2f}, gap={kld_gap:+.4f}), "
                    f"mean|μ|={mean_abs_mu:.4f}, mean(σ²)={mean_var:.4f}"
                )
                if abs(kld_gap) < 0.5:
                    print(f"  >> STATUS: KLD near target — latent space is healthy")
                elif kld_gap < -target_kld * 0.8:
                    print(f"  >> STATUS: KLD << target — likely collapsing. "
                          f"Increase skip_dropout or decrease max_beta")
                elif kld_gap > target_kld:
                    print(f"  >> STATUS: KLD >> target — latent may be over-regularised. "
                          f"Increase target_kld or decrease max_beta")
                vae.train()
        else:
            vae.eval()
            PredictImage = vae.inference(ValidationSet, ValidationLabels, ValidMask)
            ErrMag = CalculateErrorImages(
                outValidFile, ValidationSet, PredictImage, config['n_valid'],
                CleanValidation, ValidMask, ValidationCoverage
            )

            from ray import tune as _tune
            _tune.report({
                "training_loss": avg_loss,
                "val_error": ErrMag,
                "bce": avg_bce,
                "kld_raw": avg_kld_raw,
                "kld_target_loss": avg_kld_loss,
                "vort_mse": avg_vort,
                "active_units": avg_active_units,
            })

    return vae


# ---------------------------------------------------------------------------
# Evaluation 
# ---------------------------------------------------------------------------

def evaluate_cvae(vae, data_dict, args, outdir):
    vae.eval()

    Images = data_dict['Images']
    clabels = data_dict['clabels']
    Mask = data_dict['Mask']
    CleanImg = data_dict['CleanImg']
    X = data_dict['X']
    Y = data_dict['Y']
    Max = data_dict['Max']
    Min = data_dict['Min']

    PredictImg = np.zeros_like(CleanImg)
    cleanVort = np.zeros((CleanImg.shape[0], CleanImg.shape[2], CleanImg.shape[3]))
    PredictVort = np.zeros((CleanImg.shape[0], CleanImg.shape[2], CleanImg.shape[3]))
    VortError = np.zeros(CleanImg.shape[0])
    UxError = np.zeros(CleanImg.shape[0])
    UyError = np.zeros(CleanImg.shape[0])
    UzError = np.zeros(CleanImg.shape[0])

    start = time.time()
    for i in range(0, Images.shape[0], args.batch_size):
        bs = min(args.batch_size, Images.shape[0] - i)
        indices = slice(i, i + bs)
        PredictImg[indices] = vae.inference(Images[indices], clabels[indices], Mask[indices])

        CleanImg[indices] = CleanImg[indices] * (Max[indices] - Min[indices]) + Min[indices]
        PredictImg[indices] = PredictImg[indices] * (Max[indices] - Min[indices]) + Min[indices]

        for k in range(bs):
            idx = i + k
            cleanVort[idx] = np.gradient(CleanImg[idx, 1], X*1e-3, axis=0) - np.gradient(CleanImg[idx, 0], Y*1e-3, axis=1)
            PredictVort[idx] = np.gradient(PredictImg[idx, 1], X*1e-3, axis=0) - np.gradient(PredictImg[idx, 0], Y*1e-3, axis=1)

            VortError[idx] = LA.norm(cleanVort[idx] - PredictVort[idx], 'fro') / LA.norm(cleanVort[idx], 'fro')
            UxError[idx] = LA.norm(CleanImg[idx, 0] - PredictImg[idx, 0], 'fro') / LA.norm(CleanImg[idx, 0], 'fro')
            UyError[idx] = LA.norm(CleanImg[idx, 1] - PredictImg[idx, 1], 'fro') / LA.norm(CleanImg[idx, 1], 'fro')
            UzError[idx] = LA.norm(CleanImg[idx, 2] - PredictImg[idx, 2], 'fro') / LA.norm(CleanImg[idx, 2], 'fro')

    print(f"Maximum error in Ux {np.max(UxError):.4f}, Uy {np.max(UyError):.4f}, Uz {np.max(UzError):.4f}")
    print(f"Maximum error in vorticity {np.max(VortError):.4f}")

    coverage = 1.0 - np.sum(Mask[:, 0], axis=(1, 2)) / (Mask.shape[2] * Mask.shape[3])
    for error_name, errors in (
        ('Ux', UxError), ('Uy', UyError), ('Uz', UzError), ('Vorticity', VortError)
    ):
        worst_index = int(np.argmax(errors))
        print(
            f"[EVAL] {error_name:9s} mean={np.mean(errors):.6f} "
            f"median={np.median(errors):.6f} p95={np.quantile(errors, 0.95):.6f} "
            f"max={errors[worst_index]:.6f} worst_index={worst_index} "
            f"coverage={coverage[worst_index]:.6f}"
        )

    end = time.time()
    print(f"Evaluation Time taken in hours {(end-start)/3600.0:.4f}")

    output_file = os.path.join(outdir, 'AllImagesResults_phased_loss.npz')
    np.savez(output_file, PredictImg=PredictImg, cleanVort=cleanVort, PredictVort=PredictVort,
             X=X, Y=Y, VortError=VortError, UxError=UxError, UyError=UyError, UzError=UzError)


# ---------------------------------------------------------------------------
# Ray Tune integration
# ---------------------------------------------------------------------------

def tune_hyperparameters_phased(data_dict, num_samples=10, max_num_epochs=100, outdir="./ray_results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    search_space = {
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "latent_size": tune.choice([8, 16]),
        "batch_size": tune.choice([32, 64]),
        "dropout_rate": tune.uniform(0.1, 0.3),
        "leaky_relu_slope": tune.uniform(0.1, 0.3),
        # Phase scheduling
        "max_beta": tune.loguniform(1e-5, 1e-2),
        "max_gamma": tune.loguniform(1e-8, 1e-5),
        "target_kld": tune.uniform(2.0, 10.0),
        "beta_warm": 0.0,
        "phase1_frac": tune.uniform(0.05, 0.20),
        "phase2_frac": tune.uniform(0.20, 0.50),
        "skip_dropout": tune.uniform(0.1, 0.5),
        "skip_bottleneck_factor": tune.choice([2, 4]),
        "sigmoid_k": 10.0,
        "n_cycles": tune.choice([4, 8]),
        "epochs": max_num_epochs,
        "n_valid": 5,
        "output_file": "tune_tmp_phased.pkl"
    }

    asha = ASHAScheduler(
        metric="val_error",
        mode="min",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                train_cvae_phased, data_dict=data_dict,
                device=device, outdir=outdir, is_ray_tune=True
            ),
            resources={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0}
        ),
        tune_config=tune.TuneConfig(scheduler=asha, num_samples=num_samples),
        param_space=search_space,
    )
    results = tuner.fit()

    best_result = results.get_best_result("val_error", "min")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation error: {}".format(best_result.metrics.get("val_error")))
    return best_result


# ---------------------------------------------------------------------------
# Entry routine
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CVAE PIV gap-fill with 3-phase loss scaling + anti-collapse strategies'
    )
    parser.add_argument('-f', '--file', dest='file', type=str, default='ResizePIV.npz')
    parser.add_argument('-b', '--batchSize', dest='batch_size', type=int, default=64)
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=500)
    parser.add_argument('-ns', '--nSamples', dest='nSamples', type=int, default=8000)
    parser.add_argument('-l', '--nl', dest='latent_size', type=int, default=16)
    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float, default=1e-4)
    parser.add_argument('-v', '--nValidation', dest='nValid', type=int, default=5)
    parser.add_argument('-H', '--withHoles', dest='TrainHoles', type=int, default=1)
    parser.add_argument('-o', '--output', dest='oFile', type=str, default='CVAEoutput_phased.pkl')

    # ---- Anti-collapse & scheduling parameters ----
    parser.add_argument('--max_beta', type=float, default=1e-3,
                        help='Weight for target-KLD loss term. Since the loss is '
                             '(KLD−target)², this should be small (default: 1e-3)')
    parser.add_argument('--target_kld', type=float, default=5.0,
                        help='Target KLD value (sum over dims, mean over batch). '
                             'For latent_size=16, 5.0 ≈ 0.31 nats/dim (default: 5.0)')
    parser.add_argument('--max_gamma', type=float, default=3e-6,
                        help='Maximum gamma_base (vorticity loss weight) (default: 3e-6)')
    parser.add_argument('--beta_warm', type=float, default=0.0,
                        help='Phase 1 beta. Set to 0 for pure reconstruction warmup (default: 0.0)')
    parser.add_argument('--skip_dropout', type=float, default=0.3,
                        help='Dropout probability on skip connections; forces decoder to '
                             'rely on z (default: 0.3)')
    parser.add_argument('--skip_bottleneck_factor', type=int, default=2,
                        help='Channel reduction factor for 1×1 skip bottleneck convolutions. '
                             '2 = halve channels (default: 2)')
    parser.add_argument('--n_cycles', type=int, default=8,
                        help='Number of beta annealing cycles in Phase 2 (default: 8)')
    parser.add_argument('--phase1_frac', type=float, default=0.05,
                        help='Fraction of epochs for Phase 1 – reconstruction only (default: 0.05)')
    parser.add_argument('--phase2_frac', type=float, default=0.25,
                        help='Fraction of epochs marking end of Phase 2 – KLD ramp (default: 0.25)')
    parser.add_argument('--sigmoid_k', type=float, default=10.0,
                        help='(unused) Kept for backward compatibility (default: 10.0)')

    # Ray Tune
    parser.add_argument('--tune', action='store_true', help='Run Ray Tune hyperparameter search')
    parser.add_argument('--tune_samples', type=int, default=10, help='Number of Ray Tune trials')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional NumPy/PyTorch seed for comparable reruns')
    parser.add_argument('--diagnose-checkpoint', action='store_true',
                        help='Compare in-memory/reloaded weights and eval/train inference modes')
    parser.add_argument('--postprocess', action='store_true',
                        help='Run standalone turbulence-statistics post-processing without training')
    parser.add_argument('--clean-file', type=str,
                        help='Clean NPZ containing ImgScale, Min, Max, X, and Y')
    parser.add_argument('--prediction-file', type=str,
                        help="Prediction NPZ containing dimensional 'PredictImg'")
    parser.add_argument('--holes-file', type=str,
                        help="Holes NPZ containing the validity 'Mask' array")
    parser.add_argument('--model-file', type=str,
                        help='Model checkpoint used to produce the prediction (for provenance)')
    parser.add_argument('--postprocess-output-dir', type=str, default='postprocess_statistics',
                        help='Directory for standalone post-processing outputs')
    parser.add_argument('--sample-rate', type=float, default=10000.0,
                        help='PIV acquisition rate in Hz (default: 10000)')
    parser.add_argument('--snapshot-indices', type=int, nargs='+', default=None,
                        help='Snapshot indices for all instantaneous plots; when omitted, '
                             'select coverage extremes from --holes-file')
    parser.add_argument('--automatic-snapshot-count', type=int, default=5,
                        help='Number of most and least gapped snapshots selected when '
                             '--snapshot-indices is omitted (default: 5 per extreme)')
    parser.add_argument('--structure-orders', type=int, nargs='+', default=[2, 3],
                        help='Absolute structure-function orders; order 3 also saves signed values')
    parser.add_argument('--max-lag-x', type=int, default=None,
                        help='Maximum x-direction lag in grid points (default: nx/4)')
    parser.add_argument('--max-lag-y', type=int, default=None,
                        help='Maximum y-direction lag in grid points (default: ny/4)')
    parser.add_argument('--pdf-bins', type=int, default=80,
                        help='Number of bins for field PDFs and relative-error histograms '
                            '(default: 80)')
    args = parser.parse_args()

    if args.postprocess:
        missing_arguments = [
            name for name, value in (
                ('--clean-file', args.clean_file),
                ('--prediction-file', args.prediction_file),
                ('--holes-file', args.holes_file),
                ('--model-file', args.model_file),
            ) if not value
        ]
        if missing_arguments:
            parser.error('--postprocess requires ' + ', '.join(missing_arguments))
        run_postprocessing(args)
        raise SystemExit(0)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    outdir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%m-%d_%H-%M') + '_phased')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Loading and preprocessing data...")
    data_dict = load_and_preprocess_data(args.file, args.nSamples, args.TrainHoles, args.nValid)

    scalerFile = os.path.join(outdir, "LabelScaler.pkl")
    joblib.dump(data_dict['transform'], scalerFile)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Runtime: device={device}, torch={torch.__version__}, "
        f"cuda_available={torch.cuda.is_available()}, seed={args.seed}"
    )

    if args.tune:
        print(f"Starting Ray Tune. Trials: {args.tune_samples}, max_epochs: {args.epochs}")
        tune_hyperparameters_phased(data_dict, num_samples=args.tune_samples,
                                    max_num_epochs=args.epochs, outdir=outdir)
        print("Done tuning.")
    else:
        config = {
            'latent_size': args.latent_size,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'leaky_relu_slope': 0.2,
            'dropout_rate': 0.2,
            # 3-phase scheduling
            'max_beta': args.max_beta,
            'max_gamma': args.max_gamma,
            'beta_warm': args.beta_warm,
            'phase1_frac': args.phase1_frac,
            'phase2_frac': args.phase2_frac,
            'sigmoid_k': args.sigmoid_k,
            'n_cycles': args.n_cycles,
            # Anti-collapse
            'target_kld': args.target_kld,
            'skip_dropout': args.skip_dropout,
            'skip_bottleneck_factor': args.skip_bottleneck_factor,
            'n_valid': args.nValid,
            'output_file': args.oFile,
        }

        print("=" * 80)
        print("3-Phase Loss Scheduling + Anti-Collapse Configuration")
        print("=" * 80)
        print(f"  Phase 1 (0 – {args.phase1_frac*100:.0f}%): beta_warm={args.beta_warm:.1e}, gamma=0  [pure reconstruction]")
        print(f"  Phase 2 ({args.phase1_frac*100:.0f}% – {args.phase2_frac*100:.0f}%): "
              f"beta cyclical 0→{args.max_beta:.1e} ({args.n_cycles} cycles), gamma=0")
        print(f"  Phase 3 ({args.phase2_frac*100:.0f}% – 100%): beta={args.max_beta:.1e} (frozen), "
              f"gamma linear 0→{args.max_gamma:.1e}")
        print(f"  Target KLD:            {args.target_kld:.2f} "
              f"({args.target_kld/args.latent_size:.3f} nats/dim)")
        print(f"  Skip dropout:          {args.skip_dropout:.2f}")
        print(f"  Skip bottleneck:       1/{args.skip_bottleneck_factor}× channels")
        print(f"  Latent size:           {args.latent_size}")
        print(f"  Batch / Epochs / LR:   {args.batch_size} / {args.epochs} / {args.learning_rate:.1e}")
        print("=" * 80)
        print("Starting training...")

        start = time.time()
        vae = train_cvae_phased(config, data_dict, device, outdir, is_ray_tune=False)
        end = time.time()
        print(f"Training Time taken in hours {(end-start)/3600.0:.4f}")

        if args.diagnose_checkpoint:
            diagnose_checkpoint_round_trip(
                vae, os.path.join(outdir, args.oFile), data_dict, device
            )

        print("Starting evaluation...")
        evaluate_cvae(vae, data_dict, args, outdir)
