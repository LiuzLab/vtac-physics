"""
Physics-informed deep learning for false ventricular tachycardia (VT) alarm reduction (VTaC).

SE-ResNet1D classifier trained with ICU-realistic augmentations and two auxiliary
reconstruction heads (training only):
  - ABP via a differentiable 3-element Windkessel forward simulation (physics-informed)
  - PLETH via an MLP decoder (data-driven)

Channel mapping: ECG leads, PLETH and ABP are placed in 14 fixed slots by signal name.
The generic precordial lead "V" goes to the precordial (V1) slot; channels without a
canonical name are dropped. Unoccupied slots are zero-filled and flagged in the mask.

Paper: https://arxiv.org/abs/2609.08992
"""
import os
import sys
import time
import socket
import argparse
import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import warnings
import json
import optuna

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 1. Canonical Channel Ordering
# ──────────────────────────────────────────────

N_CANONICAL_CHANNELS = 14
PLETH_SLOT = 12
ABP_SLOT = 13

SIGNAL_NAME_TO_SLOT = {
    "I": 0, "MLI": 0, "ECG1": 0, "ECG": 0,
    "II": 1, "MLII": 1, "ECG2": 1,
    "III": 2,
    "AVR": 3,
    "AVL": 4,
    "AVF": 5,
    "V1": 6, "MCL1": 6, "MCL": 6, "V": 6,   # generic precordial lead -> precordial slot
    "V2": 7, "MCL2": 7,
    "V3": 8,
    "V4": 9,
    "V5": 10,
    "V6": 11,
    "PLETH": 12, "PPG": 12,
    "ABP": 13, "ART": 13, "UAP": 13, "PAP": 13,
}

def map_channels_to_canonical(header):
    sig_names = [s.upper().strip() for s in header.sig_name]
    slot_map = {}
    unmapped = []

    for ch_idx, name in enumerate(sig_names):
        if name in SIGNAL_NAME_TO_SLOT:
            slot = SIGNAL_NAME_TO_SLOT[name]
            if slot not in slot_map:
                slot_map[slot] = ch_idx
            else:
                unmapped.append((ch_idx, name))
        else:
            unmapped.append((ch_idx, name))

    # unmapped / duplicate channels are dropped (no fallback into free slots)
    return slot_map


# ──────────────────────────────────────────────
# 2. Data Caching (with canonical ordering)
# ──────────────────────────────────────────────

CACHE_DIR = None  # overridden by --cache_dir

def get_cache_path(data_root, split_name, n_channels, window_sec):
    cache_dir = CACHE_DIR or os.path.join(data_root, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    tag = f"{split_name}_ch{n_channels}_w{window_sec}_canonical_v2"   # v2 = corrected slot mapping; do not reuse older caches
    return os.path.join(cache_dir, f"{tag}.npz")

def build_and_cache_split(data_root, split_df, labels_df, n_channels, window_sec, sample_rate, split_name):
    cache_path = get_cache_path(data_root, split_name, n_channels, window_sec)

    if os.path.exists(cache_path):
        print(f"  Cache hit: {cache_path}")
        return cache_path

    print(f"  Building cache for {split_name}...")
    window_samples = window_sec * sample_rate
    alarm_sample = 5 * 60 * sample_rate

    merged = split_df.merge(labels_df, on=["record", "event"], how="inner")
    merged["decision"] = merged["decision"].map(
        {True: True, False: False, "True": True, "False": False}
    )
    merged = merged.dropna(subset=["decision"]).reset_index(drop=True)

    n = len(merged)
    all_sigs = np.zeros((n, n_channels, window_samples), dtype=np.float32)
    all_masks = np.zeros((n, n_channels, window_samples), dtype=np.float32)
    all_labels = np.zeros(n, dtype=np.int64)

    start = alarm_sample - window_samples
    end = alarm_sample

    for i, (_, row) in enumerate(merged.iterrows()):
        record_path = os.path.join(data_root, "waveforms", row["record"], row["event"])
        rec = wfdb.rdrecord(record_path, sampfrom=start, sampto=end)
        sig = rec.p_signal
        sig = np.nan_to_num(sig, nan=0.0)

        means = sig.mean(axis=0, keepdims=True)
        stds = sig.std(axis=0, keepdims=True)
        stds[stds < 1e-8] = 1.0
        sig = (sig - means) / stds
        sig = sig.T.astype(np.float32)

        slot_map = map_channels_to_canonical(rec)
        for slot, ch_idx in slot_map.items():
            if slot < n_channels:
                all_sigs[i, slot, :] = sig[ch_idx, :]
                all_masks[i, slot, :] = 1.0

        all_labels[i] = int(row["decision"])
        
        if (i + 1) % 500 == 0:
            print(f"    Processed {i + 1}/{n}")

    np.savez_compressed(cache_path, sigs=all_sigs, masks=all_masks, labels=all_labels)
    print(f"  Saved cache: {cache_path} ({n} samples)")
    return cache_path


# ──────────────────────────────────────────────
# 3. Dataset with Augmentations and Aux Targets
# ──────────────────────────────────────────────


class VTaCCachedDataset(Dataset):
    """
    Combines Augmentations with Downsampled Physics Targets
    """
    def __init__(self, cache_path, augment=False, aug_params=None, aux_target_len=50):
        data = np.load(cache_path)
        self.sigs = data["sigs"]
        self.masks = data["masks"]
        self.labels = data["labels"]
        self.augment = augment
        
        self.aux_target_len = aux_target_len
        self.ds_factor = self.sigs.shape[2] // aux_target_len

        self.max_jitter = 0
        self.noise_sigma = 0.0
        self.ch_drop_prob = 0.0
        self.scale_range = 0.0
        self.wander_amp = 0.0

        if aug_params:
            self.max_jitter = aug_params.get("max_jitter", 0)
            self.noise_sigma = aug_params.get("noise_sigma", 0.0)
            self.ch_drop_prob = aug_params.get("ch_drop_prob", 0.0)
            self.scale_range = aug_params.get("scale_range", 0.0)
            self.wander_amp = aug_params.get("wander_amp", 0.0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sig = self.sigs[idx].copy()
        mask = self.masks[idx].copy()
        label = self.labels[idx]
        n_ch, T = sig.shape

        if self.augment:
            # 1. Temporal jitter
            if self.max_jitter > 0:
                shift = np.random.randint(-self.max_jitter, self.max_jitter + 1)
                if shift != 0:
                    sig = np.roll(sig, shift, axis=1)
                    if shift > 0:
                        sig[:, :shift] = 0.0
                    else:
                        sig[:, shift:] = 0.0

            # 2. Gaussian noise
            if self.noise_sigma > 0:
                noise = np.random.normal(0, self.noise_sigma, sig.shape).astype(np.float32)
                sig += noise * mask

            # 3. Channel dropout
            if self.ch_drop_prob > 0:
                drop = np.random.random(n_ch) < self.ch_drop_prob
                sig[drop] = 0.0
                mask[drop] = 0.0

            # 4. Amplitude scaling
            if self.scale_range > 0:
                scales = np.random.uniform(
                    1.0 - self.scale_range, 1.0 + self.scale_range, (n_ch, 1)
                ).astype(np.float32)
                sig *= scales

            # 5. Baseline wander
            if self.wander_amp > 0:
                freq = np.random.uniform(0.1, 0.5)
                phase = np.random.uniform(0, 2 * np.pi)
                amp = np.random.uniform(0, self.wander_amp)
                t = np.arange(T, dtype=np.float32) / 250.0
                wander = (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)
                for ch in range(min(12, n_ch)):
                    if mask[ch, 0] > 0:
                        sig[ch] += wander

        # Assemble the input
        x = np.concatenate([sig, mask], axis=0)
        
        # Calculate Physics Aux Targets AFTER augmentations
        pleth_target = sig[PLETH_SLOT].reshape(self.aux_target_len, self.ds_factor).mean(axis=1)
        abp_target = sig[ABP_SLOT].reshape(self.aux_target_len, self.ds_factor).mean(axis=1)
        pleth_avail = mask[PLETH_SLOT, 0]
        abp_avail = mask[ABP_SLOT, 0]

        return (torch.from_numpy(x), 
                torch.tensor(label, dtype=torch.float32),
                torch.from_numpy(pleth_target),
                torch.from_numpy(abp_target),
                torch.tensor(pleth_avail, dtype=torch.float32),
                torch.tensor(abp_avail, dtype=torch.float32))


# ──────────────────────────────────────────────
# 4. Fast Windkessel ODE Solver
# ──────────────────────────────────────────────

class WindkesselSolver(nn.Module):
    """
    Differentiable 3-element Windkessel model.

    Operates directly at aux_target_len resolution (e.g. 50 steps)
    instead of full 2500 steps. Uses exact exponential integration
    for unconditional stability.

    ODE:  C * dP_wk/dt = Q(t) - P_wk(t)/R_p
    Exact step: P_wk[t+1] = gamma * P_wk[t] + R_p*(1-gamma)*Q[t]
      where gamma = exp(-dt/(R_p*C))
    Output: P(t) = R_c * Q(t) + P_wk(t)
    """

    def __init__(self, target_len=50, window_sec=10):
        super().__init__()
        self.target_len = target_len
        self.dt = window_sec / target_len  # coarse dt, e.g. 0.2s for 50 steps

    def generate_flow(self, heart_rate, systolic_frac, amplitude, device):
        """Generate Q(t) at coarse resolution. All params: (batch,)."""
        batch = heart_rate.shape[0]
        T = self.target_len
        t = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0).expand(batch, -1)
        t_sec = t * self.dt  # actual time in seconds

        period = (60.0 / heart_rate.clamp(min=40, max=250)).unsqueeze(1)
        phase = (t_sec % period) / period
        sf = systolic_frac.unsqueeze(1)
        amp = amplitude.unsqueeze(1)

        systole_phase = (phase / sf).clamp(0, 1)
        in_systole = (phase < sf).float()
        q = amp * torch.sin(np.pi * systole_phase) * in_systole
        return q  # (batch, target_len)

    def integrate_robust(self, q, R_p, R_c, C):
        """
        Exact exponential integration via simple loop.
        T is small (25-250), so loop is fast and immune to float32 overflow.
        Recurrence: p_wk[t+1] = gamma * p_wk[t] + R_p*(1-gamma)*Q[t]
        Output:     P[t] = R_c * Q[t] + p_wk[t]
        """
        batch, T = q.shape
        device = q.device

        tau = (R_p * C).clamp(min=1e-4)
        gamma = torch.exp(-self.dt / tau).unsqueeze(1)  # (batch, 1)
        beta_seq = R_p.unsqueeze(1) * (1 - gamma) * q   # (batch, T)

        p_wk = torch.zeros(batch, 1, device=device)
        pressures = []

        for t in range(T):
            p_wk = p_wk * gamma + beta_seq[:, t:t+1]
            p_total = R_c.unsqueeze(1) * q[:, t:t+1] + p_wk
            pressures.append(p_total)

        return torch.cat(pressures, dim=1)

    def forward(self, heart_rate, systolic_frac, amplitude, R_p, R_c, C):
        """Forward always in float32 — exp/log math is unstable in float16."""
        with torch.amp.autocast("cuda", enabled=False):
            heart_rate = heart_rate.float()
            systolic_frac = systolic_frac.float()
            amplitude = amplitude.float()
            R_p = R_p.float()
            R_c = R_c.float()
            C = C.float()
            device = heart_rate.device
            q = self.generate_flow(heart_rate, systolic_frac, amplitude, device)
            p = self.integrate_robust(q, R_p, R_c, C)
            return p


class WindkesselParamHead(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6),
        )

    def forward(self, features):
        raw = self.net(features)
        heart_rate = 40.0 + 200.0 * torch.sigmoid(raw[:, 0])
        systolic_frac = 0.2 + 0.3 * torch.sigmoid(raw[:, 1])
        amplitude = F.softplus(raw[:, 2]) + 0.1
        R_p = F.softplus(raw[:, 3]) + 0.1
        R_c = F.softplus(raw[:, 4]) + 0.01
        C = F.softplus(raw[:, 5]) + 0.01
        return heart_rate, systolic_frac, amplitude, R_p, R_c, C


# ──────────────────────────────────────────────
# 5. Model Components
# ──────────────────────────────────────────────

class MultiScaleStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        b1 = out_channels // 3
        b2 = out_channels // 3
        b3 = out_channels - b1 - b2

        self.branch1 = nn.Conv1d(in_channels, b1, kernel_size=15, stride=2, padding=7, bias=False)
        self.branch2 = nn.Conv1d(in_channels, b2, kernel_size=51, stride=2, padding=25, bias=False)
        self.branch3 = nn.Conv1d(in_channels, b3, kernel_size=201, stride=2, padding=100, bias=False)

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.relu(self.bn(out))
        return self.pool(out)


class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(1, channels // reduction))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(1, channels // reduction), channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = x.mean(dim=2)
        y = self.sigmoid(self.fc2(self.relu(self.fc1(y)))).view(b, c, 1)
        return x * y


class SEResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock1D(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return self.relu(out)


class SEResNet1D(nn.Module):
    def __init__(self, in_channels, base_filters=48, use_multiscale=False,
                 aux_target_len=50, use_aux=True):
        super().__init__()
        bf = base_filters
        self.use_aux = use_aux
        feat_dim = bf * 8

        if use_multiscale:
            self.stem = MultiScaleStem(in_channels, bf)
        else:
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels, bf, kernel_size=15, stride=2, padding=7, bias=False),
                nn.BatchNorm1d(bf),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            )

        self.layer1 = nn.Sequential(SEResBlock1D(bf, bf), SEResBlock1D(bf, bf))
        self.layer2 = nn.Sequential(SEResBlock1D(bf, bf * 2, stride=2), SEResBlock1D(bf * 2, bf * 2))
        self.layer3 = nn.Sequential(SEResBlock1D(bf * 2, bf * 4, stride=2), SEResBlock1D(bf * 4, bf * 4))
        self.layer4 = nn.Sequential(SEResBlock1D(bf * 4, bf * 8, stride=2), SEResBlock1D(bf * 8, bf * 8))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(feat_dim, 1)

        if use_aux:
            self.pleth_decoder = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, aux_target_len),
            )
            self.wk_param_head = WindkesselParamHead(feat_dim)
            self.wk_solver = WindkesselSolver(target_len=aux_target_len)
            self.abp_scale = nn.Parameter(torch.tensor(1.0))
            self.abp_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, return_aux=False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        features = self.gap(x).squeeze(-1)
        logit = self.fc(self.dropout(features)).squeeze(-1)

        if return_aux and self.use_aux:
            pleth_pred = self.pleth_decoder(features)
            hr, sf, amp, rp, rc, c = self.wk_param_head(features)
            abp_raw = self.wk_solver(hr, sf, amp, rp, rc, c)
            abp_pred = self.abp_scale * abp_raw + self.abp_bias
            return logit, pleth_pred, abp_pred

        return logit


# ──────────────────────────────────────────────
# 6. EMA
# ──────────────────────────────────────────────

class ModelEMA:
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] -= (1.0 - self.decay) * (self.shadow[name] - param.data)

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]


# ──────────────────────────────────────────────
# 7. Losses
# ──────────────────────────────────────────────

class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=2.0, gamma_neg=2.0, penalty_fn=4.646):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.penalty_fn = penalty_fn

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
        loss_pos = -self.penalty_fn * targets * ((1 - probs) ** self.gamma_pos) * torch.log(probs)
        loss_neg = -(1 - targets) * (probs ** self.gamma_neg) * torch.log(1 - probs)
        return (loss_pos + loss_neg).mean()

def masked_mse_loss(pred, target, avail):
    if avail.sum() < 1:
        return torch.tensor(0.0, device=pred.device)
    mask = avail.unsqueeze(1)
    diff = (pred - target) ** 2
    return (diff * mask).sum() / (mask.sum() * pred.shape[1])


# ──────────────────────────────────────────────
# 8. Metrics
# ──────────────────────────────────────────────

METRIC_NAMES = ["TPR", "TNR", "PPV", "F1", "Score", "AUC"]

def challenge_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom = tp + tn + fp + 5 * fn
    return (tp + tn) / denom if denom > 0 else 0.0

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_prob = np.nan_to_num(y_prob, nan=0.5)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    cs = challenge_score(y_true, y_pred)
    return {"TPR": tpr, "TNR": tnr, "PPV": ppv, "F1": f1, "Score": cs, "AUC": auc}

def find_best_threshold(y_true, y_prob):
    y_prob = np.nan_to_num(y_prob, nan=0.5)
    best_t, best_cs = 0.5, -1
    for t in np.arange(0.1, 0.9, 0.01):
        cs = challenge_score(y_true, (y_prob >= t).astype(int))
        if cs > best_cs:
            best_cs = cs
            best_t = t
    return best_t


# ──────────────────────────────────────────────
# 9. Training / Evaluation Engine (with AMP)
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, cls_criterion, device, hparams, scaler):
    model.train()
    total_loss = 0
    use_aux = hparams.get("use_aux", True)
    lambda_pleth = hparams.get("lambda_pleth", 0.0)
    lambda_abp = hparams.get("lambda_abp", 0.0)
    use_gradclip = hparams.get("use_gradclip", False)
    clip_norm = hparams.get("clip_norm", 1.0)

    for batch in loader:
        x, labels, pleth_t, abp_t, pleth_a, abp_a = batch
        x, labels = x.to(device), labels.to(device)
        pleth_t, abp_t = pleth_t.to(device), abp_t.to(device)
        pleth_a, abp_a = pleth_a.to(device), abp_a.to(device)

        optimizer.zero_grad()

        with autocast("cuda"):
            if use_aux:
                logits, pleth_pred, abp_pred = model(x, return_aux=True)
                cls_loss = cls_criterion(logits, labels)
                pleth_loss = masked_mse_loss(pleth_pred, pleth_t, pleth_a)
                abp_loss = masked_mse_loss(abp_pred, abp_t, abp_a)
                loss = cls_loss + lambda_pleth * pleth_loss + lambda_abp * abp_loss
            else:
                logits = model(x)
                loss = cls_criterion(logits, labels)

        scaler.scale(loss).backward()
        if use_gradclip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        x = batch[0].to(device)
        labels = batch[1]
        with autocast("cuda"):
            logits = model(x, return_aux=False)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def run_experiment(seed, hparams, args_dict, train_cache, val_cache, test_cache,
                   n_channels, return_val_score=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()

    aug_params = {
        "max_jitter": hparams.get("max_jitter", 0),
        "noise_sigma": hparams.get("noise_sigma", 0.0),
        "ch_drop_prob": hparams.get("ch_drop_prob", 0.0),
        "scale_range": hparams.get("scale_range", 0.0),
        "wander_amp": hparams.get("wander_amp", 0.0),
    }

    aux_target_len = hparams.get("aux_target_len", 50)
    train_ds = VTaCCachedDataset(train_cache, augment=True, aug_params=aug_params, aux_target_len=aux_target_len)
    val_ds = VTaCCachedDataset(val_cache, augment=False, aux_target_len=aux_target_len)

    train_loader = DataLoader(train_ds, batch_size=args_dict["batch_size"], shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args_dict["batch_size"], shuffle=False,
                            num_workers=4, pin_memory=True)

    in_channels = n_channels * 2
    model = SEResNet1D(
        in_channels=in_channels,
        base_filters=hparams.get("base_filters", 48),
        use_multiscale=hparams.get("use_multiscale", False),
        aux_target_len=aux_target_len,
        use_aux=hparams.get("use_aux", True),
    ).to(device)

    cls_criterion = AsymmetricFocalLoss(penalty_fn=hparams["penalty_fn"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"],
                                 weight_decay=hparams["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args_dict["epochs"])
    scaler = GradScaler("cuda")

    use_ema = hparams.get("use_ema", True)
    if use_ema:
        ema = ModelEMA(model, decay=hparams.get("ema_decay", 0.99))

    best_val_score = -1
    patience_counter = 0
    best_model_thresh = 0.5
    best_epoch = 1
    save_path = os.path.join(args_dict["output_dir"], f"best_model_seed_{seed}.pt")

    for epoch in range(1, args_dict["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, cls_criterion, device, hparams, scaler)
        scheduler.step()

        if use_ema:
            ema.update(model)
            original_state = {k: v.clone() for k, v in model.state_dict().items()}
            ema.apply_shadow(model)
            val_labels, val_probs = evaluate(model, val_loader, device)
            model.load_state_dict(original_state)
        else:
            val_labels, val_probs = evaluate(model, val_loader, device)

        val_thresh = find_best_threshold(val_labels, val_probs)
        val_metrics = compute_metrics(val_labels, val_probs, threshold=val_thresh)

        if val_metrics["Score"] > best_val_score:
            best_val_score = val_metrics["Score"]
            patience_counter = 0
            if use_ema:
                original_state = {k: v.clone() for k, v in model.state_dict().items()}
                ema.apply_shadow(model)
                torch.save({"model_state_dict": model.state_dict(), "threshold": val_thresh,
                            "val_score": best_val_score, "epoch": epoch, "hparams": hparams}, save_path)
                model.load_state_dict(original_state)
            else:
                torch.save({"model_state_dict": model.state_dict(), "threshold": val_thresh,
                            "val_score": best_val_score, "epoch": epoch, "hparams": hparams}, save_path)
            best_model_thresh = val_thresh
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= args_dict["patience"]:
                break

    if return_val_score:
        return best_val_score

    # Only create test loader when actually needed
    test_ds = VTaCCachedDataset(test_cache, augment=False, aux_target_len=aux_target_len)
    test_loader = DataLoader(test_ds, batch_size=args_dict["batch_size"], shuffle=False,
                             num_workers=4, pin_memory=True)

    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    best_thresh = ckpt["threshold"]

    test_labels, test_probs = evaluate(model, test_loader, device)
    test_metrics = compute_metrics(test_labels, test_probs, threshold=best_thresh)
    test_metrics["best_epoch"] = best_epoch
    test_metrics["val_score"] = best_val_score
    test_metrics["threshold"] = best_model_thresh
    if device.type == "cuda":
        test_metrics["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return test_metrics


def append_run_log(path, record):
    """Append one JSON line per run (trial or seed) to the shared run log."""
    record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(),
              "pid": os.getpid(), **record}
    with open(path, "a") as f:
        f.write(json.dumps(record, default=float) + "\n")


# ──────────────────────────────────────────────
# 10. Parallel worker
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# 11. Optuna Search Space & Baselines
# ──────────────────────────────────────────────

def optuna_objective(trial, args_dict, train_cache, val_cache, test_cache, n_channels):
    hparams = {
        # Model architecture
        "base_filters": trial.suggest_categorical("base_filters", [32, 48, 64]),
        "use_multiscale": trial.suggest_categorical("use_multiscale", [True, False]),
        # Training
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
        "penalty_fn": trial.suggest_float("penalty_fn", 2.0, 12.0),
        "use_gradclip": trial.suggest_categorical("use_gradclip", [True, False]),
        "clip_norm": trial.suggest_float("clip_norm", 0.5, 5.0),
        # EMA
        "use_ema": trial.suggest_categorical("use_ema", [True, False]),
        "ema_decay": trial.suggest_float("ema_decay", 0.9, 0.999),
        # Augmentation intensities
        "max_jitter": trial.suggest_int("max_jitter", 0, 50),
        "noise_sigma": trial.suggest_float("noise_sigma", 0.0, 0.1),
        "ch_drop_prob": trial.suggest_float("ch_drop_prob", 0.0, 0.2),
        "scale_range": trial.suggest_float("scale_range", 0.0, 0.3),
        "wander_amp": trial.suggest_float("wander_amp", 0.0, 0.3),
        # Physics / Aux
        "use_aux": True,
        "lambda_pleth": trial.suggest_float("lambda_pleth", 0.001, 1.0, log=True),
        "lambda_abp": trial.suggest_float("lambda_abp", 0.001, 1.0, log=True),
        "aux_target_len": trial.suggest_categorical("aux_target_len", [25, 50, 100, 250]),
    }

    t0 = time.time()
    val_score = run_experiment(
        seed=42, hparams=hparams, args_dict=args_dict,
        train_cache=train_cache, val_cache=val_cache, test_cache=test_cache,
        n_channels=n_channels, return_val_score=True,
    )
    vram = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
    append_run_log(args_dict["run_log"], {"kind": "optuna_trial", "trial": trial.number,
                   "val_score": val_score, "hparams": hparams, "minutes": (time.time() - t0) / 60,
                   "peak_vram_mb": vram})
    return val_score

# Generic defaults: fallback for keys missing from --hparams_file and the first Optuna trial.
# Run --mode optuna to obtain tuned hyperparameters (written to best_hparams.json).
DEFAULT_HPARAMS = {
    "base_filters": 48,
    "use_multiscale": False,
    "lr": 1e-3,
    "weight_decay": 1e-6,
    "penalty_fn": 5.0,
    "use_ema": False,
    "ema_decay": 0.99,
    "max_jitter": 0,
    "noise_sigma": 0.0,
    "ch_drop_prob": 0.0,
    "scale_range": 0.0,
    "wander_amp": 0.0,
    "use_gradclip": False,
    "clip_norm": 1.0,
    "use_aux": True,
    "lambda_pleth": 0.05,
    "lambda_abp": 0.05,
    "aux_target_len": 50,
}

# Default seeds (drawn with numpy default_rng(2026)).
# NOTE: these differ from the seeds used in the paper, so performance may differ
# from the values reported there (seed-to-seed variation is several Challenge-Score points).
DEFAULT_SEEDS = [265, 1789, 3655, 6398, 8515]


# ──────────────────────────────────────────────
# 12. Main
# ──────────────────────────────────────────────

def split_events(split_df, labels_df, split_name):
    """(record, event) for each row of a split cache, in cache order."""
    merged = split_df[split_df["split"] == split_name].merge(labels_df, on=["record", "event"], how="inner")
    merged["decision"] = merged["decision"].map({True: True, False: False, "True": True, "False": False})
    return merged.dropna(subset=["decision"]).reset_index(drop=True)[["record", "event"]]


def predict(args, hparams, cache_path, split_df, labels_df, device):
    """Inference with trained checkpoint(s); auxiliary heads are not used."""
    aux_len = hparams["aux_target_len"]
    ds = VTaCCachedDataset(cache_path, augment=False, aux_target_len=aux_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    events = split_events(split_df, labels_df, args.split)
    assert len(events) == len(ds), (len(events), len(ds))
    summary = []
    for ckpt_path in args.checkpoint:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        hp_ck = {**hparams, **ckpt.get("hparams", {})}
        model = SEResNet1D(in_channels=N_CANONICAL_CHANNELS * 2, base_filters=hp_ck["base_filters"],
                           use_multiscale=hp_ck["use_multiscale"], aux_target_len=hp_ck["aux_target_len"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        thr = float(ckpt["threshold"])
        labels, probs = evaluate(model, loader, device)
        name = os.path.splitext(os.path.basename(ckpt_path))[0]
        out = events.assign(label=labels.astype(int), prob_true_alarm=probs,
                            pred_true_alarm=(probs >= thr).astype(int))
        out.to_csv(os.path.join(args.output_dir, f"predictions_{args.split}_{name}.csv"), index=False)
        m = compute_metrics(labels, probs, threshold=thr)
        summary.append({"checkpoint": ckpt_path, "threshold": thr, **{k: float(v) for k, v in m.items()}})
        print(f"{name} | thr {thr:.2f} | Score {m['Score']:.4f} | AUC {m['AUC']:.3f} | "
              f"TPR {m['TPR']:.3f} | PPV {m['PPV']:.3f}", flush=True)
    with open(os.path.join(args.output_dir, f"predict_{args.split}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="VTaC SE-ResNet + Augmentations + Physics-informed Windkessel")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--mode", type=str, default="eval", choices=["cache", "optuna", "eval", "predict"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--n_trials", type=int, default=50, help="TOTAL trials across all workers")
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--study_name", type=str, default="vtac_aug_physics")
    parser.add_argument("--storage", type=str, default=None, help="Optuna journal file (shared by workers)")
    parser.add_argument("--hparams_file", type=str, default="best_hparams.json",
                        help="hyperparameters for eval (written by --mode optuna)")
    parser.add_argument("--run_log", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, nargs="+", default=None,
                        help="predict mode: one or more trained checkpoints (best_model_seed_<seed>.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="predict mode: which split to run inference on")
    parser.add_argument("--gpu_mem_frac", type=float, default=None,
                        help="cap this process's share of GPU memory (for several processes per GPU)")
    args = parser.parse_args()

    global CACHE_DIR
    CACHE_DIR = args.cache_dir
    os.makedirs(args.output_dir, exist_ok=True)
    run_log = args.run_log or os.path.join(args.output_dir, "run_log.jsonl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda" and args.gpu_mem_frac:
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)

    split_df = pd.read_csv(os.path.join(args.data_root, "benchmark_data_split.csv"))
    labels_df = pd.read_csv(os.path.join(args.data_root, "event_labels.csv"))
    n_channels = N_CANONICAL_CHANNELS
    caches = {}
    for name in ["train", "val", "test"]:
        caches[name] = build_and_cache_split(args.data_root, split_df[split_df["split"] == name],
                                             labels_df, n_channels, 10, 250, name)
    if args.mode == "cache":
        return

    args_dict = {"output_dir": args.output_dir, "epochs": args.epochs, "batch_size": args.batch_size,
                 "patience": args.patience, "run_log": run_log}

    if args.mode == "optuna":
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend
        from optuna.study import MaxTrialsCallback
        from optuna.trial import TrialState
        storage = JournalStorage(JournalFileBackend(args.storage))
        study = optuna.create_study(direction="maximize", study_name=args.study_name, storage=storage,
                                    load_if_exists=True,
                                    sampler=optuna.samplers.TPESampler(seed=1000 + args.worker_id))
        if args.worker_id == 0 and len(study.trials) == 0:
            study.enqueue_trial(DEFAULT_HPARAMS)
            aug_baseline = DEFAULT_HPARAMS.copy()
            aug_baseline.update({"max_jitter": 25, "noise_sigma": 0.05, "ch_drop_prob": 0.1,
                                 "scale_range": 0.1, "wander_amp": 0.1})
            study.enqueue_trial(aug_baseline)
        study.optimize(lambda t: optuna_objective(t, args_dict, caches["train"], caches["val"],
                                                  caches["test"], n_channels),
                       callbacks=[MaxTrialsCallback(args.n_trials, states=(TrialState.COMPLETE, TrialState.FAIL))],
                       catch=(RuntimeError,))   # e.g. CUDA OOM: trial marked FAIL, worker continues
        done = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if done:
            best = max(done, key=lambda t: t.value)
            rec = {"trial": best.number, "val_score": best.value, "params": best.params}
            with open(os.path.join(args.output_dir, f"best_hparams_worker{args.worker_id}.json"), "w") as f:
                json.dump(rec, f, indent=2)
            # best over the whole (shared) study; the last worker to finish writes the final version
            with open(os.path.join(os.path.dirname(os.path.abspath(args.storage)), "best_hparams.json"), "w") as f:
                json.dump(rec, f, indent=2)
        return

    if os.path.exists(args.hparams_file):
        with open(args.hparams_file) as f:
            hp = json.load(f)
        hp = hp.get("params", hp)
    elif args.mode == "predict":
        hp = {}   # hyperparameters are read from the checkpoint
    else:
        sys.exit(f"{args.hparams_file} not found: run --mode optuna first or pass --hparams_file")
    full_hparams = {**DEFAULT_HPARAMS, **hp, "use_aux": True}

    if args.mode == "predict":
        predict(args, full_hparams, caches[args.split], split_df, labels_df, device)
        return

    # eval: train + test one model per seed
    for seed in args.seeds:
        t0 = time.time()
        metrics = run_experiment(seed, full_hparams, args_dict, caches["train"], caches["val"],
                                 caches["test"], n_channels, return_val_score=False)
        rec = {"kind": "seed_eval", "seed": seed, "minutes": (time.time() - t0) / 60,
               "hparams": full_hparams, **{k: float(v) for k, v in metrics.items()}}
        append_run_log(run_log, rec)
        with open(os.path.join(args.output_dir, f"result_seed_{seed}.json"), "w") as f:
            json.dump(rec, f, indent=2)
        print(f"Seed {seed:>5} | Score {metrics['Score']:.4f} | AUC {metrics['AUC']:.3f} | "
              f"TPR {metrics['TPR']:.3f} | PPV {metrics['PPV']:.3f} | epoch {metrics['best_epoch']} | "
              f"thr {metrics['threshold']:.2f}", flush=True)


if __name__ == "__main__":
    main()
