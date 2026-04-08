import os
import random
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from app.utils.global_utils import (
    load_mat_file,
    find_data_and_gt,
    preprocess_data,
    extract_patches,
    save_mat_file,
)


# =========================
# SPECTRAL ATTENTION
# =========================
class SpectralAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# =========================
# TRANSFORMER BLOCKS
# =========================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        out, _ = self.mha(x, x, x)
        return out


# =========================
# REDUCED 2D-CNN MODEL
# =========================
class Reduced2DCNN(nn.Module):
    """Reduced 2D-CNN with Spectral Attention for hyperspectral classification.

    Architecture:
      Conv2D(num_bands, 64) -> BN -> ReLU -> SpectralAttention
      Conv2D(64, 128) -> BN -> ReLU -> SpectralAttention
      AdaptiveAvgPool2d(1) -> FC(128,128) -> ReLU -> Dropout -> FC(128, num_classes)

    Input shape: (B, C, H, W) where C = num_bands (PCA components).
    """

    def __init__(self, num_bands: int = 30, num_classes: int = 16, patch_size: int = 9):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_bands, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.attn1 = SpectralAttention(64, reduction=8)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.attn2 = SpectralAttention(128, reduction=8)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.attn1(x)
        x = self.conv2(x)
        x = self.attn2(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x


def get_model(model_type: str = "simple", num_bands: int = 30, num_classes: int = 16, patch_size: int = 9) -> nn.Module:
    if model_type == "simple":
        return Reduced2DCNN(num_bands=num_bands, num_classes=num_classes, patch_size=patch_size)
    raise ValueError(f"Unknown model_type: {model_type}")


class HyperspectralPatchDataset(Dataset):
    def __init__(self, patches: np.ndarray, labels: np.ndarray, augment: bool = False):
        # patches: (N, H, W, C)
        self.X = torch.from_numpy(patches).float().permute(0, 3, 1, 2)
        self.y = torch.from_numpy(labels).long()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        patch = self.X[idx]
        label = self.y[idx]

        if self.augment:
            if random.random() < 0.5:
                patch = torch.flip(patch, dims=[1])
            if random.random() < 0.5:
                patch = torch.flip(patch, dims=[2])

        return patch, label


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
) -> Tuple[nn.Module, Optional[str], float]:
    """Train `model` using given DataLoaders and return trained model, best path, and best val acc.

    Saves best checkpoint to `save_path` if provided.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = 0.0
    best_path = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0

        val_acc = 0.0
        if val_loader is not None:
            model.eval()
            v_total = 0
            v_correct = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    out = model(xb)
                    preds = out.argmax(dim=1)
                    v_total += yb.size(0)
                    v_correct += (preds == yb).sum().item()
            val_acc = v_correct / v_total if v_total > 0 else 0.0

        if save_path and val_acc > best_val:
            best_val = val_acc
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), save_path)
            best_path = save_path

        print(f"Epoch {epoch+1}/{epochs} — train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} val_acc: {val_acc:.4f}")

    return model, best_path, best_val


def save_model(model: nn.Module, path: str):
    """Save full model state dict to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(
    path: str,
    model_type: str = "simple",
    num_bands: int = 30,
    num_classes: int = 16,
    patch_size: int = 9,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Instantiate a model for `model_type`, load its weights from `path`, and return the model.

    If the file at `path` is a state dict, it will be loaded into the instantiated model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(model_type, num_bands=num_bands, num_classes=num_classes, patch_size=patch_size)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        raise ValueError("Model file did not contain a state dict")
    model.to(device)
    model.eval()
    return model


def predict_image(model: nn.Module, image_path: str, patch_size: int = 9, n_components: int = 30) -> str:
    """Run inference over an input .mat image and save predicted map to disk.

    Returns the path to the saved .mat file containing `pred_map`.
    """
    mat = load_mat_file(image_path)
    
    # Extract 3D data (no need for GT in prediction)
    data = None
    for k, v in mat.items():
        if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 3:
            data = v
            break
    if data is None:
        raise ValueError(f"No 3D data array found in {image_path}")
    
    data_pre, _meta = preprocess_data(data, n_components=n_components)

    # generate full map (sliding window)
    h, w, c = data_pre.shape
    margin = patch_size // 2
    padded = np.pad(data_pre, ((margin, margin), (margin, margin), (0, 0)), mode="constant")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    pred_map = np.zeros((h, w), dtype=int)
    with torch.no_grad():
        for i in range(h):
            for j in range(w):
                patch = padded[i : i + patch_size, j : j + patch_size, :]
                patch_t = torch.from_numpy(patch).float().permute(2, 0, 1).unsqueeze(0).to(device)
                out = model(patch_t)
                pred = int(out.argmax(dim=1).cpu().numpy()[0]) + 1
                pred_map[i, j] = pred

    out_path = os.path.splitext(os.path.basename(image_path))[0] + "_pred_map.mat"
    out_full = os.path.join("predictions", out_path)
    save_mat_file(out_full, {"pred_map": pred_map})
    return out_full
