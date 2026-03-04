# =========================
# IMPORT LIBRARIES
# =========================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import scipy.io as sio
import random


# =========================
# REPRODUCIBILITY
# =========================


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# LOAD DATASET
# =========================
def load_dataset(name):
    base_path = r"C:\Users\MIT\vit"

    if name == 'IN':
        data = sio.loadmat(f"{base_path}\Indian_pines_corrected.mat")['indian_pines_corrected']
        gt = sio.loadmat(f"{base_path}\Indian_pines_gt.mat")['indian_pines_gt']

    elif name == 'SV':
        data = sio.loadmat(f"{base_path}\Salinas_corrected.mat")['salinas_corrected']
        gt = sio.loadmat(f"{base_path}\Salinas_gt.mat")['salinas_gt']

    elif name == 'UP':
        data = sio.loadmat(f"{base_path}\PaviaU.mat")['paviaU']
        gt = sio.loadmat(f"{base_path}\PaviaU_gt.mat")['paviaU_gt']

    else:
        raise ValueError("Unsupported dataset")

    return data, gt


# =========================
# PREPROCESSING
# =========================
def preprocess(data):
    h, w, c = data.shape
    data = data.reshape(-1, c)
    data = MinMaxScaler().fit_transform(data)
    return data.reshape(h, w, c)

# =========================
# PATCH EXTRACTION
# =========================
def create_patches(data, gt, patch_size=9):
    margin = patch_size // 2
    padded = np.pad(data, ((margin, margin), (margin, margin), (0, 0)), mode='constant')
    patches, labels = [], []

    for i in range(margin, padded.shape[0] - margin):
        for j in range(margin, padded.shape[1] - margin):
            if gt[i - margin, j - margin] != 0:
                patch = padded[i-margin:i+margin+1, j-margin:j+margin+1, :]
                patches.append(patch)
                labels.append(gt[i-margin, j-margin] - 1)

    return np.array(patches), np.array(labels)

# =========================
# VISION TRANSFORMER (HSI-OPTIMIZED)
# =========================
class ViT(nn.Module):
    def __init__(self, image_size=9, patch_size=3, num_classes=16,
                 num_bands=200, dim=64, depth=6, heads=8, mlp_dim=128):
        super().__init__()

        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * num_bands

        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_dim, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.contiguous().view(B, -1, p * p * C)

        x = self.patch_embed(patches)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1)]

        x = self.transformer(x)
        return self.mlp_head(x[:, 0])

# =========================
# TRAIN FUNCTION
# =========================
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total

# =========================
# EVALUATION METRICS (OA, AA, KAPPA)
# =========================
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    oa = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    aa = np.mean(cm.diagonal() / cm.sum(axis=1))
    kappa = cohen_kappa_score(y_true, y_pred)

    return oa, aa, kappa

# =========================
# MAIN SCRIPT
# =========================
if __name__ == "__main__":

    dataset = 'IN'   # 'IN', 'SV', 'UP'
    patch_size = 9
    batch_size = 32
    epochs = 1
    lr = 1e-3

    data, gt = load_dataset(dataset)
    data = preprocess(data)

    patches, labels = create_patches(data, gt, patch_size)
    num_classes = len(np.unique(labels))

    X_train, X_test, y_train, y_test = train_test_split(
        patches, labels, test_size=0.2, stratify=labels, random_state=42)

    X_train = torch.tensor(X_train).float().permute(0, 3, 1, 2)
    X_test = torch.tensor(X_test).float().permute(0, 3, 1, 2)

    y_train = torch.tensor(y_train).long()
    y_test = torch.tensor(y_test).long()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    model = ViT(
        image_size=patch_size,
        patch_size=3,
        num_classes=num_classes,
        num_bands=data.shape[-1]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loss, acc = train(model, train_loader, criterion, optimizer)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss:.4f} Acc: {acc:.4f}")

    oa, aa, kappa = evaluate(model, test_loader)
    print("\n==== FINAL RESULTS ====")
    print(f"OA     : {oa:.4f}")
    print(f"AA     : {aa:.4f}")
    print(f"Kappa  : {kappa:.4f}")


