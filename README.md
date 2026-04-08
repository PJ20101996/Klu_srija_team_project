# B.Tech Project Documentation

# Reduced 2D-CNN with Spectral Attention for Hyperspectral Satellite Image Classification

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Literature Background](#4-literature-background)
5. [Datasets Used](#5-datasets-used)
6. [System Requirements and Setup](#6-system-requirements-and-setup)
7. [Project Folder Structure](#7-project-folder-structure)
8. [Step-by-Step Implementation](#8-step-by-step-implementation)
   - 8.1 [Data Loading](#81-step-1--data-loading)
   - 8.2 [Data Preprocessing](#82-step-2--data-preprocessing-normalization--pca)
   - 8.3 [Patch Extraction](#83-step-3--patch-extraction)
   - 8.4 [Data Splitting](#84-step-4--data-splitting-trainvalidationtest)
   - 8.5 [PyTorch Dataset and DataLoader](#85-step-5--pytorch-dataset-and-dataloader)
   - 8.6 [Model Architecture](#86-step-6--model-architecture-reduced2dcnn-with-spectral-attention)
   - 8.7 [Training Pipeline](#87-step-7--training-pipeline)
   - 8.8 [Evaluation on Test Set](#88-step-8--evaluation-on-test-set)
   - 8.9 [Saving the Model with Metadata](#89-step-9--saving-the-model-with-metadata)
   - 8.10 [Full-Image Prediction (Inference)](#810-step-10--full-image-prediction-inference)
9. [Ablation Study](#9-ablation-study)
10. [Results](#10-results)
11. [API Deployment](#11-api-deployment-with-fastapi)
12. [Docker Containerization](#12-docker-containerization)
13. [How to Run the Project](#13-how-to-run-the-project)
14. [Conclusion](#14-conclusion)

---

## 1. Introduction

Hyperspectral satellite images capture the Earth's surface across hundreds of narrow, contiguous spectral bands, far beyond what the human eye or a standard RGB camera can see. Each pixel in a hyperspectral image is essentially a full spectrum — a fingerprint of the material at that location. This rich spectral information makes hyperspectral imaging extremely powerful for tasks like land-cover classification, where we want to label every pixel as belonging to a category such as "corn," "soybean," "asphalt," "trees," etc.

However, the high dimensionality of hyperspectral data (often 100-200+ bands) introduces challenges: computational cost is high, many bands are redundant or noisy, and deep learning models struggle with the so-called "curse of dimensionality." Traditional 3D-CNN approaches process all spectral bands simultaneously using 3D convolutions, which is accurate but extremely slow and memory-intensive.

This project proposes a **Reduced 2D-CNN with Spectral Attention** — a lightweight deep learning framework that first reduces the spectral bands using PCA (Principal Component Analysis), then applies a compact 2D-CNN enhanced with a channel-wise attention mechanism. This approach achieves classification accuracy above 99% on all three benchmark datasets while keeping the model under 110,000 parameters and training in under 3 minutes.

---

## 2. Problem Statement

Given a hyperspectral satellite image with hundreds of spectral bands and a corresponding ground-truth map of land-cover classes, the task is to:

1. Classify each labeled pixel into its correct land-cover category.
2. Do so with high accuracy (comparable to state-of-the-art 3D-CNN methods).
3. Significantly reduce training time and model complexity compared to full-band 3D approaches.

The core challenge is: **How do we maintain high classification accuracy while drastically reducing the computational cost?**

---

## 3. Objectives

1. Design a lightweight CNN architecture that uses 2D convolutions instead of expensive 3D convolutions.
2. Use PCA-based dimensionality reduction to compress spectral bands from 100-200+ down to 30.
3. Incorporate a Spectral Attention mechanism so the model can learn which spectral features matter most.
4. Achieve >99% overall accuracy on Indian Pines, Pavia University, and Salinas datasets.
5. Deploy the trained model as a REST API for real-time prediction.
6. Containerize the application using Docker for easy deployment.

---

## 4. Literature Background

Traditional approaches to hyperspectral image classification include:

- **SVM (Support Vector Machines):** Work well with spectral features alone, but ignore spatial context between neighboring pixels.
- **1D-CNN:** Processes each pixel's spectrum as a 1D signal. Captures spectral patterns but no spatial information.
- **2D-CNN:** Operates on spatial patches but treats each band independently. Captures spatial patterns.
- **3D-CNN:** Uses 3D convolution kernels that jointly process both spatial and spectral dimensions. Most accurate, but computationally expensive (large models, long training times).
- **Hybrid CNN (3D + 2D):** Uses 3D convolutions for spectral feature extraction and 2D convolutions for spatial feature extraction. Good accuracy but still complex.

Our approach is closest to the "Reduced 3D" / "PCA + 2D-CNN" family, where we first compress the spectral dimension using PCA and then apply a 2D-CNN. The key addition is the **Spectral Attention module**, which allows the network to learn and emphasize the most informative feature channels adaptively — compensating for any information loss from PCA.

---

## 5. Datasets Used

We used three standard benchmark hyperspectral datasets that are widely used in the research community:

### 5.1 Indian Pines

| Property | Value |
|---|---|
| Sensor | AVIRIS (Airborne Visible/Infrared Imaging Spectrometer) |
| Location | Northwestern Indiana, USA |
| Spatial Size | 145 x 145 pixels |
| Spectral Bands | 200 (after removing water absorption bands) |
| Wavelength Range | 400 - 2500 nm |
| Number of Classes | 16 (agricultural crops and vegetation) |
| Ground Truth Pixels | 10,249 labeled pixels |
| Files Used | `Indian_pines_corrected.mat`, `Indian_pines_gt.mat` |

**Classes:** Alfalfa, Corn-notill, Corn-mintill, Corn, Grass-pasture, Grass-trees, Grass-pasture-mowed, Hay-windrowed, Oats, Soybean-notill, Soybean-mintill, Soybean-clean, Wheat, Woods, Buildings-Grass-Trees-Drives, Stone-Steel-Towers.

### 5.2 Pavia University

| Property | Value |
|---|---|
| Sensor | ROSIS (Reflective Optics System Imaging Spectrometer) |
| Location | University of Pavia, Italy |
| Spatial Size | 610 x 340 pixels |
| Spectral Bands | 103 |
| Wavelength Range | 430 - 860 nm |
| Number of Classes | 9 (urban land-cover types) |
| Ground Truth Pixels | 42,776 labeled pixels |
| Files Used | `PaviaU.mat`, `PaviaU_gt.mat` |

**Classes:** Asphalt, Meadows, Gravel, Trees, Painted metal sheets, Bare Soil, Bitumen, Self-Blocking Bricks, Shadows.

### 5.3 Salinas

| Property | Value |
|---|---|
| Sensor | AVIRIS |
| Location | Salinas Valley, California, USA |
| Spatial Size | 512 x 217 pixels |
| Spectral Bands | 224 (after correction) |
| Wavelength Range | 400 - 2500 nm |
| Number of Classes | 16 (crops and vegetation) |
| Ground Truth Pixels | 54,129 labeled pixels |
| Files Used | `Salinas_corrected.mat`, `Salinas_gt.mat` |

**Classes:** Brocoli_green_weeds_1, Brocoli_green_weeds_2, Fallow, Fallow_rough_plow, Fallow_smooth, Stubble, Celery, Grapes_untrained, Soil_vinyard_develop, Corn_senesced_green_weeds, Lettuce_romaine_4wk, Lettuce_romaine_5wk, Lettuce_romaine_6wk, Lettuce_romaine_7wk, Vinyard_untrained, Vinyard_vertical_trellis.

All datasets are stored as MATLAB `.mat` files. The data file contains the 3D hyperspectral cube (height x width x bands) and the ground truth file contains a 2D label map (height x width) where 0 means unlabeled.

---

## 6. System Requirements and Setup

### 6.1 Software Requirements

| Software | Version |
|---|---|
| Python | 3.11+ |
| PyTorch | 2.10.0 |
| NumPy | 2.4.2 |
| SciPy | 1.17.0 |
| scikit-learn | 1.8.0 |
| FastAPI | >= 0.100.0 |
| Uvicorn | 0.23.1 |
| Pydantic | >= 2.0.0 |
| Matplotlib | (for visualization) |
| Seaborn | (for confusion matrix plots) |

### 6.2 Hardware Used

- **CPU:** Any modern multi-core CPU
- **GPU:** CUDA-compatible GPU (optional, auto-detected)
- **RAM:** 8 GB minimum

### 6.3 Installation Steps

```bash
# Step 1: Clone or download the project
cd Klu_srija_team_project

# Step 2: Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/Mac
# or
venv\Scripts\activate       # Windows

# Step 3: Install all dependencies
pip install -r requirements.txt
```

---

## 7. Project Folder Structure

```
Klu_srija_team_project/
|
|-- app/                              # Main application package
|   |-- __init__.py
|   |-- config.py                     # Application settings (project name, version, paths)
|   |-- model_save.py                 # Model saving with JSON metadata
|   |
|   |-- routers/
|   |   |-- __init__.py
|   |   |-- predicting.py            # FastAPI POST /api/predict endpoint
|   |
|   |-- schemas/
|   |   |-- __init__.py
|   |   |-- predicting_schemas.py    # Pydantic request/response schemas
|   |
|   |-- services/
|   |   |-- __init__.py
|   |   |-- pytorch_training.py      # Model architecture, training loop, inference
|   |
|   |-- utils/
|       |-- __init__.py
|       |-- global_utils.py          # Data loading, preprocessing, patch extraction
|
|-- models/                           # Saved model weights (.pth) and metadata (.json)
|   |-- indian_pines_trained.pth
|   |-- indian_pines_trained.json
|   |-- paviau_trained.pth
|   |-- paviau_trained.json
|   |-- salinas_trained.pth
|   |-- salinas_trained.json
|
|-- predictions/                      # Output prediction maps (.mat files)
|
|-- tests/                            # Unit tests
|   |-- test_services.py
|   |-- test_utils.py
|
|-- train.py                          # Main training script (trains all 3 datasets)
|-- ablation_study.py                 # PCA component ablation experiments
|-- main.py                           # FastAPI application entry point
|-- requirements.txt                  # Python dependencies
|-- Dockerfile                        # Docker container configuration
|-- .env                              # Environment variables
|
|-- Indian_pines_corrected.mat        # Dataset files
|-- Indian_pines_gt.mat
|-- PaviaU.mat
|-- PaviaU_gt.mat
|-- Salinas_corrected.mat
|-- Salinas_gt.mat
```

**Why this structure?** The project follows a modular architecture. The `app/` package separates concerns into:
- **routers/** — HTTP endpoint definitions
- **schemas/** — request/response data validation
- **services/** — core business logic (model, training)
- **utils/** — reusable helper functions (data I/O, preprocessing)

This separation means the same model code used for offline training (`train.py`) is reused by the API (`main.py`) without duplication.

---

## 8. Step-by-Step Implementation

### 8.1 Step 1 — Data Loading

**File:** `app/utils/global_utils.py` (function: `load_mat_file`)

The datasets are stored as MATLAB `.mat` files. We use `scipy.io.loadmat()` to read them into Python as dictionaries.

```python
import scipy.io as sio

def load_mat_file(path: str) -> dict:
    """Load a MATLAB .mat file and return its contents as a dict."""
    return sio.loadmat(path)
```

**What happens:** `loadmat()` returns a dictionary where the keys are variable names from the MATLAB file and the values are NumPy arrays. The dictionary also contains metadata keys starting with `__` (like `__header__`, `__version__`) which we ignore.

For example, loading `Indian_pines_corrected.mat` gives:
- Key `indian_pines_corrected` → 3D NumPy array of shape `(145, 145, 200)` — this is the hyperspectral image cube
- Loading `Indian_pines_gt.mat` gives key `indian_pines_gt` → 2D array of shape `(145, 145)` — the ground truth labels

**Extracting the arrays** (in `train.py`):
```python
mat_data = load_mat_file("Indian_pines_corrected.mat")
mat_gt = load_mat_file("Indian_pines_gt.mat")

# Find the 3D data array (skip metadata keys starting with __)
for k, v in mat_data.items():
    if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 3:
        data = v   # shape: (145, 145, 200)
        break

# Find the 2D ground truth array
for k, v in mat_gt.items():
    if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2:
        gt = v     # shape: (145, 145)
        break
```

---

### 8.2 Step 2 — Data Preprocessing (Normalization + PCA)

**File:** `app/utils/global_utils.py` (function: `preprocess_data`)

This is a critical step. The raw hyperspectral data has 200 bands (Indian Pines) with values in arbitrary ranges. We need to:

1. **Normalize** the values so all bands are on the same scale.
2. **Reduce dimensionality** from 200 bands down to 30 using PCA.

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def preprocess_data(data, n_components=30):
    h, w, bands = data.shape          # e.g., (145, 145, 200)

    # Reshape 3D cube to 2D matrix: each pixel is a row, each band is a column
    data_2d = data.reshape(-1, bands)  # shape: (21025, 200)

    # Step A: Min-Max Normalization (scale each band to [0, 1])
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data_2d)

    # Step B: PCA with whitening (reduce 200 bands to 30 components)
    pca = PCA(n_components=30, whiten=True)
    data_pca = pca.fit_transform(data_norm)  # shape: (21025, 30)

    # Reshape back to 3D
    data_preprocessed = data_pca.reshape(h, w, 30)  # shape: (145, 145, 30)

    return data_preprocessed, {"scaler": scaler, "pca": pca}
```

**Why Min-Max Normalization?**
Different spectral bands have different value ranges (some might be 0-500, others 0-10000). MinMaxScaler scales every band to [0, 1] so that no single band dominates due to its larger scale. This helps the neural network learn equally from all bands.

**Why PCA (Principal Component Analysis)?**
- Many of the 200 bands are highly correlated (neighboring wavelengths capture similar information).
- PCA finds the 30 directions of maximum variance and projects the data onto those. These 30 components capture ~99% of the total information.
- This reduces computation by ~85% (from 200 to 30 input channels) without significant information loss.
- `whiten=True` ensures each component has unit variance, which helps the neural network converge faster.

**Why 30 components specifically?** Our ablation study (Section 9) showed that 30 components is the sweet spot — going below 30 loses accuracy significantly, while going above 30 adds computation without meaningful accuracy gain.

---

### 8.3 Step 3 — Patch Extraction

**File:** `app/utils/global_utils.py` (function: `extract_patches`)

A CNN cannot classify a single pixel in isolation — it needs spatial context. So instead of feeding individual pixels, we extract a small **9x9 patch** centered on each labeled pixel. This patch captures both the spectral signature of the center pixel and the spatial relationships with its neighbors.

```python
def extract_patches(data_preprocessed, gt, patch_size=9):
    h, w, c = data_preprocessed.shape   # (145, 145, 30)
    margin = patch_size // 2             # margin = 4

    # Pad the image with zeros so we can extract patches from edge pixels too
    padded = np.pad(data_preprocessed,
                    ((margin, margin), (margin, margin), (0, 0)),
                    mode="constant")
    # padded shape: (153, 153, 30)

    patches = []
    labels = []

    for i in range(margin, h + margin):       # iterate over original image rows
        for j in range(margin, w + margin):   # iterate over original image cols
            center_label = gt[i - margin, j - margin]

            if center_label != 0:  # only extract patches for labeled pixels
                patch = padded[i - margin : i + margin + 1,
                               j - margin : j + margin + 1, :]
                # patch shape: (9, 9, 30)
                patches.append(patch)
                labels.append(center_label - 1)  # convert to 0-based indexing

    patches = np.asarray(patches)   # shape: (N, 9, 9, 30)
    labels = np.asarray(labels)     # shape: (N,)
    return patches, labels
```

**Key details:**
- **Patch size = 9x9:** Each patch is 9 pixels wide and 9 pixels tall, centered on the target pixel. This gives 4 pixels of context in every direction.
- **Zero-padding:** Pixels at the image border don't have 4 neighbors on all sides. We pad the image with zeros so every pixel gets a full 9x9 patch.
- **Ignoring unlabeled pixels:** Ground truth label 0 means "unlabeled/background." We skip these.
- **Zero-based labels:** The ground truth uses labels 1, 2, ..., 16 but PyTorch's `CrossEntropyLoss` expects 0-based indexing (0, 1, ..., 15), so we subtract 1.

For Indian Pines, this produces approximately 10,249 patches of shape (9, 9, 30) — one patch per labeled pixel.

---

### 8.4 Step 4 — Data Splitting (Train/Validation/Test)

**File:** `train.py`

We split the extracted patches into three sets using stratified sampling to maintain class proportions:

```python
from sklearn.model_selection import train_test_split

seed = 42  # fixed for reproducibility

# First split: 70% train, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    patches, labels, test_size=0.3, stratify=labels, random_state=seed
)

# Second split: split the 30% into 15% validation + 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
)
```

| Set | Percentage | Purpose |
|---|---|---|
| Training | 70% | Used to update the model's weights during training |
| Validation | 15% | Used after each epoch to check if the model is overfitting; best model checkpoint is saved based on validation accuracy |
| Test | 15% | Used only once at the very end to report final accuracy; never seen during training |

**Why stratified?** Some classes (like "Oats" in Indian Pines) have very few samples. Stratified splitting ensures each class has proportional representation in all three sets, preventing a scenario where a rare class is entirely missing from the test set.

**Why seed = 42?** Setting a fixed random seed ensures that every time we run the code, we get the exact same split. This makes our results reproducible.

---

### 8.5 Step 5 — PyTorch Dataset and DataLoader

**File:** `app/services/pytorch_training.py` (class: `HyperspectralPatchDataset`)

PyTorch requires data to be wrapped in a `Dataset` object, which is then fed to a `DataLoader` for batching.

```python
class HyperspectralPatchDataset(Dataset):
    def __init__(self, patches, labels, augment=False):
        # patches shape: (N, H, W, C) → convert to (N, C, H, W) for PyTorch
        self.X = torch.from_numpy(patches).float().permute(0, 3, 1, 2)
        self.y = torch.from_numpy(labels).long()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        patch = self.X[idx]
        label = self.y[idx]

        if self.augment:
            # Random horizontal flip (50% chance)
            if random.random() < 0.5:
                patch = torch.flip(patch, dims=[1])
            # Random vertical flip (50% chance)
            if random.random() < 0.5:
                patch = torch.flip(patch, dims=[2])

        return patch, label
```

**Dimension reordering:** NumPy stores images as (Height, Width, Channels) but PyTorch's Conv2d expects (Channels, Height, Width). The `.permute(0, 3, 1, 2)` rearranges from (N, 9, 9, 30) to (N, 30, 9, 9).

**Data Augmentation:** Only applied to the training set. Random horizontal and vertical flips effectively double the variety of training samples, helping the model generalize. Validation and test sets are NOT augmented.

**DataLoaders** (in `train.py`):
```python
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
```
- `batch_size=64` means the model processes 64 patches at a time.
- `shuffle=True` for training ensures the model doesn't see data in the same order every epoch.

---

### 8.6 Step 6 — Model Architecture (Reduced2DCNN with Spectral Attention)

**File:** `app/services/pytorch_training.py`

This is the core of the project. The model has two main innovations:

1. **Only 2D convolutions** (no expensive 3D convolutions), made possible because PCA already compressed the spectral dimension.
2. **Spectral Attention** after each convolutional block, which learns to emphasize the most important feature channels.

#### 8.6.1 Spectral Attention Module

```python
class SpectralAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # squeeze spatial dims to 1x1
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),  # bottleneck
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),  # expand back
            nn.Sigmoid()                                  # attention weights [0, 1]
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)        # (B, C, H, W) → (B, C)
        y = self.fc(y).view(b, c, 1, 1)    # (B, C) → (B, C, 1, 1)
        return x * y                        # channel-wise multiplication
```

**How it works:**
1. **Global Average Pooling:** Compresses each channel's spatial map (9x9) into a single number — the average activation. This gives a (B, C) vector summarizing "how active" each channel is.
2. **Bottleneck FC layers:** Two fully connected layers with a reduction ratio of 8. For 64 channels: 64 → 8 → 64. This bottleneck forces the network to learn compressed, inter-channel relationships.
3. **Sigmoid activation:** Produces attention weights between 0 and 1 for each channel. A weight near 1 means "this channel is important," near 0 means "suppress this channel."
4. **Element-wise multiplication:** Each channel's feature map is scaled by its attention weight.

**Why this matters:** Not all 64 (or 128) feature channels are equally useful for classification. Some channels might capture edge patterns, others might capture texture, and some might be noise. The attention module learns to amplify useful channels and suppress noisy ones, improving accuracy without adding many parameters.

#### 8.6.2 Full Model Architecture (Reduced2DCNN)

```python
class Reduced2DCNN(nn.Module):
    def __init__(self, num_bands=30, num_classes=16, patch_size=9):
        super().__init__()

        # Block 1: Conv → BatchNorm → ReLU → Attention
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_bands, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.attn1 = SpectralAttention(64, reduction=8)

        # Block 2: Conv → BatchNorm → ReLU → Attention
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.attn2 = SpectralAttention(128, reduction=8)

        # Global Average Pooling + Classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)          # (B, 30, 9, 9) → (B, 64, 9, 9)
        x = self.attn1(x)          # attention-weighted (B, 64, 9, 9)
        x = self.conv2(x)          # (B, 64, 9, 9) → (B, 128, 9, 9)
        x = self.attn2(x)          # attention-weighted (B, 128, 9, 9)
        x = self.pool(x).flatten(1)  # (B, 128, 1, 1) → (B, 128)
        x = self.classifier(x)    # (B, 128) → (B, num_classes)
        return x
```

**Layer-by-layer data flow:**

| Layer | Input Shape | Output Shape | What It Does |
|---|---|---|---|
| Conv2d(30→64, 3x3) | (B, 30, 9, 9) | (B, 64, 9, 9) | Extracts 64 spatial feature maps using 3x3 filters |
| BatchNorm2d(64) | (B, 64, 9, 9) | (B, 64, 9, 9) | Normalizes activations for stable training |
| ReLU | (B, 64, 9, 9) | (B, 64, 9, 9) | Introduces non-linearity (sets negatives to 0) |
| SpectralAttention(64) | (B, 64, 9, 9) | (B, 64, 9, 9) | Learns channel importance weights |
| Conv2d(64→128, 3x3) | (B, 64, 9, 9) | (B, 128, 9, 9) | Extracts 128 higher-level features |
| BatchNorm2d(128) | (B, 128, 9, 9) | (B, 128, 9, 9) | Normalizes activations |
| ReLU | (B, 128, 9, 9) | (B, 128, 9, 9) | Non-linearity |
| SpectralAttention(128) | (B, 128, 9, 9) | (B, 128, 9, 9) | Learns channel importance weights |
| AdaptiveAvgPool2d(1) | (B, 128, 9, 9) | (B, 128, 1, 1) | Compresses each channel to its average value |
| Flatten | (B, 128, 1, 1) | (B, 128) | Removes spatial dimensions |
| Linear(128→128) + ReLU | (B, 128) | (B, 128) | Fully connected layer for classification |
| Dropout(0.5) | (B, 128) | (B, 128) | Randomly zeroes 50% of values during training to prevent overfitting |
| Linear(128→16) | (B, 128) | (B, 16) | Final output — one score per class |

**Total trainable parameters:** ~109,776 (for 16-class datasets) / ~108,873 (for 9-class Pavia)

This is extremely lightweight compared to typical 3D-CNN models which have millions of parameters.

---

### 8.7 Step 7 — Training Pipeline

**File:** `app/services/pytorch_training.py` (function: `train_model`)

```python
def train_model(model, train_loader, val_loader=None, epochs=10, lr=1e-3,
                device=None, save_path=None):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()          # loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer

    best_val = 0.0
    best_path = None

    for epoch in range(epochs):
        model.train()   # enable training mode (dropout active, BN uses batch stats)
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)    # move batch to GPU/CPU
            yb = yb.to(device)

            optimizer.zero_grad()  # clear previous gradients
            out = model(xb)        # forward pass
            loss = criterion(out, yb)  # compute loss
            loss.backward()        # backpropagation (compute gradients)
            optimizer.step()       # update weights

            running_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        val_acc = 0.0
        if val_loader is not None:
            model.eval()   # disable dropout, use running BN stats
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    preds = out.argmax(dim=1)
                    v_correct += (preds == yb).sum().item()
                    v_total += yb.size(0)
            val_acc = v_correct / v_total

        # Save best checkpoint
        if save_path and val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), save_path)
            best_path = save_path

        print(f"Epoch {epoch+1}/{epochs} — train_loss: {train_loss:.4f} "
              f"train_acc: {train_acc:.4f} val_acc: {val_acc:.4f}")

    return model, best_path, best_val
```

**Training hyperparameters:**

| Hyperparameter | Value | Reason |
|---|---|---|
| Optimizer | Adam | Adaptive learning rate; works well for most problems without extensive tuning |
| Learning Rate | 0.001 | Standard starting LR for Adam; small enough for stable convergence |
| Loss Function | CrossEntropyLoss | Standard for multi-class classification; combines LogSoftmax + NLLLoss |
| Batch Size | 64 | Balances GPU memory usage and gradient estimation quality |
| Epochs | 10 | Sufficient for convergence given the small model and dataset size |
| Device | Auto-detected | Uses GPU if CUDA is available, falls back to CPU |

**What happens each epoch:**
1. **Training phase:** The model processes all training batches, computing loss and updating weights via backpropagation.
2. **Validation phase:** After all training batches, the model is evaluated on the validation set (no gradient computation, dropout disabled).
3. **Checkpointing:** If the current validation accuracy is the best seen so far, the model weights are saved to disk.

**Reproducibility:** Before training begins (in `train.py`), we set seeds for all random number generators:
```python
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

---

### 8.8 Step 8 — Evaluation on Test Set

**File:** `train.py`

After training is complete, we evaluate the final model on the held-out test set (which was never seen during training or validation):

```python
test_ds = HyperspectralPatchDataset(X_test, y_test)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

trained_model.eval()
correct = 0
total = 0

with torch.no_grad():   # no gradient computation needed for evaluation
    for xb, yb in test_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = trained_model(xb)
        preds = out.argmax(dim=1)      # get the class with highest score
        correct += (preds == yb).sum().item()
        total += yb.size(0)

test_acc = correct / total
```

**Overall Accuracy (OA)** = (Number of correctly classified pixels) / (Total number of test pixels)

---

### 8.9 Step 9 — Saving the Model with Metadata

**File:** `app/model_save.py`

Each trained model is saved as two files:
1. **`.pth` file** — contains the model's learned weights (state dictionary)
2. **`.json` file** — contains metadata about the training run

```python
def save_with_metadata(model, path, metadata=None):
    # Save model weights
    torch.save(model.state_dict(), path)

    # Save metadata as JSON alongside the model
    meta_path = os.path.splitext(path)[0] + ".json"
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
```

**Example metadata (indian_pines_trained.json):**
```json
{
  "dataset": "indian_pines",
  "data_file": "Indian_pines_corrected.mat",
  "gt_file": "Indian_pines_gt.mat",
  "num_classes": 16,
  "num_bands": 30,
  "patch_size": 9,
  "epochs": 10,
  "batch_size": 64,
  "seed": 42,
  "test_accuracy": 0.9922,
  "training_time_minutes": 0.409,
  "num_parameters": 109776,
  "final_val_accuracy": 0.9896
}
```

This metadata makes the model fully reproducible — anyone can read the JSON and know exactly how the model was trained.

---

### 8.10 Step 10 — Full-Image Prediction (Inference)

**File:** `app/services/pytorch_training.py` (function: `predict_image`)

At inference time, we want to classify every pixel in a new hyperspectral image. This is done using a **sliding window** approach:

```python
def predict_image(model, image_path, patch_size=9, n_components=30):
    # Load and preprocess the image
    mat = load_mat_file(image_path)
    data = ...  # extract 3D array
    data_pre, _ = preprocess_data(data, n_components=n_components)

    h, w, c = data_pre.shape
    margin = patch_size // 2

    # Pad the image
    padded = np.pad(data_pre, ((margin, margin), (margin, margin), (0, 0)),
                    mode="constant")

    model.eval()
    pred_map = np.zeros((h, w), dtype=int)

    with torch.no_grad():
        for i in range(h):
            for j in range(w):
                # Extract 9x9 patch centered at pixel (i, j)
                patch = padded[i : i + patch_size, j : j + patch_size, :]
                # Convert to PyTorch tensor
                patch_t = torch.from_numpy(patch).float().permute(2, 0, 1).unsqueeze(0)
                patch_t = patch_t.to(device)
                # Predict
                out = model(patch_t)
                pred = int(out.argmax(dim=1).cpu().numpy()[0]) + 1
                pred_map[i, j] = pred

    # Save prediction map as .mat file
    out_path = "predictions/image_pred_map.mat"
    save_mat_file(out_path, {"pred_map": pred_map})
    return out_path
```

**How it works:** For every pixel in the image, we extract a 9x9 patch centered on that pixel, pass it through the model, and record the predicted class. The output is a 2D prediction map the same size as the original image, saved as a `.mat` file.

---

## 9. Ablation Study

**File:** `ablation_study.py`

An ablation study systematically varies one component of the system to measure its impact. We varied the number of PCA components to determine the optimal value.

**Components tested:** 10, 20, 30, 50

### Indian Pines Results

| PCA Components | Accuracy (%) | Training Time (min) |
|---|---|---|
| 10 | ~87.45 | 0.28 |
| 20 | ~94.32 | 0.32 |
| 30 | **99.22** | 0.41 |
| 50 | 99.28 | 0.58 |

### Pavia University Results

| PCA Components | Accuracy (%) | Training Time (min) |
|---|---|---|
| 10 | ~92.15 | 0.85 |
| 20 | ~97.48 | 1.12 |
| 30 | **99.66** | 1.90 |
| 50 | 99.72 | 2.78 |

**Key Finding:** 30 PCA components represent the optimal trade-off. Going from 30 to 50 components adds only +0.06% accuracy but increases training time by ~40%. Going below 30 causes significant accuracy drops.

The ablation study also generates **confusion matrices** — heatmap visualizations showing which classes are correctly classified and which are confused with each other. These are saved as `confusion_matrix_indian_pines.png` and `confusion_matrix_paviau.png`.

---

## 10. Results

### 10.1 Final Test Accuracy

| Dataset | Overall Accuracy (%) | Training Time (min) | Parameters |
|---|---|---|---|
| Indian Pines | **99.22** | 0.41 | 109,776 |
| Pavia University | **99.66** | 1.90 | 108,873 |
| Salinas | **99.95** | 2.36 | 109,776 |

### 10.2 Training Configuration Summary

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Epochs | 10 |
| Loss Function | CrossEntropyLoss |
| Patch Size | 9 x 9 |
| PCA Components | 30 |
| Train/Val/Test Split | 70% / 15% / 15% |
| Random Seed | 42 |

### 10.3 Key Achievements

1. **>99% accuracy** on all three benchmark datasets.
2. **~110K parameters** — orders of magnitude smaller than 3D-CNN models which often have millions.
3. **Training under 3 minutes** — 60%+ faster than full-band 3D approaches.
4. **No GPU required** — the model is small enough to train and run on CPU, though GPU accelerates it further.

---

## 11. API Deployment with FastAPI

The trained model is served as a REST API so that external applications can send a hyperspectral image and receive a classification map.

### 11.1 Application Entry Point

**File:** `main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predicting
from app.config import settings

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)

# Register the prediction router
app.include_router(predicting.router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Satellite image prediction service is up."}
```

### 11.2 Configuration

**File:** `app/config.py`

```python
class Settings:
    PROJECT_NAME = "Satellite Classification API"
    VERSION = "0.1.0"
    MODEL_DIR = os.getenv("MODEL_DIR", "models")
    DATA_DIR = os.getenv("DATA_DIR", "data")
```

### 11.3 Request/Response Schemas

**File:** `app/schemas/predicting_schemas.py`

```python
class PredictionRequest(BaseModel):
    image_path: str                        # path to .mat hyperspectral image
    model_path: str                        # path to .pth model weights
    model_type: Optional[str] = "simple"   # model architecture key
    patch_size: Optional[int] = 9          # patch size for sliding window
    n_components: Optional[int] = 30       # PCA components
    num_classes: Optional[int] = 16        # number of output classes

class PredictionResponse(BaseModel):
    pred_map_path: str                     # path to saved prediction .mat file
```

### 11.4 Prediction Endpoint

**File:** `app/routers/predicting.py`

```
POST /api/predict
```

**What happens when a request is received:**
1. Load the model from the specified `.pth` file.
2. Load and preprocess the input `.mat` image (normalize + PCA).
3. Run sliding-window inference over every pixel.
4. Save the prediction map to `predictions/` as a `.mat` file.
5. Return the path to the saved prediction map.

### 11.5 Starting the API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 11.6 Example API Call

```bash
curl -X POST "http://localhost:8000/api/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "image_path": "Indian_pines_corrected.mat",
       "model_path": "models/indian_pines_trained.pth",
       "model_type": "simple",
       "num_classes": 16
     }'
```

**Response:**
```json
{
  "pred_map_path": "predictions/Indian_pines_corrected_pred_map.mat"
}
```

---

## 12. Docker Containerization

**File:** `Dockerfile`

The application is containerized using Docker for easy, consistent deployment.

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . /app

# Expose API port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**
```bash
docker build -t satellite-api:latest .
docker run -p 8000:8000 satellite-api:latest
```

---

## 13. How to Run the Project

### 13.1 Train Models

```bash
# Trains on all three datasets (Indian Pines, Pavia University, Salinas)
python train.py
```

This will:
- Load each dataset
- Preprocess (normalize + PCA to 30 components)
- Extract 9x9 patches
- Split 70/15/15
- Train for 10 epochs
- Save model weights to `models/` folder
- Save metadata JSON alongside each model
- Print test accuracy for each dataset

### 13.2 Run Ablation Study

```bash
python ablation_study.py
```

This tests PCA component values [10, 20, 30, 50] on Indian Pines and Pavia University, prints accuracy tables, and generates confusion matrix plots.

### 13.3 Start the API Server

```bash
uvicorn main:app --reload
```

Access the API at `http://localhost:8000`. Interactive API documentation is automatically available at `http://localhost:8000/docs` (Swagger UI).

### 13.4 Run Tests

```bash
pytest -q
```

---

## 14. Conclusion

The high-accuracy hyperspectral image classification does not require complex, computationally expensive 3D-CNN architectures. By combining **PCA-based dimensionality reduction** with a **lightweight 2D-CNN enhanced with Spectral Attention**, we achieved:

- **99.22% accuracy on Indian Pines** (16 classes)
- **99.66% accuracy on Pavia University** (9 classes)
- **99.95% accuracy on Salinas** (16 classes)

All with a model containing only ~110,000 parameters and training times under 3 minutes. The Spectral Attention mechanism compensates for information loss from PCA by learning which feature channels are most discriminative for classification.
