import os
import os
from typing import Tuple

import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_mat_file(path: str) -> dict:
    """Load a MATLAB `.mat` file and return its contents as a dict."""
    return sio.loadmat(path)


def find_data_and_gt(mat_dict: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Given the loaded .mat dict, find the 3D data cube and 2D ground truth.

    Returns: (data_cube (H,W,B), gt (H,W))
    """
    data = None
    gt = None
    for k, v in mat_dict.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray):
            if v.ndim == 3 and data is None:
                data = v
            elif v.ndim == 2 and gt is None:
                gt = v
    if data is None:
        raise ValueError("No 3D data array found in .mat file")
    if gt is None:
        raise ValueError("No 2D ground-truth array found in .mat file")
    return data, gt


def preprocess_data(data: np.ndarray, n_components: int = 30) -> Tuple[np.ndarray, dict]:
    """Normalize and optionally reduce spectral bands using PCA.

    Returns the preprocessed 3D array (H,W,n_components) and a small metadata dict
    containing scaler and pca objects (if used) so the same transform can be
    applied at inference.
    """
    h, w, bands = data.shape
    data_2d = data.reshape(-1, bands)

    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data_2d)

    if n_components is None or n_components >= bands:
        data_pca = data_norm
        pca = None
    else:
        pca = PCA(n_components=n_components)
        data_pca = pca.fit_transform(data_norm)

    if pca is not None:
        out_bands = n_components
    else:
        out_bands = bands

    data_preprocessed = data_pca.reshape(h, w, out_bands)
    meta = {"scaler": scaler, "pca": pca, "out_bands": out_bands}
    return data_preprocessed, meta


def extract_patches(data_preprocessed: np.ndarray, gt: np.ndarray, patch_size: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """Extract square patches centered on labeled pixels (gt != 0).

    Returns patches shaped (N, H, W, C) and labels (N,).
    """
    h, w, c = data_preprocessed.shape
    margin = patch_size // 2
    padded = np.pad(data_preprocessed, ((margin, margin), (margin, margin), (0, 0)), mode="constant")

    patches = []
    labels = []
    for i in range(margin, h + margin):
        for j in range(margin, w + margin):
            center_label = gt[i - margin, j - margin]
            if center_label != 0:
                patch = padded[i - margin : i + margin + 1, j - margin : j + margin + 1, :]
                patches.append(patch)
                labels.append(center_label - 1)

    patches = np.asarray(patches)
    labels = np.asarray(labels)
    return patches, labels


def save_mat_file(path: str, mapping: dict):
    """Save a dictionary to a MATLAB .mat file."""
    ensure_dir(os.path.dirname(path) or ".")
    sio.savemat(path, mapping)
