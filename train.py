"""Example script to train a satellite classification model."""
import os
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from app.services.pytorch_training import (
    train_model,
    save_model,
    get_model,
    HyperspectralPatchDataset,
)
from app.utils.global_utils import (
    load_mat_file,
    find_data_and_gt,
    preprocess_data,
    extract_patches,
)
from app.model_save import save_with_metadata

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(data_file="Indian_pines_corrected.mat", gt_file="Indian_pines_gt.mat", epochs=10, batch_size=64):
    print(f"\n{'='*60}")
    print(f"Training on: {data_file} + {gt_file}")
    print(f"{'='*60}\n")

    # 1. Load and preprocess
    print("Loading .mat files...")
    mat_data = load_mat_file(data_file)
    mat_gt = load_mat_file(gt_file)
    
    # Extract 3D data from data file
    data = None
    for k, v in mat_data.items():
        if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 3:
            data = v
            break
    if data is None:
        raise ValueError(f"No 3D data array found in {data_file}")
    
    # Extract 2D GT from GT file
    gt = None
    for k, v in mat_gt.items():
        if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2:
            gt = v
            break
    if gt is None:
        raise ValueError(f"No 2D ground-truth array found in {gt_file}")
    
    print(f"Data shape: {data.shape}, GT shape: {gt.shape}")

    print("Preprocessing (normalize + PCA)...")
    data_pre, meta = preprocess_data(data, n_components=30)
    print(f"Preprocessed shape: {data_pre.shape}")

    # 2. Extract patches
    print("Extracting patches from labeled pixels...")
    patches, labels = extract_patches(data_pre, gt, patch_size=9)
    print(f"Patches: {patches.shape}, Labels: {labels.shape}")

    # 3. Train/val/test split (stratified)
    print("Stratified split (70/15/15)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        patches, labels, test_size=0.3, stratify=labels, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # 4. Create datasets and dataloaders
    train_ds = HyperspectralPatchDataset(X_train, y_train)
    val_ds = HyperspectralPatchDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 5. Instantiate model
    num_classes = len(np.unique(labels))
    num_bands = data_pre.shape[2]
    print(f"\nInstantiating model (SimpleCNN)...")
    model = get_model(
        model_type="simple",
        num_bands=num_bands,
        num_classes=num_classes,
        patch_size=9,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")

    # 6. Train
    print(f"\nTraining for {epochs} epochs...")
    model_name = os.path.splitext(data_file)[0].lower().replace("_corrected", "")
    save_path = os.path.join("models", f"{model_name}_trained.pth")
    trained_model, best_path = train_model(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=1e-3,
        device=device,
        save_path=save_path,
    )

    # 7. Save with metadata
    if best_path:
        metadata = {
            "dataset": model_name,
            "data_file": data_file,
            "gt_file": gt_file,
            "num_classes": num_classes,
            "num_bands": num_bands,
            "patch_size": 9,
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
        }
        save_with_metadata(trained_model, best_path, metadata)
        print(f"\n✓ Model saved to {best_path}")
        print(f"✓ Metadata saved to {os.path.splitext(best_path)[0]}.json")


if __name__ == "__main__":
    # Train on Indian Pines
    main(
        data_file="Indian_pines_corrected.mat",
        gt_file="Indian_pines_gt.mat",
        epochs=10,
        batch_size=64,
    )

    # Uncomment to try other datasets:
    # main(data_file="Salinas_corrected.mat", gt_file="Salinas_gt.mat", epochs=10, batch_size=64)
    # main(data_file="PaviaU.mat", gt_file="PaviaU_gt.mat", epochs=10, batch_size=64)
