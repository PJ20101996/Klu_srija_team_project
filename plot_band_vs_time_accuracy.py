import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from app.services.pytorch_training import get_model, HyperspectralPatchDataset, train_model
from app.utils.global_utils import load_mat_file, preprocess_data, extract_patches


def run_experiment(data_file, gt_file, components_list, epochs=10, batch_size=64, model_type='simple'):
    results = []

    mat_data = load_mat_file(data_file)
    mat_gt = load_mat_file(gt_file)

    data = None
    for k, v in mat_data.items():
        if not k.startswith('__') and isinstance(v, np.ndarray) and v.ndim == 3:
            data = v
            break
    if data is None:
        raise ValueError(f'No 3D data found in {data_file}')

    gt = None
    for k, v in mat_gt.items():
        if not k.startswith('__') and isinstance(v, np.ndarray) and v.ndim == 2:
            gt = v
            break
    if gt is None:
        raise ValueError(f'No 2D GT found in {gt_file}')

    for n_bands in components_list:
        print(f'Running {data_file} with n_components={n_bands}')
        data_pre, _meta = preprocess_data(data, n_components=n_bands)
        patches, labels = extract_patches(data_pre, gt, patch_size=9)

        X_train, X_temp, y_train, y_temp = __import__('sklearn.model_selection', fromlist=['train_test_split']).train_test_split(
            patches, labels, test_size=0.3, stratify=labels, random_state=42
        )
        X_val, X_test, y_val, y_test = __import__('sklearn.model_selection', fromlist=['train_test_split']).train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        train_ds = HyperspectralPatchDataset(X_train, y_train)
        val_ds = HyperspectralPatchDataset(X_val, y_val)
        test_ds = HyperspectralPatchDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = get_model(model_type=model_type, num_bands=n_bands, num_classes=len(np.unique(labels)), patch_size=9)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        start = time.time()
        trained_model, best_path, best_val = train_model(
            model,
            train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=1e-3,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            save_path=None,
        )
        training_time = (time.time() - start) / 60

        trained_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                xb = xb.to(device)
                yb = yb.to(device)
                out = trained_model(xb)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        test_acc = correct / total if total > 0 else 0.0

        results.append({
            'n_bands': n_bands,
            'training_time_mins': training_time,
            'test_accuracy': test_acc,
            'val_accuracy': best_val,
            'num_params': num_params,
        })

        print(f' . n_bands={n_bands} time={training_time:.2f}m test_acc={test_acc:.4f} val_acc={best_val:.4f}')

    return results


def plot_results(dataset_name, results, out_file):
    bands = [r['n_bands'] for r in results]
    times = [r['training_time_mins'] for r in results]
    accuracies = [r['test_accuracy'] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(bands, times, marker='o', label='Training Time (minutes)')
    plt.title(f'{dataset_name}: Number of Bands vs Training Time')
    plt.xlabel('Number of PCA Bands')
    plt.ylabel('Training Time (minutes)')
    plt.grid(True)
    plt.legend()
    plt.savefig(out_file)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(bands, accuracies, marker='s', color='tab:green', label='Test Accuracy')
    plt.title(f'{dataset_name}: Number of Bands vs Accuracy')
    plt.xlabel('Number of PCA Bands')
    plt.ylabel('Overall Test Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(out_file.replace('fig', 'fig_acc' if 'pavia' in out_file else 'fig_acc'))
    plt.close()


if __name__ == '__main__':
    # Example set of PCA band values to experiment with
    band_values = [10, 20, 30, 50]

    # Run and plot for Pavia
    pavia_results = run_experiment('PaviaU.mat', 'PaviaU_gt.mat', band_values, epochs=10, batch_size=64)
    with open('pavia_band_experiment.json', 'w') as f:
        json.dump(pavia_results, f, indent=2)
    plot_results('Pavia University', pavia_results, 'fig4_pavia.png')

    # Run and plot for Indian Pines
    indian_results = run_experiment('Indian_pines_corrected.mat', 'Indian_pines_gt.mat', band_values, epochs=10, batch_size=64)
    with open('indian_pines_band_experiment.json', 'w') as f:
        json.dump(indian_results, f, indent=2)
    plot_results('Indian Pines', indian_results, 'fig5_indian_pines.png')
