import os
import numpy as np
import torch
from app.services.pytorch_training import get_model, save_model, load_model
from scipy.io import savemat


def test_model_save_and_load(tmp_path):
    model = get_model(model_type="simple", num_bands=5, num_classes=3, patch_size=3)
    path = tmp_path / "m.pth"
    save_model(model, str(path))
    # saved file exists
    assert path.exists()
    # load as state dict raises if model_type mismatch handled by load_model
    loaded = load_model(str(path), model_type="simple", num_bands=5, num_classes=3, patch_size=3)
    assert isinstance(loaded, torch.nn.Module)


def test_predict_image_smoke(tmp_path):
    # create tiny synthetic image and gt
    H, W, B = 7, 7, 5
    data = np.random.rand(H, W, B).astype(np.float32)
    gt = np.zeros((H, W), dtype=int)
    gt[3, 3] = 1
    mat_path = tmp_path / "img.mat"
    savemat(str(mat_path), {"data": data, "gt": gt})

    model = get_model(model_type="simple", num_bands=B, num_classes=2, patch_size=3)
    # save and load
    model_path = tmp_path / "m2.pth"
    save_model(model, str(model_path))
    loaded = load_model(str(model_path), model_type="simple", num_bands=B, num_classes=2, patch_size=3)

    out = loaded  # ensure loaded model is usable
    # run a single forward pass
    x = torch.randn(1, B, 3, 3)
    y = out(x)
    assert y.shape[0] == 1
