import numpy as np
from app.utils.global_utils import preprocess_data, extract_patches


def test_preprocess_and_patches():
    # create synthetic data: HxWxB
    H, W, B = 11, 11, 10
    data = np.random.rand(H, W, B).astype(np.float32)
    gt = np.zeros((H, W), dtype=int)
    # mark a small square as labeled classes 1 and 2
    gt[4:7, 4:7] = 1
    gt[7, 7] = 2

    data_pre, meta = preprocess_data(data, n_components=5)
    assert data_pre.shape == (H, W, 5)
    patches, labels = extract_patches(data_pre, gt, patch_size=3)
    # expect number of patches equals labeled pixels
    assert patches.shape[0] == np.sum(gt != 0)
    assert patches.shape[1:] == (3, 3, 5)
    assert labels.shape[0] == patches.shape[0]
