# import scipy.io as sio
# import numpy as np
# import cv2

# # Load the data
# data = sio.loadmat(r"C:\Users\MIT\vit\Indian_pines_corrected.mat")['indian_pines_corrected']
# gt = sio.loadmat(r"C:\Users\MIT\vit\Indian_pines_gt.mat")['indian_pines_gt']

# # Pick 3 bands to simulate RGB (e.g., bands 29, 19, 9)
# rgb = data[:, :, [29, 19, 9]]
# rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  # normalize to 0-1
# rgb = (rgb * 255).astype(np.uint8)

# # Show a single spectral band (grayscale)
# band50 = data[:, :, 50]
# band50 = ((band50 - band50.min()) / (band50.max() - band50.min()) * 255).astype(np.uint8)

# # Ground truth (label map) - colorize it
# gt_color = (gt / gt.max() * 255).astype(np.uint8)
# gt_color = cv2.applyColorMap(gt_color, cv2.COLORMAP_JET)

# cv2.imshow("False-Color RGB (bands 29,19,9)", rgb)
# cv2.imshow("Single Band 50 (grayscale)", band50)
# cv2.imshow("Ground Truth Labels", gt_color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# Load the predicted map from the API response
pred_data = sio.loadmat('predictions/Indian_pines_corrected_pred_map.mat')
pred_map = pred_data['pred_map']

print(f'Predicted map shape: {pred_map.shape}')
print(f'Unique classes in prediction: {np.unique(pred_map)}')

# Visualize the predicted classification map
plt.figure(figsize=(10, 8))
plt.imshow(pred_map, cmap='tab20')
plt.colorbar()
plt.title('Predicted Classification Map - Indian Pines')
plt.show()