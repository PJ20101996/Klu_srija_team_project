import scipy.io as sio
import numpy as np
import cv2

# Load the data
data = sio.loadmat(r"C:\Users\MIT\vit\Indian_pines_corrected.mat")['indian_pines_corrected']
gt = sio.loadmat(r"C:\Users\MIT\vit\Indian_pines_gt.mat")['indian_pines_gt']

# Pick 3 bands to simulate RGB (e.g., bands 29, 19, 9)
rgb = data[:, :, [29, 19, 9]]
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  # normalize to 0-1
rgb = (rgb * 255).astype(np.uint8)

# Show a single spectral band (grayscale)
band50 = data[:, :, 50]
band50 = ((band50 - band50.min()) / (band50.max() - band50.min()) * 255).astype(np.uint8)

# Ground truth (label map) - colorize it
gt_color = (gt / gt.max() * 255).astype(np.uint8)
gt_color = cv2.applyColorMap(gt_color, cv2.COLORMAP_JET)

cv2.imshow("False-Color RGB (bands 29,19,9)", rgb)
cv2.imshow("Single Band 50 (grayscale)", band50)
cv2.imshow("Ground Truth Labels", gt_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
