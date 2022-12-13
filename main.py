import os
import cv2
import numpy as np
import torch
import time

from dct import DCTransform, DCTransformTroch


save_path = 'results/'

# prepare a mask
tgt_mask_1 = cv2.imread('tgt_mask_1.png', 0)
tgt_mask_2 = cv2.imread('tgt_mask_2.png', 0)
tgt_mask_1 = tgt_mask_1.astype(np.float32)
tgt_mask_2 = tgt_mask_2.astype(np.float32)
vmax, hmax = tgt_mask_1.shape[:2]

# ================ OpenCV DCT deployment ==================
save_path_cv2 = os.path.join(save_path, 'opencv')
os.makedirs(save_path_cv2, exist_ok=True) 
# OpenCV DCT encode
cv2_coeffs_1 = cv2.dct(tgt_mask_1)
cv2_coeffs_2 = cv2.dct(tgt_mask_2)
print("==============================================")
print("-- DCT coeffs-1 shape: {}".format(cv2_coeffs_1.shape))
print("-- DCT coeffs-2 shape: {}".format(cv2_coeffs_2.shape))
cv2.imwrite(save_path_cv2 + '/cv2_coeffs_1.png', cv2_coeffs_1)
cv2.imwrite(save_path_cv2 + '/cv2_coeffs_2.png', cv2_coeffs_2)

# OpenCV DCT decodde
recover_mask_1 = cv2.idct(cv2_coeffs_1)
max_v = np.max(recover_mask_1)
min_v = np.min(recover_mask_1)
recover_mask_1 = np.where(recover_mask_1>(max_v+min_v) / 2., 255, 0)
cv2.imwrite(save_path_cv2 + '/cv2_recover_1.png', recover_mask_1)
recover_mask_2 = cv2.idct(cv2_coeffs_2)
max_v = np.max(recover_mask_2)
min_v = np.min(recover_mask_2)
recover_mask_2 = np.where(recover_mask_2>(max_v+min_v) / 2., 255, 0)
cv2.imwrite(save_path_cv2 + '/cv2_recover_2.png', recover_mask_2)

# ================ My DCT deployment with Numpy ==================
save_path_my_np = os.path.join(save_path, 'numpy')
os.makedirs(save_path_my_np, exist_ok=True) 
tgt_masks = np.stack([tgt_mask_1, tgt_mask_2]) # [B, N, N]
# my DCT encode
mydct_np = DCTransform(vmax=vmax, hmax=hmax)
my_coeffs = mydct_np.dct(tgt_masks)
cv2.imwrite(save_path_my_np + '/my_coeffs_1.png', my_coeffs[0])
cv2.imwrite(save_path_my_np + '/my_coeffs_2.png', my_coeffs[1])

# my DCT decode
my_recover_masks = mydct_np.idct(my_coeffs)
max_v = np.max(my_recover_masks[0])
min_v = np.min(my_recover_masks[0])
my_recover_mask_1 = np.where(my_recover_masks[0]>(max_v+min_v) / 2., 255, 0)
cv2.imwrite(save_path_my_np + '/my_recover_1.png', my_recover_mask_1)
max_v = np.max(my_recover_masks[1])
min_v = np.min(my_recover_masks[1])
my_recover_mask_1 = np.where(my_recover_masks[1]>(max_v+min_v) / 2., 255, 0)
cv2.imwrite(save_path_my_np + '/my_recover_2.png', my_recover_mask_1)

# ================ My DCT deployment with PyTorch ==================
save_path_my_torch = os.path.join(save_path, 'torch')
os.makedirs(save_path_my_torch, exist_ok=True) 
tgt_masks = np.stack([tgt_mask_1, tgt_mask_2]) # [B, N, N]
tgt_masks = torch.from_numpy(tgt_masks)
# my DCT encode
mydct_torch = DCTransformTroch(vmax=vmax, hmax=hmax, device=torch.device('cpu'))
my_coeffs = mydct_torch.dct(tgt_masks)
cv2.imwrite(save_path_my_torch + '/my_coeffs_1.png', my_coeffs[0].numpy())
cv2.imwrite(save_path_my_torch + '/my_coeffs_2.png', my_coeffs[1].numpy())

# my DCT decode
my_recover_masks = mydct_torch.idct(my_coeffs)
max_v = np.max(my_recover_masks[0].numpy())
min_v = np.min(my_recover_masks[0].numpy())
my_recover_mask_1 = np.where(my_recover_masks[0].numpy()>(max_v+min_v) / 2., 255, 0)
cv2.imwrite(save_path_my_torch + '/my_recover_1.png', my_recover_mask_1)
max_v = np.max(my_recover_masks[1].numpy())
min_v = np.min(my_recover_masks[1].numpy())
my_recover_mask_1 = np.where(my_recover_masks[1].numpy()>(max_v+min_v) / 2., 255, 0)
cv2.imwrite(save_path_my_torch + '/my_recover_2.png', my_recover_mask_1)

# =================== test encode time ==========================
t0 = time.time()

for i in range(3000):
    cv2_coeffs_1 = cv2.dct(tgt_mask_1)
print("OpenCV encode time: {} ms.".format((time.time() - t0)*1000))

tgt_masks = np.stack([tgt_mask_1]*3000)
t0 = time.time()
my_coeffs = mydct_np.dct(tgt_masks)
print("Numpy encode time: {} ms.".format((time.time() - t0)*1000))

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
mydct_torch = DCTransformTroch(vmax=vmax, hmax=hmax, device=device)
tgt_masks = torch.from_numpy(np.stack([tgt_mask_1]*3000))
t0 = time.time()
my_coeffs = mydct_torch.dct(tgt_masks)
print("Torch encode time: {} ms.".format((time.time() - t0)*1000))

