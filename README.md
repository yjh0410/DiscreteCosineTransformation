# Discrete Cosine Transformation

We release the codes of 2D DCT with Numpy and PyTorch.

Our deployment can process a batch of square images with `[B, N, N]` shape, rather than process image one by one.

# Examples
* Original mask images:

![image](tgt_mask_1.png)
![image](tgt_mask_2.png)

* transformed by cv2.dct:

![image](./img_files/opencv/cv2_coeffs_1.png)
![image](./img_files/opencv/cv2_coeffs_2.png)

* transformed by our Numpy deployment:

![image](./img_files/numpy/my_coeffs_1.png)
![image](./img_files/numpy/my_coeffs_2.png)

* transformed by our PyTorch deployment:

![image](./img_files/torch/my_coeffs_1.png)
![image](./img_files/torch/my_coeffs_2.png)

* Recover by cv2.idct:

![image](./img_files/opencv/cv2_recover_1.png)
![image](./img_files/opencv/cv2_recover_2.png)

* Recover by our Numpy deployment:

![image](./img_files/numpy/my_recover_1.png)
![image](./img_files/numpy/my_recover_2.png)

* Recover by our PyTorch deployment:

![image](./img_files/torch/my_recover_1.png)
![image](./img_files/torch/my_recover_2.png)
