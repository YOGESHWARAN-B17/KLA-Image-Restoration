import torch
import numpy as np
import cv2
from PIL import Image

print("=" * 50)
print("KLA IMAGE RESTORATION ENVIRONMENT")
print("=" * 50)

print("PyTorch version :", torch.__version__)
print("NumPy version   :", np.__version__)
print("OpenCV version  :", cv2.__version__)

print("CUDA available  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU             :", torch.cuda.get_device_name(0))
else:
    print("GPU             : Not detected")

print("=" * 50)
print("Environment test completed!")
print("=" * 50)