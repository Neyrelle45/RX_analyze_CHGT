import cv2
import numpy as np
import torch


# =====================================================
# UTIL — resize to multiple of 8 (UNet safe)
# =====================================================
def resize_to_multiple(img, div=8):
    h, w = img.shape[:2]
    new_h = (h // div) * div
    new_w = (w // div) * div
    return cv2.resize(img, (new_w, new_h))


# =====================================================
# PREPROCESS RX
# =====================================================
def preprocess_rx(image, contrast=1.5, clahe_clip=2.2, gamma=1.1):
    # --- grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # --- global contrast
    gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=0)

    # --- CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)

    # --- gamma correction
    inv = 1.0 / gamma
    table = np.array(
        [(i / 255.0) ** inv * 255 for i in range(256)],
        dtype="uint8"
    )
    gray = cv2.LUT(gray, table)

    # --- normalize
    gray = gray.astype(np.float32) / 255.0

    # --- CRITICAL FIX: UNet-safe size
    gray = resize_to_multiple(gray, div=8)

    # --- to tensor (N,C,H,W)
    tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)

    return tensor, gray
