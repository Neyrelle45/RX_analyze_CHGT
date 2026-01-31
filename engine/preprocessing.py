import cv2
import numpy as np
import torch

def preprocess_rx(
    image_rgb,
    contrast=1.6,
    clahe_clip=2.2,
    gamma=1.1,
    target_size=512
):
    # --- grayscale ---
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # --- resize (model friendly) ---
    gray = cv2.resize(gray, (target_size, target_size))

    # --- normalize ---
    gray = gray.astype(np.float32) / 255.0

    # --- global contrast ---
    gray = np.clip(gray * contrast, 0, 1)

    # --- CLAHE (local contrast) ---
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply((gray * 255).astype(np.uint8)).astype(np.float32) / 255.0

    # --- gamma correction (micro contrast) ---
    gray = np.power(gray, gamma)

    # --- tensor ---
    tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
    tensor = tensor.float()

    return tensor, gray
