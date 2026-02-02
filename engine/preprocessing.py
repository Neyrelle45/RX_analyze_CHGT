import cv2
import numpy as np
import torch


def preprocess_rx(image, contrast=1.5, clahe_clip=2.2, gamma=1.1):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Global contrast
    gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=0)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Gamma
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    gray = cv2.LUT(gray, table)

    gray = gray.astype(np.float32) / 255.0

    tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)

    return tensor, gray
