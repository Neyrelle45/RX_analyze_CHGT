import cv2
import numpy as np
import torch


def preprocess_rx(image, contrast, clahe_clip, gamma):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # global contrast
    gray = cv2.convertScaleAbs(gray, alpha=contrast)

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip,
        tileGridSize=(8,8)
    )

    gray = clahe.apply(gray)

    # gamma
    inv = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv) * 255 for i in range(256)
    ]).astype("uint8")

    gray = cv2.LUT(gray, table)

    tensor = torch.from_numpy(gray).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor, gray

