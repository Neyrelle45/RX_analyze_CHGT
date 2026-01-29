import cv2
import numpy as np
import torch


# ---------------------------------------------------------
# RX PREPROCESS — INDUSTRIAL
# ---------------------------------------------------------

def preprocess_rx(
    image,
    contrast=1.4,
    clahe_clip=2.5,
    void_boost=2.0,
    gamma=1.15
):
    """
    Returns:
        tensor -> pour le modèle
        processed -> image numpy affichable
    """

    # -------------------------------------------------
    # GRAYSCALE
    # -------------------------------------------------

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # -------------------------------------------------
    # GLOBAL CONTRAST
    # -------------------------------------------------

    gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=0)

    # -------------------------------------------------
    # CLAHE (micro contrast)
    # -------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip,
        tileGridSize=(8,8)
    )

    gray = clahe.apply(gray)

    # -------------------------------------------------
    # GAMMA — reveals subtle density differences
    # -------------------------------------------------

    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")

    gray = cv2.LUT(gray, table)

    # -------------------------------------------------
    # TOPHAT — VOID enhancer
    # detects darker circular regions
    # -------------------------------------------------

    if void_boost > 0:

        k = int(5 + void_boost * 2)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (k, k)
        )

        tophat = cv2.morphologyEx(
            gray,
            cv2.MORPH_BLACKHAT,
            kernel
        )

        gray = cv2.addWeighted(
            gray,
            1.0,
            tophat,
            0.6,
            0
        )

    # -------------------------------------------------
    # NORMALIZE (VERY IMPORTANT)
    # -------------------------------------------------

    gray = gray.astype(np.float32) / 255.0

    tensor = torch.tensor(gray).unsqueeze(0).unsqueeze(0)

    return tensor, (gray * 255).astype(np.uint8)

