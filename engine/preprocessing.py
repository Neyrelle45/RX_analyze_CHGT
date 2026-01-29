import cv2
import numpy as np

TARGET_SIZE = 512


def preprocess_rx(
    image,
    contrast=1.4,
    clahe_clip=2.5,
    void_boost=2.0,
    gamma=1.15
):
    """
    Industrial RX preprocessing.

    Objectif :
    - augmenter la séparation void / solder
    - éviter le bruit artificiel
    - garder les contours nets
    """

    # -----------------------------
    # GRAYSCALE SAFE
    # -----------------------------

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    gray = gray.astype(np.uint8)

    # -----------------------------
    # GLOBAL CONTRAST
    # -----------------------------

    gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=0)

    # -----------------------------
    # CLAHE (LOCAL CONTRAST)
    # -----------------------------

    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    # -----------------------------
    # VOID BOOST — TOPHAT
    # -----------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    tophat = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        kernel
    )

    gray = cv2.addWeighted(
        gray,
        1.0,
        tophat,
        void_boost,
        0
    )

    # -----------------------------
    # GAMMA (micro contrast)
    # -----------------------------

    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in range(256)
    ]).astype("uint8")

    gray = cv2.LUT(gray, table)

    # -----------------------------
    # NORMALIZE
    # -----------------------------

    img = gray.astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)

    return img, gray

