import cv2
import numpy as np


# ---------------------------------------------------
# GLOBAL TARGET SIZE
# ---------------------------------------------------

TARGET_SIZE = 512


# ---------------------------------------------------
# LETTERBOX (INDUSTRIAL VERSION)
# ---------------------------------------------------

def letterbox(img, target=TARGET_SIZE):
    """
    Resize image WITHOUT distortion.
    Pads remaining area with zeros.

    Returns:
        canvas        -> resized + padded image
        valid_mask    -> True where real pixels exist
        scale         -> resize factor
        shape         -> (new_h, new_w)
    """

    h, w = img.shape

    # compute scale
    scale = target / max(h, w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    # resize
    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA  # best for downscale
    )

    # padded canvas
    canvas = np.zeros((target, target), dtype=img.dtype)

    canvas[:new_h, :new_w] = resized

    # mask of REAL pixels
    valid_mask = np.zeros((target, target), dtype=bool)
    valid_mask[:new_h, :new_w] = True

    return canvas, valid_mask, scale, (new_h, new_w)


# ---------------------------------------------------
# PREPROCESS RX
# ---------------------------------------------------

def preprocess_rx(img, contrast=1.0, denoise=5):
    """
    Industrial preprocessing pipeline.

    Steps:
        - letterbox resize
        - contrast adjustment
        - denoising

    Returns:
        img_net
        valid_mask
        scale
        shape
    """

    # ensure grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # letterbox
    img_net, valid_mask, scale, shape = letterbox(img)

    # contrast
    if contrast != 1.0:
        img_net = cv2.convertScaleAbs(img_net, alpha=contrast)

    # denoise (very effective for RX)
    if denoise > 0:
        img_net = cv2.fastNlMeansDenoising(
            img_net,
            None,
            h=denoise,
            templateWindowSize=7,
            searchWindowSize=21
        )

    return img_net, valid_mask, scale, shape

