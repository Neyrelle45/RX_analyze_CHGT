import cv2
import numpy as np


# =====================================================
# CONSTANT
# =====================================================

TARGET_SIZE = 512   # doit matcher le training


# =====================================================
# LETTERBOX (NO DISTORTION)
# =====================================================

def letterbox(image, target=TARGET_SIZE):
    """
    Resize image WITHOUT distortion.
    Keeps aspect ratio and pads with black.

    Returns:
        canvas
        valid_mask
        scale
        new_shape
    """

    h, w = image.shape

    scale = target / max(h, w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((target, target), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized

    valid_mask = np.zeros((target, target), dtype=bool)
    valid_mask[:new_h, :new_w] = True

    return canvas, valid_mask, scale, (new_h, new_w)


# =====================================================
# SAFE CONTRAST
# =====================================================

def apply_contrast(img, alpha):
    """
    Linear contrast.
    NON destructive.
    """

    if alpha == 1.0:
        return img

    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


# =====================================================
# SAFE DENOISE
# =====================================================

def apply_denoise(img, strength):
    """
    Gentle denoise.

    Avoid strong blur which kills void edges.
    """

    if strength == 0:
        return img

    return cv2.fastNlMeansDenoising(
        img,
        None,
        h=strength,
        templateWindowSize=7,
        searchWindowSize=21
    )


# =====================================================
# PREPROCESS RX (INDUSTRIAL)
# =====================================================

def preprocess_rx(
    img,
    contrast=1.0,
    denoise=0,
    clahe_clip=0,        # left for future — not applied
    void_boost=0        # left for future inference engine
):
    """
    Industrial-safe preprocessing.

    Steps:
        1. grayscale
        2. letterbox
        3. gentle contrast
        4. gentle denoise
    """

    # ensure grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # letterbox
    img_net, valid_mask, scale, shape = letterbox(img)

    # contrast
    img_net = apply_contrast(img_net, contrast)

    # denoise
    img_net = apply_denoise(img_net, denoise)

    return img_net, valid_mask, scale, shape
