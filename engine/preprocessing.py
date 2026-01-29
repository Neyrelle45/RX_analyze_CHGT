import cv2
import numpy as np

TARGET_SIZE = 512


def letterbox(img):

    h, w = img.shape
    scale = TARGET_SIZE / max(h, w)

    nh, nw = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (nw, nh))

    canvas = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=img.dtype)
    canvas[:nh, :nw] = resized

    # 🔥 mask pixels valides
    valid_mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=bool)
    valid_mask[:nh, :nw] = True

    return canvas, valid_mask, scale, (nh, nw)


def preprocess_rx(img, contrast=1.0, denoise=5):

img, valid_mask, scale, shape = letterbox(img)
return img, valid_mask, scale, shape

    img = cv2.convertScaleAbs(img, alpha=contrast)

    if denoise > 0:
        img = cv2.fastNlMeansDenoising(img, None, denoise)

    return img, scale, shape
