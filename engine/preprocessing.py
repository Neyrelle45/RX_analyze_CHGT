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

    return canvas, scale, (nh, nw)


def preprocess_rx(img, contrast=1.0, denoise=5):

    img, scale, shape = letterbox(img)

    img = cv2.convertScaleAbs(img, alpha=contrast)

    if denoise > 0:
        img = cv2.fastNlMeansDenoising(img, None, denoise)

    return img, scale, shape
