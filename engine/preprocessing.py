import cv2
import numpy as np

TARGET_SIZE = 512


def letterbox(img, target=TARGET_SIZE):

    h, w = img.shape

    scale = target / max(h, w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((target, target), dtype=img.dtype)
    canvas[:new_h, :new_w] = resized

    valid_mask = np.zeros((target, target), dtype=bool)
    valid_mask[:new_h, :new_w] = True

    return canvas, valid_mask, scale, (new_h, new_w)


def preprocess_rx(img, contrast=1.0, denoise=5):

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_net, valid_mask, scale, shape = letterbox(img)

    if contrast != 1.0:
        img_net = cv2.convertScaleAbs(img_net, alpha=contrast)

    if denoise > 0:
        img_net = cv2.fastNlMeansDenoising(img_net, None, denoise)

    return img_net, valid_mask, scale, shape

