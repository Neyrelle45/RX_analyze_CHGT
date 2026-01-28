import cv2

def preprocess_rx(img, contrast=1.0, denoise=5):
    img = cv2.convertScaleAbs(img, alpha=contrast)
    if denoise > 0:
        img = cv2.fastNlMeansDenoising(img, None, denoise)
    return img
