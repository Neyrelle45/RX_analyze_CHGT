import cv2

def preprocess_rx(img, contrast=1.0, denoise=5):

    h, w = img.shape
    scale = 512 / max(h, w)

    nh, nw = int(h*scale), int(w*scale)

    img = cv2.resize(img, (nw, nh))

    canvas = np.zeros((512,512), dtype=img.dtype)
    canvas[:nh, :nw] = img

    img = cv2.convertScaleAbs(canvas, alpha=contrast)

    if denoise > 0:
        img = cv2.fastNlMeansDenoising(img, None, denoise)

    return img, scale, (nh, nw)
