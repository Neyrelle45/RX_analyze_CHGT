import cv2
import numpy as np


# =====================================================
# TARGET SIZE (réseau)
# =====================================================

TARGET_SIZE = 512


# =====================================================
# LETTERBOX (CRITIQUE — NE JAMAIS REMPLACER PAR RESIZE)
# =====================================================

def letterbox(img, target=TARGET_SIZE):
    """
    Resize WITHOUT distortion.
    Pads remaining area.

    Returns:
        canvas
        valid_mask
        scale
        (new_h, new_w)
    """

    h, w = img.shape

    scale = target / max(h, w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((target, target), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized

    valid_mask = np.zeros((target, target), dtype=bool)
    valid_mask[:new_h, :new_w] = True

    return canvas, valid_mask, scale, (new_h, new_w)


# =====================================================
# CLAHE — LOCAL CONTRAST BOOSTER
# (ULTRA efficace en RX)
# =====================================================

def apply_clahe(img, strength):

    if strength <= 0:
        return img

    clip = 2.0 + strength * 4.0

    clahe = cv2.createCLAHE(
        clipLimit=clip,
        tileGridSize=(8, 8)
    )

    return clahe.apply(img)


# =====================================================
# BLACKHAT — VOID BOOSTER
# (détecte zones sombres dans métal)
# =====================================================

def apply_blackhat(img, strength):

    if strength <= 0:
        return img

    kernel_size = int(3 + strength * 4)

    # kernel impair obligatoire
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    blackhat = cv2.morphologyEx(
        img,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    # boost contrôlé
    boosted = cv2.addWeighted(
        img,
        1.0,
        blackhat,
        1.5,
        0
    )

    return boosted


# =====================================================
# NORMALISATION DYNAMIQUE
# (souvent négligée — énorme gain)
# =====================================================

def normalize_dynamic(img):
    """
    Étire les niveaux de gris
    sans amplifier le bruit.
    """

    p2, p98 = np.percentile(img, (2, 98))

    if p98 - p2 < 10:
        return img

    img = np.clip(img, p2, p98)
    img = ((img - p2) / (p98 - p2) * 255).astype(np.uint8)

    return img


# =====================================================
# PREPROCESS RX — VERSION INDUSTRIELLE
# =====================================================

def preprocess_rx(
    img,
    contrast=1.2,
    denoise=4,
    clahe_strength=1.5,
    blackhat_strength=2
):

    # -------------------------------------------------
    # GRAYSCALE SAFE
    # -------------------------------------------------

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------
    # NORMALISATION (FAIBLE MAIS PUISSANTE)
    # -------------------------------------------------

    img = normalize_dynamic(img)

    # -------------------------------------------------
    # CLAHE
    # -------------------------------------------------

    img = apply_clahe(img, clahe_strength)

    # -------------------------------------------------
    # VOID BOOSTER
    # -------------------------------------------------

    img = apply_blackhat(img, blackhat_strength)

    # -------------------------------------------------
    # CONTRASTE GLOBAL (léger seulement)
    # -------------------------------------------------

    if contrast != 1.0:
        img = cv2.convertScaleAbs(img, alpha=contrast)

    # -------------------------------------------------
    # DENOISE RX
    # (garde les blobs)
    # -------------------------------------------------

    if denoise > 0:
        img = cv2.fastNlMeansDenoising(
            img,
            None,
            h=denoise,
            templateWindowSize=7,
            searchWindowSize=21
        )

    # -------------------------------------------------
    # LETTERBOX (TOUJOURS EN DERNIER)
    # -------------------------------------------------

    img_net, valid_mask, scale, shape = letterbox(img)

    return img_net, valid_mask, scale, shape
