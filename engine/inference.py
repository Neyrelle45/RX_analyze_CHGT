import torch
import numpy as np
import cv2


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(model_file):

    model = torch.jit.load(model_file, map_location=DEVICE) \
        if str(model_file).endswith(".pt") \
        else torch.load(model_file, map_location=DEVICE)

    model.eval()
    return model


# =========================================================
# CORE INFERENCE
# =========================================================

def run_inference(model, img_tensor):

    with torch.no_grad():
        logits = model(torch.from_numpy(img_tensor).to(DEVICE))

    logits = logits[0].cpu().numpy()

    solder_logit = logits[1]
    defect_logit = logits[2]

    return solder_logit, defect_logit


# =========================================================
# INDUSTRIAL DECISION ENGINE
# =========================================================

def decision_engine(
    solder_logit,
    defect_logit,
    inspect_mask,
    defect_th=0.4,
    dominance=0.8
):

    # -----------------------------------------------------
    # convert logits → pseudo probabilities
    # (without destructive softmax)
    # -----------------------------------------------------

    solder = 1 / (1 + np.exp(-solder_logit))
    defect = 1 / (1 + np.exp(-defect_logit))

    # -----------------------------------------------------
    # focal decision
    # -----------------------------------------------------

    defect_mask = (
        (defect > defect_th) &
        (defect > solder * dominance) &
        inspect_mask
    )

    solder_mask = (
        (solder > 0.25) &
        ~defect_mask &
        inspect_mask
    )

    # -----------------------------------------------------
    # MORPHO CLEANUP
    # -----------------------------------------------------

    kernel = np.ones((3,3), np.uint8)

    defect_mask = cv2.morphologyEx(
        defect_mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        kernel
    ).astype(bool)

    defect_mask = cv2.morphologyEx(
        defect_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel
    ).astype(bool)

    # remove tiny noise
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        defect_mask.astype(np.uint8),
        connectivity=8
    )

    clean = np.zeros_like(defect_mask)

    for i in range(1, num):

        area = stats[i, cv2.CC_STAT_AREA]

        if area > 20:   # industrial threshold
            clean[labels == i] = True

    defect_mask = clean

    solder_mask = solder_mask & ~defect_mask

    return solder_mask, defect_mask


# =========================================================
# VOID ANALYSIS
# =========================================================

def analyze_voids(defect_mask):

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        defect_mask.astype(np.uint8),
        connectivity=8
    )

    largest_void = 0
    circularity_best = 0

    for i in range(1, num):

        area = stats[i, cv2.CC_STAT_AREA]

        component = (labels == i).astype(np.uint8)

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        cnt = contours[0]

        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter**2)

        if area > largest_void:
            largest_void = area
            circularity_best = circularity

    return largest_void, circularity_best


# =========================================================
# METRICS
# =========================================================

def compute_metrics(solder_mask, defect_mask):

    defect_pixels = int(np.sum(defect_mask))
    solder_pixels = int(np.sum(solder_mask))

    metal = defect_pixels + solder_pixels

    ratio = defect_pixels / metal * 100 if metal > 0 else 0

    largest_void, circ = analyze_voids(defect_mask)

    return {
        "manque_%": round(ratio,2),
        "pixels_defaut": defect_pixels,
        "pixels_soudure": solder_pixels,
        "largest_void_pixels": int(largest_void),
        "largest_void_circularity": round(circ,3)
    }

