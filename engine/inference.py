import torch
import io
import numpy as np
import cv2

from engine.model import UNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# BUILD MODEL SAFE (signature tolerant)
# =========================================================

def build_model():

    try:
        model = UNet(n_classes=3)
    except TypeError:
        model = UNet()

    return model


# =========================================================
# LOAD MODEL (bulletproof)
# =========================================================

def load_model(model_file):

    if hasattr(model_file, "read"):
        buffer = io.BytesIO(model_file.read())
        obj = torch.load(buffer, map_location=DEVICE)
    else:
        obj = torch.load(model_file, map_location=DEVICE)

    # ---------- FULL PICKLE ----------
    if isinstance(obj, torch.nn.Module):
        model = obj

    # ---------- STATE DICT ----------
    elif isinstance(obj, dict):

        if "model_state" in obj:
            state = obj["model_state"]

        elif "state_dict" in obj:
            state = obj["state_dict"]

        else:
            state = obj

        model = build_model()
        model.load_state_dict(state)

    else:
        raise RuntimeError("Unsupported model format")

    model.to(DEVICE)
    model.eval()

    return model


# =========================================================
# RX INDUSTRIAL INFERENCE
# =========================================================

def infer_rx(model, img_tensor):
    """
    Industrial inference using LOGITS (NOT softmax).

    Returns:
        defect_score
        solder_score
    """

    with torch.no_grad():

        logits = model(torch.from_numpy(img_tensor).to(DEVICE))

        logits = logits[0].cpu().numpy()

        logit_bg = logits[0]
        logit_solder = logits[1]
        logit_defect = logits[2]

        # ⭐ DIFFERENTIAL SCORES (VERY IMPORTANT)
        defect_score = logit_defect - logit_solder
        solder_score = logit_solder - logit_defect

    return defect_score, solder_score


# =========================================================
# POST PROCESS (industrial cleanup)
# =========================================================

def clean_defects(defect_mask):
    """
    Remove pixel noise while keeping real voids.
    """

    kernel = np.ones((3,3), np.uint8)

    cleaned = cv2.morphologyEx(
        defect_mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        kernel
    )

    return cleaned.astype(bool)



