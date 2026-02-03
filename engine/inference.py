import torch
import torch.nn.functional as F
import numpy as np
import cv2

# ⚠️ adapte si besoin le chemin exact
from engine.unet import UNet


# =================================================
# MODEL LOADING — DEFINITIVE / NO REGRESSION
# =================================================

def load_model(model_file, device=None):
    """
    Compatible avec :
    1) torch.save(model)
    2) torch.save(model.state_dict())
    3) torch.save({
           "model_state": state_dict,
           "model_name": "UNet",
           "n_classes": 3
       })
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(model_file, map_location=device)

    # -------------------------------------------------
    # Case 1 — full nn.Module
    # -------------------------------------------------
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint.to(device)
        model.eval()
        return model

    # -------------------------------------------------
    # Case 2 — state_dict only
    # -------------------------------------------------
    if isinstance(checkpoint, dict) and all(
        k.startswith(("encoder", "decoder", "conv", "down", "up"))
        for k in checkpoint.keys()
    ):
        model = UNet().to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    # -------------------------------------------------
    # Case 3 — TRAINING CHECKPOINT (TON CAS)
    # -------------------------------------------------
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:

        n_classes = checkpoint.get("n_classes", 3)

        model = UNet(n_classes=n_classes).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model

    # -------------------------------------------------
    # Unsupported
    # -------------------------------------------------
    raise RuntimeError(
        "Format de modèle non supporté.\n"
        "Le .pth doit contenir :\n"
        "- un torch.nn.Module\n"
        "- un state_dict\n"
        "- ou un dict avec 'model_state'"
    )


# =================================================
# PREDICTION
# =================================================

def predict_mask(model, tensor, threshold=0.25, temperature=1.0):
    """
    Returns:
    - pred_mask (bool)
    - void_probability heatmap (float 0..1)
    """

    with torch.no_grad():
        logits = model(tensor)

        # Temperature scaling (safe)
        logits = logits / max(temperature, 1e-6)

        probs = F.softmax(logits, dim=1)

        # Convention entraînement : classe 1 = VOID
        void_prob = probs[0, 1].cpu().numpy()

        pred_mask = void_prob >= threshold

    return pred_mask, void_prob


# =================================================
# LARGEST VOID
# =================================================

def find_largest_void(void_mask, heatmap, inspect_mask=None):
    """
    Returns:
    - largest_void_mask (bool or None)
    - largest_area_px (int)
    - ai_confidence (float 0..1)
    """

    if inspect_mask is None:
        inspect_mask = np.ones_like(void_mask, dtype=bool)

    mask = (void_mask & inspect_mask).astype(np.uint8)

    if mask.sum() == 0:
        return None, 0, 0.0

    num_labels, labels = cv2.connectedComponents(mask)

    largest_area = 0
    largest_label = None

    for label in range(1, num_labels):
        area = np.sum(labels == label)
        if area > largest_area:
            largest_area = area
            largest_label = label

    if largest_label is None:
        return None, 0, 0.0

    largest_void_mask = labels == largest_label

    ai_conf = float(np.mean(heatmap[largest_void_mask]))

    return largest_void_mask, int(largest_area), ai_conf
