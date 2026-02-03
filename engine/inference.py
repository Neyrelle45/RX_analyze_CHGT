import torch
import torch.nn.functional as F
import numpy as np
import cv2

# ⚠️ adapte cet import si ton UNet est ailleurs
from engine.unet import UNet


# =================================================
# MODEL LOADING — ROBUST / NO REGRESSION
# =================================================

def load_model(model_file, device=None):
    """
    Safe model loader.
    Supports:
    - torch.save(model)
    - torch.save(model.state_dict())

    Never crashes Streamlit.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        obj = torch.load(model_file, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Impossible de charger le modèle : {e}")

    # -------------------------------------------------
    # Case 1 — Full torch.nn.Module
    # -------------------------------------------------
    if isinstance(obj, torch.nn.Module):
        model = obj.to(device)
        model.eval()
        return model

    # -------------------------------------------------
    # Case 2 — state_dict
    # -------------------------------------------------
    if isinstance(obj, dict):
        try:
            model = UNet().to(device)
            model.load_state_dict(obj)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(
                "Le fichier .pth contient un state_dict incompatible "
                f"avec l'architecture UNet : {e}"
            )

    # -------------------------------------------------
    # Unknown format
    # -------------------------------------------------
    raise RuntimeError(
        "Format de modèle non reconnu.\n"
        "Le fichier .pth doit contenir soit :\n"
        "• un torch.nn.Module\n"
        "• un state_dict compatible UNet"
    )


# =================================================
# PREDICTION — VOID MASK + HEATMAP
# =================================================

def predict_mask(model, tensor, threshold=0.25, temperature=1.0):
    """
    Returns:
    - pred_mask (bool)
    - void_prob heatmap (float 0..1)
    """

    with torch.no_grad():
        logits = model(tensor)

        # Temperature scaling (robust, safe)
        logits = logits / max(temperature, 1e-6)

        probs = F.softmax(logits, dim=1)

        # Convention: class 1 = VOID
        void_prob = probs[0, 1].detach().cpu().numpy()

        pred_mask = void_prob >= threshold

    return pred_mask, void_prob


# =================================================
# LARGEST VOID — CONNECTED COMPONENTS
# =================================================

def find_largest_void(void_mask, heatmap, inspect_mask):
    """
    Returns:
    - largest_void_mask (bool or None)
    - largest_area_px (int)
    - ai_confidence (float 0..1)
    """

    if inspect_mask is None:
        inspect_mask = np.ones_like(void_mask, dtype=bool)

    # Ensure boolean
    mask = (void_mask & inspect_mask).astype(np.uint8)

    if mask.sum() == 0:
        return None, 0, 0.0

    # Connected components
    num_labels, labels = cv2.connectedComponents(mask)

    largest_area = 0
    largest_label = None

    for label in range(1, num_labels):  # 0 = background
        area = np.sum(labels == label)
        if area > largest_area:
            largest_area = area
            largest_label = label

    if largest_label is None:
        return None, 0, 0.0

    largest_void_mask = labels == largest_label

    # IA confidence = mean probability inside largest void
    ai_confidence = float(np.mean(heatmap[largest_void_mask]))

    return largest_void_mask, int(largest_area), ai_confidence
