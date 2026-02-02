import torch
import torch.nn.functional as F
import numpy as np
import cv2

from engine.model import UNet  # DOIT être le même UNet que pour l'entraînement

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# LOAD MODEL (STREAMLIT / COLAB SAFE)
# ======================================================

def load_model(model_file):

    checkpoint = torch.load(
        model_file,
        map_location=DEVICE
    )

    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise RuntimeError(
            "❌ Modèle invalide.\n"
            "➡️ Le fichier doit contenir un checkpoint avec 'model_state'."
        )

    model = UNet()
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    return model


# ======================================================
# PREDICTION (AVEC TEMPERATURE SCALING)
# ======================================================

def predict_mask(
    model,
    tensor,
    threshold=0.25,
    temperature=1.6
):
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        logits = logits / temperature
        probs = F.softmax(logits, dim=1)

    # Convention : 0=background, 1=solder, 2=void
    void_prob = probs[0, 2].cpu().numpy()

    pred_mask = void_prob > threshold

    return pred_mask, void_prob


# ======================================================
# LARGEST REAL VOID (ENCAPSULÉ)
# ======================================================

def find_largest_void(pred_mask, heatmap, inspect_mask):

    # masque final
    mask = (pred_mask & inspect_mask).astype(np.uint8)

    # connected components (ordre CORRECT)
    num_labels, labels = cv2.connectedComponents(mask)

    largest_area = 0
    largest_mask = None
    confidence = 0.0

    # label 0 = background → on commence à 1
    for i in range(1, num_labels):

        comp = labels == i
        area = int(comp.sum())

        # filtrage bruit
        if area < 20:
            continue

        if area > largest_area:
            largest_area = area
            largest_mask = comp
            confidence = float(heatmap[comp].mean())

    return largest_mask, largest_area, confidence


