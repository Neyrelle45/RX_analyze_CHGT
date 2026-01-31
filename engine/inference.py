import torch
import torch.nn.functional as F
import numpy as np
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================
# SAFE MODEL LOADER (MODEL OR STATE_DICT)
# =====================================

def load_model(model_file):
    obj = torch.load(model_file, map_location=DEVICE)

    # CASE 1 — full model saved (torch.save(model))
    if hasattr(obj, "forward"):
        model = obj
        model.to(DEVICE)
        model.eval()
        return model

    # CASE 2 — state_dict saved (torch.save(model.state_dict()))
    if isinstance(obj, dict):
        raise RuntimeError(
            "❌ Le fichier .pth contient un state_dict.\n"
            "➡️ Charge le modèle avec la même classe UNet que pour l'entraînement "
            "OU réexporte avec torch.save(model)."
        )

    raise RuntimeError("❌ Format de modèle non reconnu.")


# ============================
# PREDICTION + TEMPERATURE
# ============================

def predict_mask(
    model,
    tensor,
    threshold=0.25,
    temperature=1.6
):
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)

        # Temperature scaling
        logits = logits / temperature

        probs = F.softmax(logits, dim=1)

    # class mapping (assumed)
    # 0 = background
    # 1 = solder
    # 2 = void
    void_prob = probs[0, 2].cpu().numpy()

    pred_mask = void_prob > threshold

    return pred_mask, void_prob


# ============================
# LARGEST REAL VOID
# ============================

def find_largest_void(pred_mask, heatmap, inspect_mask):
    masked = pred_mask & inspect_mask

    labeled, n = cv2.connectedComponents(masked.astype(np.uint8))

    largest_area = 0
    largest_mask = None
    confidence = 0.0

    for i in range(1, n):
        comp = labeled == i
        area = np.sum(comp)

        if area < 20:
            continue  # ignore noise

        if area > largest_area:
            largest_area = area
            largest_mask = comp
            confidence = float(np.mean(heatmap[comp]))

    return largest_mask, largest_area, confidence

