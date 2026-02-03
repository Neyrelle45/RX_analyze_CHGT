import torch
import torch.nn.functional as F
import numpy as np
import cv2


# =================================================
# MODEL LOADING — STREAMLIT SAFE
# =================================================

def load_model(model_file, device=None):
    """
    Loader STRICTEMENT compatible avec ton entraînement Colab.
    Ne dépend PAS du code UNet côté app.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(model_file, map_location=device)

    # -------------------------------------------------
    # Cas attendu : checkpoint d'entraînement
    # -------------------------------------------------
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        raise RuntimeError(
            "❌ Le modèle a été sauvegardé comme state_dict.\n\n"
            "➡️ Pour l'inférence Streamlit, tu dois sauvegarder le modèle COMPLET :\n\n"
            "   torch.save(model, 'best_model.pth')\n\n"
            "Cela évite toute dépendance au code UNet."
        )

    # -------------------------------------------------
    # Cas valide : modèle complet
    # -------------------------------------------------
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint.to(device)
        model.eval()
        return model

    # -------------------------------------------------
    # Cas invalide
    # -------------------------------------------------
    raise RuntimeError(
        "❌ Format de modèle non supporté.\n"
        "Le fichier .pth doit contenir un modèle torch.nn.Module complet."
    )


# =================================================
# PREDICTION
# =================================================

def predict_mask(model, tensor, threshold=0.25, temperature=1.0):
    with torch.no_grad():
        logits = model(tensor)

        logits = logits / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=1)

        # Convention : classe 1 = VOID
        void_prob = probs[0, 1].cpu().numpy()
        pred_mask = void_prob >= threshold

    return pred_mask, void_prob


# =================================================
# LARGEST VOID
# =================================================

def find_largest_void(void_mask, heatmap, inspect_mask=None):

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

