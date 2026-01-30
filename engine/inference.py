import torch
import numpy as np
import cv2


# =====================================================
# LOAD MODEL — ULTRA ROBUST
# =====================================================

def load_model(model_file):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # charge checkpoint proprement
    checkpoint = torch.load(
        model_file,
        map_location=device
    )

    # cas 1 — modèle complet sauvegardé
    if hasattr(checkpoint, "eval"):
        model = checkpoint

    # cas 2 — state_dict uniquement
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:

        from engine.model import UNet

        model = UNet(n_classes=3)
        model.load_state_dict(checkpoint["state_dict"])

    else:
        from engine.model import UNet
        model = UNet(n_classes=3)
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


# =====================================================
# PREDICTION — HEATMAP PROPRE
# =====================================================

@torch.no_grad()
def predict_mask(model, tensor, threshold=0.25):

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    output = model(tensor)

    # shape : [1, C, H, W]
    probs = torch.softmax(output, dim=1)[0]

    # classe 1 = void
    void_prob = probs[1].cpu().numpy()

    # NORMALISATION CRITIQUE
    void_prob = cv2.normalize(
        void_prob,
        None,
        0,
        1,
        cv2.NORM_MINMAX
    )

    pred_mask = void_prob > threshold

    return pred_mask, void_prob

