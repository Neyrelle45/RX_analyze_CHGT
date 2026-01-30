import torch
import numpy as np
import cv2
import inspect


# =====================================================
# SAFE MODEL LOADER — INDUSTRIAL
# =====================================================

def build_unet():

    from engine.model import UNet

    sig = inspect.signature(UNet.__init__)
    params = sig.parameters

    # On adapte automatiquement selon ton UNet

    if "n_classes" in params:
        return UNet(n_classes=3)

    elif "num_classes" in params:
        return UNet(num_classes=3)

    elif "out_channels" in params:
        return UNet(out_channels=3)

    elif "classes" in params:
        return UNet(classes=3)

    else:
        # fallback ultra-safe
        return UNet(3)


def load_model(model_file):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        model_file,
        map_location=device
    )

    # -------------------------------------------------
    # CASE 1 — modèle complet sauvegardé
    # -------------------------------------------------

    if hasattr(checkpoint, "eval"):
        model = checkpoint

    # -------------------------------------------------
    # CASE 2 — checkpoint dict
    # -------------------------------------------------

    else:

        model = build_unet()

        if isinstance(checkpoint, dict):

            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])

            else:
                model.load_state_dict(checkpoint)

        else:
            raise RuntimeError(
                "Unsupported model format."
            )

    model.to(device)
    model.eval()

    return model


# =====================================================
# PREDICTION — HIGH CONTRAST HEATMAP
# =====================================================

@torch.no_grad()
def predict_mask(model, tensor, threshold=0.25):

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    output = model(tensor)

    probs = torch.softmax(output, dim=1)[0]

    void_prob = probs[1].cpu().numpy()

    # ⭐ NORMALISATION CRITIQUE
    p2, p98 = np.percentile(void_prob, (2, 98))
    void_prob = np.clip((void_prob - p2) / (p98 - p2 + 1e-6), 0, 1)

    pred_mask = void_prob > threshold

    return pred_mask, void_prob

