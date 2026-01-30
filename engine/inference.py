import torch
import numpy as np
import cv2


# =====================================================
# LOAD MODEL — DO NOT REBUILD
# =====================================================

def load_model(model_file):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ⭐ CRITIQUE : on charge tel quel
    model = torch.load(
        model_file,
        map_location=device
    )

    # sécurité minimale
    if not hasattr(model, "eval"):
        raise RuntimeError(
            "The .pth file does not contain a full model. "
            "Please export with torch.save(model)."
        )

    model.to(device)
    model.eval()

    return model


# =====================================================
# PREDICT — HIGH DYNAMIC HEATMAP
# =====================================================

@torch.no_grad()
def predict_mask(model, tensor, threshold=0.25):

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    output = model(tensor)

    probs = torch.softmax(output, dim=1)[0]

    # ⚠️ IMPORTANT
    # Sur 99% des datasets void :
    # classe 0 = background
    # classe 1 = solder
    # classe 2 = void

    void_prob = probs[-1].cpu().numpy()

    # ⭐ NORMALISATION PERCENTILE
    p1, p99 = np.percentile(void_prob, (1, 99))

    void_prob = np.clip(
        (void_prob - p1) / (p99 - p1 + 1e-6),
        0,
        1
    )

    pred_mask = void_prob > threshold

    return pred_mask, void_prob

