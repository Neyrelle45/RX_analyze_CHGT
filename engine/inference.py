import torch
import numpy as np
import cv2

from engine.model import UNet


# =====================================================
# LOAD MODEL — DEPLOYMENT SAFE
# =====================================================

def load_model(model_file):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = UNet()   # ⚠️ doit matcher training

    state_dict = torch.load(
        model_file,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


# =====================================================
# PREDICTION
# =====================================================

@torch.no_grad()
def predict_mask(model, tensor, threshold=0.25):

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    output = model(tensor)

    probs = torch.softmax(output, dim=1)[0]

    void_prob = probs[-1].cpu().numpy()

    # ⭐ normalisation robuste
    p2, p98 = np.percentile(void_prob, (2, 98))

    void_prob = np.clip(
        (void_prob - p2) / (p98 - p2 + 1e-6),
        0,
        1
    )

    pred_mask = void_prob > threshold

    return pred_mask, void_prob

