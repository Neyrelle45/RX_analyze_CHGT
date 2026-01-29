import torch
import numpy as np
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# LOAD MODEL — INDUSTRIAL SAFE
# ---------------------------------------------------------

def load_model(model_file):
    """
    Supports:
    ✔ full saved model
    ✔ state_dict
    ✔ DataParallel
    """

    obj = torch.load(
        model_file,
        map_location=DEVICE
    )

    # -------------------------------------------------
    # CASE 1 — FULL MODEL (BEST)
    # -------------------------------------------------

    if isinstance(obj, torch.nn.Module):

        model = obj

    # -------------------------------------------------
    # CASE 2 — state_dict only
    # -------------------------------------------------

    elif isinstance(obj, dict):

        from engine.model import UNet

        model = UNet()

        if "state_dict" in obj:
            obj = obj["state_dict"]

        # remove "module." prefix if needed
        new_state = {}

        for k, v in obj.items():
            new_state[k.replace("module.", "")] = v

        model.load_state_dict(new_state)

    else:
        raise RuntimeError("Unsupported model format")

    model.to(DEVICE)
    model.eval()

    return model


# ---------------------------------------------------------
# PREDICTION — HEATMAP DRIVEN
# ---------------------------------------------------------

@torch.no_grad()
def predict_mask(model, tensor, threshold):

    tensor = tensor.to(DEVICE)

    logits = model(tensor)

    probs = torch.softmax(logits, dim=1)[0]

    # assume defect = LAST CHANNEL
    defect_prob = probs[-1].cpu().numpy()

    # smooth heatmap
    defect_prob = cv2.GaussianBlur(
        defect_prob,
        (5,5),
        0
    )

    mask = defect_prob > threshold

    return mask, defect_prob


