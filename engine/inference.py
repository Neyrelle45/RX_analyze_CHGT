import torch
import numpy as np
from engine.model import UNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# LOAD MODEL — SAFE
# ---------------------------------------------------------

def load_model(model_file):

    model = UNet(n_classes=3)

    state = torch.load(
        model_file,
        map_location=DEVICE
    )

    # compatible DataParallel
    if "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()

    return model


# ---------------------------------------------------------
# PREDICT MASK — HEATMAP DRIVEN
# ---------------------------------------------------------

@torch.no_grad()
def predict_mask(model, tensor, threshold):

    tensor = tensor.to(DEVICE)

    logits = model(tensor)

    probs = torch.softmax(logits, dim=1)[0]

    # classes:
    # 0 = background
    # 1 = solder
    # 2 = void / defect

    defect_prob = probs[2].cpu().numpy()

    # -------------------------------------------------
    # POST FILTER — removes salt noise
    # -------------------------------------------------

    defect_prob = cv2.GaussianBlur(
        defect_prob,
        (5,5),
        0
    )

    mask = defect_prob > threshold

    return mask, defect_prob




