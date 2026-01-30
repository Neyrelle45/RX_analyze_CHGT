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
def predict_mask(model, tensor, threshold):

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with torch.inference_mode():

        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(tensor)
        else:
            out = model(tensor)

        probs = torch.softmax(out, dim=1)[0,1]

        heatmap = probs.detach().cpu().numpy()

        pred_mask = heatmap > threshold

    return pred_mask, heatmap



    
    return pred_mask, void_prob

def find_largest_void(void_mask, heatmap, inspect_mask):

    mask = (void_mask & inspect_mask).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    largest_area = 0
    largest_label = None

    H, W = mask.shape

    for i in range(1, num_labels):

        x, y, w, h, area = stats[i]

        # rejet blobs ouverts (touchent bord)
        if x == 0 or y == 0 or (x+w) >= W-1 or (y+h) >= H-1:
            continue

        if area > largest_area:
            largest_area = area
            largest_label = i

    if largest_label is None:
        return None, 0, 0

    largest_mask = labels == largest_label

    # confiance IA = prob moyenne
    confidence = heatmap[largest_mask].mean()

    return largest_mask, largest_area, confidence
