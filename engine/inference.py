import torch
import numpy as np
import cv2


# ------------------------------------------------
# LOAD MODEL (SAFE + FAST)
# ------------------------------------------------

def load_model(model_file):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(
        model_file,
        map_location=device
    )

    model.eval()
    model.to(device)

    return model


# ------------------------------------------------
# PREDICT
# ------------------------------------------------

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

        # ⭐ CONTRAST BOOST (CRUCIAL)
        p1, p995 = np.percentile(heatmap, (1, 99.5))
        heatmap = np.clip((heatmap - p1) / (p995 - p1 + 1e-6), 0, 1)
        heatmap = np.clip((heatmap - p2) / (p98 - p2 + 1e-6), 0, 1)

        pred_mask = heatmap > threshold

    return pred_mask, heatmap


# ------------------------------------------------
# LARGEST VOID
# ------------------------------------------------

def find_largest_void(pred_mask, heatmap, inspect_mask):

    mask = pred_mask & inspect_mask

    mask = mask.astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if n <= 1:
        return None, 0, 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = np.argmax(areas) + 1

    largest = labels == idx
    area = areas[idx-1]

    confidence = float(heatmap[largest].mean())

    return largest, area, confidence
