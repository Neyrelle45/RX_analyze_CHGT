import torch
import torch.nn.functional as F
import numpy as np
import cv2


# -------------------------------------------------
# MODEL LOADING (SAFE)
# -------------------------------------------------

def load_model(model_file, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(model_file, map_location=device)
    model.eval()
    return model


# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

def predict_mask(model, tensor, threshold=0.25, temperature=1.0):
    with torch.no_grad():
        logits = model(tensor)

        logits = logits / temperature
        probs = F.softmax(logits, dim=1)

        void_prob = probs[0, 1].cpu().numpy()  # class 1 = VOID

        pred_mask = void_prob > threshold

    return pred_mask, void_prob


# -------------------------------------------------
# LARGEST VOID
# -------------------------------------------------

def find_largest_void(void_mask, heatmap, inspect_mask):
    mask = void_mask & inspect_mask

    mask = mask.astype(np.uint8)
    num, labels = cv2.connectedComponents(mask)

    largest_area = 0
    largest_label = None

    for i in range(1, num):
        area = np.sum(labels == i)
        if area > largest_area:
            largest_area = area
            largest_label = i

    if largest_label is None:
        return None, 0, 0.0

    largest_mask = labels == largest_label

    # IA confidence = mean prob on largest void
    ai_conf = float(np.mean(heatmap[largest_mask]))

    return largest_mask, largest_area, ai_conf
