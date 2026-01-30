import torch
import numpy as np
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_file):

    model = torch.load(model_file, map_location=DEVICE)

    if hasattr(model, "eval"):
        model.eval()
        model.to(DEVICE)
        return model

    raise RuntimeError("Invalid model format")


@torch.no_grad()
def predict_mask(model, tensor, percentile=82):

    tensor = tensor.to(DEVICE)

    logits = model(tensor)

    probs = torch.softmax(logits, dim=1)[0]

    # assume last channel = defect
    defect_prob = probs[-1].cpu().numpy()

    defect_prob = cv2.GaussianBlur(defect_prob, (5,5), 0)

    # ⭐ adaptive threshold
    t = np.percentile(defect_prob, percentile)

    defect_mask = defect_prob > t

    return defect_mask, defect_prob

