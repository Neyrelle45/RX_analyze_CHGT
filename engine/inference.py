import torch
import numpy as np

from .model import UNet

DEVICE = "cpu"


def load_model(weights):

    model = UNet()
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.eval()

    return model


def predict(model, img, defect_threshold=0.35):

    img = img.astype("float32") / 255.0
    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():

        logits = model(t)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    solder = probs[1] > 0.5
    defect = probs[2] > defect_threshold

    pred = np.zeros_like(probs[0], dtype=np.uint8)
    pred[solder] = 1
    pred[defect] = 2

    return pred, probs[2]
