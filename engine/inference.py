import torch
import numpy as np
import cv2

from .model import UNet

DEVICE = "cpu"

def load_model(weights):

    model = UNet()
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.eval()

    return model


def predict(model, img):

    img = img.astype("float32") / 255.0
    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(t)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs,1)[0].cpu().numpy()
        defect_prob = probs[0,2].cpu().numpy()

    return pred, defect_prob
