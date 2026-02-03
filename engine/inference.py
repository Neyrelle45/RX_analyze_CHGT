import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


# =================================================
# UNET — EMBARQUÉ (MATCH TRAINING)
# =================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


class UNet(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()

        self.d1 = DoubleConv(1, 32)
        self.d2 = DoubleConv(32, 64)
        self.d3 = DoubleConv(64, 128)
        self.d4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.u3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c3 = DoubleConv(256, 128)

        self.u2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c2 = DoubleConv(128, 64)

        self.u1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.c1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))

        x = self.u3(c4)
        x = self.c3(torch.cat([x, c3], dim=1))

        x = self.u2(x)
        x = self.c2(torch.cat([x, c2], dim=1))

        x = self.u1(x)
        x = self.c1(torch.cat([x, c1], dim=1))

        return self.out(x)


# =================================================
# MODEL LOADING — DEFINITIVE
# =================================================

def load_model(model_file, device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(model_file, map_location=device)

    # === TRAINING CHECKPOINT (TON CAS) ===
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:

        n_classes = checkpoint.get("n_classes", 3)

        model = UNet(n_classes=n_classes).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        return model

    # === FULL MODEL (OPTIONNEL) ===
    if isinstance(checkpoint, nn.Module):
        model = checkpoint.to(device)
        model.eval()
        return model

    raise RuntimeError(
        "Format de modèle non reconnu.\n"
        "Le fichier .pth doit provenir de ton entraînement UNet."
    )


# =================================================
# PREDICTION
# =================================================

def predict_mask(model, tensor, threshold=0.25, temperature=1.0):

    with torch.no_grad():
        logits = model(tensor)

        logits = logits / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=1)

        # Classe 1 = VOID
        void_prob = probs[0, 1].cpu().numpy()
        pred_mask = void_prob >= threshold

    return pred_mask, void_prob


# =================================================
# LARGEST VOID
# =================================================

def find_largest_void(void_mask, heatmap, inspect_mask=None):

    if inspect_mask is None:
        inspect_mask = np.ones_like(void_mask, dtype=bool)

    mask = (void_mask & inspect_mask).astype(np.uint8)

    if mask.sum() == 0:
        return None, 0, 0.0

    num_labels, labels = cv2.connectedComponents(mask)

    largest_area = 0
    largest_label = None

    for label in range(1, num_labels):
        area = np.sum(labels == label)
        if area > largest_area:
            largest_area = area
            largest_label = label

    if largest_label is None:
        return None, 0, 0.0

    largest_mask = labels == largest_label
    ai_conf = float(np.mean(heatmap[largest_mask]))

    return largest_mask, int(largest_area), ai_conf
