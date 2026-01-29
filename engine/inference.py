import torch
import io

from engine.model import UNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_file):

    # Read uploaded file safely
    if hasattr(model_file, "read"):
        buffer = io.BytesIO(model_file.read())
        obj = torch.load(buffer, map_location=DEVICE)
    else:
        obj = torch.load(model_file, map_location=DEVICE)

    # -------------------------------------------------
    # CASE 1 — FULL MODEL (best case)
    # -------------------------------------------------

    if isinstance(obj, torch.nn.Module):

        model = obj
        model.to(DEVICE)
        model.eval()

        return model

    # -------------------------------------------------
    # CASE 2 — CHECKPOINT
    # -------------------------------------------------

    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj

    # -------------------------------------------------
    # BUILD MODEL
    # -------------------------------------------------

    model = UNet(n_classes=3)

    model.load_state_dict(state, strict=True)

    model.to(DEVICE)
    model.eval()

    return model

