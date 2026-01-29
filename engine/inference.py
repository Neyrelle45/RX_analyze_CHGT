import torch
import io

from engine.model import UNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model_safely():
    """
    Instantiate UNet regardless of constructor signature.
    """

    try:
        # try modern signature
        model = UNet(n_classes=3)

    except TypeError:
        # fallback for older architecture
        model = UNet()

    return model


def load_model(model_file):

    # safe read
    if hasattr(model_file, "read"):
        buffer = io.BytesIO(model_file.read())
        obj = torch.load(buffer, map_location=DEVICE)
    else:
        obj = torch.load(model_file, map_location=DEVICE)

    # -------------------------------------------------
    # CASE 1 — state_dict
    # -------------------------------------------------

    if isinstance(obj, dict):

        if "model_state" in obj:
            state = obj["model_state"]

        elif "state_dict" in obj:
            state = obj["state_dict"]

        else:
            state = obj

        model = build_model_safely()
        model.load_state_dict(state)

    # -------------------------------------------------
    # CASE 2 — full model pickle
    # -------------------------------------------------

    elif isinstance(obj, torch.nn.Module):

        model = obj

    else:
        raise RuntimeError("Unsupported model format")

    model.to(DEVICE)
    model.eval()

    return model


