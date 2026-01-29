import torch
import io

from engine.model import UNet   # IMPORTANT : doit exister


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_file):

    # read safely
    if hasattr(model_file, "read"):
        buffer = io.BytesIO(model_file.read())
        obj = torch.load(
            buffer,
            map_location=DEVICE,
            weights_only=False   # CRITIQUE pour compat pickle
        )
    else:
        obj = torch.load(
            model_file,
            map_location=DEVICE,
            weights_only=False
        )

    # -------------------------------------------------
    # CASE 1 — state_dict
    # -------------------------------------------------

    if isinstance(obj, dict):

        # checkpoint format
        if "model_state" in obj:
            state = obj["model_state"]

        elif "state_dict" in obj:
            state = obj["state_dict"]

        else:
            state = obj

        model = UNet(n_classes=3)
        model.load_state_dict(state)

    # -------------------------------------------------
    # CASE 2 — full pickle model
    # -------------------------------------------------

    elif isinstance(obj, torch.nn.Module):

        model = obj

    else:
        raise RuntimeError(
            "Unsupported model format. Save with state_dict."
        )

    model.to(DEVICE)
    model.eval()

    return model

