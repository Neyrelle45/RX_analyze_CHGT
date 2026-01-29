import streamlit as st
import cv2
import yaml
import numpy as np
import pandas as pd

from engine.preprocessing import preprocess_rx, TARGET_SIZE
from engine.inference import load_model, predict
from engine.postprocessing import classify_defects
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

st.set_page_config(layout="wide")
st.title("RX Void Analyzer")


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    st.header("Model")

    model_file = st.file_uploader("Model (.pth)")
    cfg_file = st.file_uploader("config.yaml")
    mask_file = st.file_uploader("Mask (green = inspect)")

    st.divider()

    st.header("Detection")

    defect_th = st.slider(
        "Defect threshold",
        min_value=0.10,
        max_value=0.60,
        value=0.30,
        step=0.01
    )

    st.divider()

    st.header("Mask alignment")

    tx = st.slider("Translate X", -250, 250, 0)
    ty = st.slider("Translate Y", -250, 250, 0)
    scale = st.slider("Scale", 0.5, 1.5, 1.0)
    angle = st.slider("Rotation", -15, 15, 0)

    st.divider()

    st.header("Preprocessing")

    contrast = st.slider("Contrast", 0.5, 3.0, 1.0)
    denoise = st.slider("Denoise", 0, 20, 5)

    show_heatmap = st.checkbox("Show defect heatmap")


if not (model_file and cfg_file and mask_file):
    st.info("Load model, config and mask to start.")
    st.stop()


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

cfg = yaml.safe_load(cfg_file)
model = load_model(model_file)


# ---------------------------------------------------
# UTILS
# ---------------------------------------------------

def crop_valid(img, shape):
    """
    Remove letterbox padding.
    """
    nh, nw = shape
    return img[:nh, :nw]


# ---------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------

img_file = st.file_uploader("RX image")

if img_file:

    original = cv2.imdecode(
        np.frombuffer(img_file.read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    # ---------------- PREPROCESS ----------------

    img_net, valid_mask, scale_factor, shape = preprocess_rx(
        original,
        contrast,
        denoise
    )

    # ---------------- MASK ----------------

    mask = cv2.imdecode(
        np.frombuffer(mask_file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    mask = cv2.resize(
        mask,
        (TARGET_SIZE, TARGET_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    center = (TARGET_SIZE // 2, TARGET_SIZE // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[:, 2] += [tx, ty]

    mask = cv2.warpAffine(
        mask,
        M,
        (TARGET_SIZE, TARGET_SIZE),
        flags=cv2.INTER_NEAREST
    )

    inspect_mask = (mask[:, :, 1] > 200) & valid_mask

    # ---------------- INFERENCE ----------------

    pred, heat = predict(model, img_net, defect_th)

    solder = (pred == 1) & inspect_mask
    defect = (pred == 2) & inspect_mask

    # ---------------- METRICS (INDUSTRIAL) ----------------

    red_pixels = int(np.sum(defect))
    blue_pixels = int(np.sum(solder))

    metal_mask = ((pred == 1) | (pred == 2)) & inspect_mask
    metal_pixels = int(np.sum(metal_mask))

    lack_ratio = (red_pixels / metal_pixels * 100) if metal_pixels > 0 else 0

    metrics = {
        "manque_%": round(lack_ratio, 2),
        "pixels_rouges": red_pixels,
        "pixels_bleus": blue_pixels,
        "pixels_metal": metal_pixels
    }

    # ---------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------

    # ----- MASK OVERLAY -----

    mask_overlay = cv2.cvtColor(img_net, cv2.COLOR_GRAY2BGR)

    green = np.zeros_like(mask_overlay)
    green[inspect_mask] = [0, 255, 0]

    mask_overlay = cv2.addWeighted(mask_overlay, 1, green, 0.35, 0)

    # ----- ANALYSIS OVERLAY -----

    overlay = cv2.cvtColor(img_net, cv2.COLOR_GRAY2BGR)

    blue_layer = np.zeros_like(overlay)
    blue_layer[solder] = [180, 0, 0]  # BLEU = soudure

    red_layer = np.zeros_like(overlay)
    red_layer[defect] = [0, 0, 255]  # ROUGE = manque

    overlay = cv2.addWeighted(overlay, 1, blue_layer, 0.35, 0)
    overlay = cv2.addWeighted(overlay, 1, red_layer, 0.85, 0)

    # ----- REMOVE LETTERBOX -----

    mask_overlay = crop_valid(mask_overlay, shape)
    overlay = crop_valid(overlay, shape)

    # ---------------------------------------------------
    # DISPLAY
    # ---------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.image(
        original,
        caption="Original (ratio conservé)",
        use_column_width=True
    )

    col2.image(
        mask_overlay,
        caption="Masque ajusté",
        use_column_width=True
    )

    col3.image(
        overlay,
        caption="Analyse — BLEU=soudure | ROUGE=manque",
        use_column_width=True
    )

    if show_heatmap:
        heat_crop = crop_valid(heat, shape)

        st.image(
            heat_crop,
            clamp=True,
            caption="Heatmap défaut"
        )

    st.divider()

    st.subheader("Metrics")

    st.dataframe(pd.DataFrame([metrics]))


