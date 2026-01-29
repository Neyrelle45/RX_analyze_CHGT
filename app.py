import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import cv2
import yaml
import numpy as np
import pandas as pd

from engine.preprocessing import preprocess_rx, TARGET_SIZE
from engine.inference import load_model


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

st.set_page_config(layout="wide")
st.title("RX Void Analyzer — Advanced")


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    model_file = st.file_uploader("Model (.pth)")
    cfg_file = st.file_uploader("Config (.yaml)")
    mask_file = st.file_uploader("Mask")

    st.divider()

    st.header("Detection")

    defect_th = st.slider("Base defect threshold", 0.05, 0.6, 0.20, 0.01)

    dominance = st.slider(
        "Defect dominance vs solder",
        0.5,
        2.0,
        0.9,
        0.05
    )

    st.divider()

    st.header("Mask alignment")

    tx = st.slider("Translate X", -250, 250, 0)
    ty = st.slider("Translate Y", -250, 250, 0)
    scale_mask = st.slider("Mask scale", 0.5, 1.5, 1.0)
    angle = st.slider("Rotation", -10, 10, 0)

    st.divider()

    st.header("Image")

    contrast = st.slider("Contrast", 0.5, 3.0, 1.2)
    denoise = st.slider("Denoise", 0, 20, 4)

    show_heatmap = st.checkbox("Show heatmap")


if not (model_file and cfg_file and mask_file):
    st.stop()

cfg = yaml.safe_load(cfg_file)
model = load_model(model_file)


# ---------------------------------------------------
# UTILS
# ---------------------------------------------------

def crop_valid(img, shape):
    nh, nw = shape
    return img[:nh, :nw]


# ---------------------------------------------------
# IMAGE
# ---------------------------------------------------

img_file = st.file_uploader("RX image")

if img_file:

    original = cv2.imdecode(
        np.frombuffer(img_file.read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    img_net, valid_mask, scale, shape = preprocess_rx(
        original,
        contrast,
        denoise
    )

    # ---------------------------------------------------
    # MASK — ALIGNED WITH LETTERBOX
    # ---------------------------------------------------

    raw_mask = cv2.imdecode(
        np.frombuffer(mask_file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    mh, mw = raw_mask.shape[:2]

    new_h = int(mh * scale)
    new_w = int(mw * scale)

    resized_mask = cv2.resize(
        raw_mask,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    mask_canvas = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
    mask_canvas[:new_h, :new_w] = resized_mask

    center = (TARGET_SIZE // 2, TARGET_SIZE // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale_mask)
    M[:, 2] += [tx, ty]

    mask_canvas = cv2.warpAffine(
        mask_canvas,
        M,
        (TARGET_SIZE, TARGET_SIZE),
        flags=cv2.INTER_NEAREST
    )

    inspect_mask = (mask_canvas[:, :, 1] > 200) & valid_mask

    # ---------------------------------------------------
    # MODEL
    # ---------------------------------------------------

    img_tensor = img_net.astype("float32") / 255.0
    img_tensor = np.expand_dims(img_tensor, (0, 1))

    import torch

    with torch.no_grad():
        logits = model(torch.from_numpy(img_tensor))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    prob_bg = probs[0]
    prob_solder = probs[1]
    prob_defect = probs[2]

    # 🔥 ADVANCED DECISION
    defect = (
        (prob_defect > defect_th) &
        (prob_defect > prob_solder * dominance) &
        inspect_mask
    )

    solder = (
        (prob_solder > prob_defect) &
        inspect_mask
    )

    # ---------------------------------------------------
    # METRICS
    # ---------------------------------------------------

    red_pixels = int(np.sum(defect))
    blue_pixels = int(np.sum(solder))

    metal_pixels = red_pixels + blue_pixels
    lack_ratio = red_pixels / metal_pixels * 100 if metal_pixels > 0 else 0

    metrics = {
        "manque_%": round(lack_ratio, 2),
        "pixels_rouges": red_pixels,
        "pixels_bleus": blue_pixels
    }

    # ---------------------------------------------------
    # VISUALS
    # ---------------------------------------------------

    mask_overlay = cv2.cvtColor(img_net, cv2.COLOR_GRAY2BGR)
    green = np.zeros_like(mask_overlay)
    green[inspect_mask] = [0, 255, 0]
    mask_overlay = cv2.addWeighted(mask_overlay, 1, green, 0.35, 0)

    overlay = cv2.cvtColor(img_net, cv2.COLOR_GRAY2BGR)

    blue = np.zeros_like(overlay)
    blue[solder] = [180, 0, 0]

    red = np.zeros_like(overlay)
    red[defect] = [0, 0, 255]

    overlay = cv2.addWeighted(overlay, 1, blue, 0.35, 0)
    overlay = cv2.addWeighted(overlay, 1, red, 0.9, 0)

    overlay = crop_valid(overlay, shape)
    mask_overlay = crop_valid(mask_overlay, shape)

    # ---------------------------------------------------
    # DISPLAY
    # ---------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.image(original, caption="Original", use_column_width=True)
    col2.image(mask_overlay, caption="Mask aligned", use_column_width=True)
    col3.image(overlay, caption="Detection", use_column_width=True)

    if show_heatmap:
        heat = crop_valid(prob_defect, shape)
        st.image(heat, clamp=True, caption="Defect heatmap")

    st.dataframe(pd.DataFrame([metrics]))



