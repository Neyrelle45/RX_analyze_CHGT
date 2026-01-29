import sys
import os
import zipfile
import io

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import cv2
import yaml
import numpy as np
import pandas as pd
import torch

from engine.preprocessing import preprocess_rx, TARGET_SIZE
from engine.inference import load_model


# ---------------------------------------------------
# SESSION STORAGE
# ---------------------------------------------------

if "results" not in st.session_state:
    st.session_state.results = []

if "images" not in st.session_state:
    st.session_state.images = []


# ---------------------------------------------------
# PAGE
# ---------------------------------------------------

st.set_page_config(layout="wide")
st.title("RX Void Analyzer — Inspection Mode")


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    model_file = st.file_uploader("Model (.pth)")
    cfg_file = st.file_uploader("Config (.yaml)")
    mask_file = st.file_uploader("Mask")

    st.divider()

    st.header("Detection")

    defect_th = st.slider("Threshold", 0.05, 0.6, 0.22, 0.01)
    dominance = st.slider("Defect dominance", 0.5, 2.0, 1.1, 0.05)

    st.divider()

    st.header("Image")

    contrast = st.slider("Contrast", 0.5, 3.0, 1.2)
    denoise = st.slider("Denoise", 0, 20, 4)
    clahe = st.slider("CLAHE", 0.0, 3.0, 1.5)
    blackhat = st.slider("Void enhancer", 0, 5, 2)

    show_heatmap = st.checkbox("Heatmap")


if not (model_file and cfg_file and mask_file):
    st.stop()

model = load_model(model_file)
model.eval()


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
        denoise,
        clahe,
        blackhat
    )

    # ---------------------------------------------------
    # MASK ALIGN
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

    inspect_mask = (mask_canvas[:, :, 1] > 200) & valid_mask

    # ---------------------------------------------------
    # MODEL
    # ---------------------------------------------------

    img_tensor = img_net.astype("float32") / 255.0
    img_tensor = np.expand_dims(img_tensor, (0, 1))

    with torch.no_grad():
        probs = torch.softmax(
            model(torch.from_numpy(img_tensor)),
            dim=1
        )[0].cpu().numpy()

    prob_solder = probs[1]
    prob_defect = probs[2]

    metal_mask = prob_solder > 0.25

    defect = (
        (prob_defect > defect_th) &
        (prob_defect > prob_solder * dominance) &
        metal_mask &
        inspect_mask
    )

    solder = (
        (prob_solder >= prob_defect) &
        inspect_mask
    )

    # ---------------------------------------------------
    # CROP (ANTI BLACK BARS)
    # ---------------------------------------------------

    nh, nw = shape

    img_crop = img_net[:nh, :nw]
    defect_crop = defect[:nh, :nw]
    solder_crop = solder[:nh, :nw]
    mask_crop = inspect_mask[:nh, :nw]

    # ---------------------------------------------------
    # OVERLAY — INDUSTRIAL COLORS
    # ---------------------------------------------------

    overlay = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2BGR)

    # 🟡 SOLDER
    overlay[solder_crop] = (0, 255, 255)

    # 🔴 DEFECT
    overlay[defect_crop] = (0, 0, 255)

    # mask preview
    mask_vis = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2BGR)
    mask_vis[mask_crop] = (0, 180, 0)

    # ---------------------------------------------------
    # METRICS
    # ---------------------------------------------------

    red_pixels = int(np.sum(defect_crop))
    yellow_pixels = int(np.sum(solder_crop))

    metal_pixels = red_pixels + yellow_pixels

    lack_ratio = (
        red_pixels / metal_pixels * 100
        if metal_pixels > 0 else 0
    )

    metrics = {
        "manque_%": round(lack_ratio, 2),
        "pixels_defaut": red_pixels,
        "pixels_soudure": yellow_pixels
    }

    # ---------------------------------------------------
    # DISPLAY
    # ---------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.image(original, use_column_width=True)
    col2.image(mask_vis, use_column_width=True)
    col3.image(overlay, use_column_width=True)

    if show_heatmap:
        heat = prob_defect[:nh, :nw]
        st.image(heat, clamp=True)

    st.dataframe(pd.DataFrame([metrics]))

    # ---------------------------------------------------
    # SAVE BUTTON
    # ---------------------------------------------------

    if st.button("Save result"):

        st.session_state.results.append(metrics)

        _, buffer = cv2.imencode(".png", overlay)
        st.session_state.images.append(buffer.tobytes())

        st.success("Saved ✔")


# ---------------------------------------------------
# RESULTS TABLE
# ---------------------------------------------------

if st.session_state.results:

    st.subheader("Inspection table")

    df = pd.DataFrame(st.session_state.results)
    st.dataframe(df)

    csv = df.to_csv(index=False).encode()

    st.download_button(
        "Download CSV",
        csv,
        "inspection_results.csv"
    )

    # ZIP images
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as z:
        for i, img in enumerate(st.session_state.images):
            z.writestr(f"inspection_{i}.png", img)

        z.writestr("results.csv", csv)

    st.download_button(
        "Download FULL report (ZIP)",
        zip_buffer.getvalue(),
        "inspection_report.zip"
    )

    if st.button("RESET SESSION"):
        st.session_state.results = []
        st.session_state.images = []
        st.success("Session cleared")

