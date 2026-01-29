import sys
import os
import io
import zipfile

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import torch

from engine.preprocessing import preprocess_rx, TARGET_SIZE
from engine.inference import load_model


# =========================================================
# SESSION STORAGE
# =========================================================

if "results" not in st.session_state:
    st.session_state.results = []

if "images" not in st.session_state:
    st.session_state.images = []


# =========================================================
# PAGE
# =========================================================

st.set_page_config(layout="wide")
st.title("RX Void Analyzer — Stable Industrial Build")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Model")

    model_file = st.file_uploader("Model (.pth)")
    mask_file = st.file_uploader("Inspection mask")

    st.divider()

    st.header("Detection")

    defect_th = st.slider("Defect threshold", 0.05, 0.6, 0.25, 0.01)

    dominance = st.slider(
        "Defect dominance vs solder",
        1.0,
        2.0,
        1.2,
        0.05
    )

    st.divider()

    st.header("Image")

    contrast = st.slider("Contrast", 0.8, 1.5, 1.1)
    denoise = st.slider("Denoise", 0, 5, 2)

    show_heatmap = st.checkbox("Show defect heatmap")


if not (model_file and mask_file):
    st.info("Load a model and a mask.")
    st.stop()


# =========================================================
# LOAD MODEL
# =========================================================

model = load_model(model_file)
model.eval()


# =========================================================
# IMAGE
# =========================================================

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
        0,
        0
    )

    # -----------------------------------------------------
    # MASK ALIGN (LETTERBOX SAFE)
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    img_tensor = img_net.astype("float32") / 255.0
    img_tensor = np.expand_dims(img_tensor, (0, 1))

    with torch.no_grad():
        probs = torch.softmax(
            model(torch.from_numpy(img_tensor)),
            dim=1
        )[0].cpu().numpy()

    prob_solder = probs[1]
    prob_defect = probs[2]

    # -----------------------------------------------------
    # INDUSTRIAL DECISION LOGIC
    # -----------------------------------------------------

    metal_mask = prob_solder > 0.35

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

    # -----------------------------------------------------
    # CROP (REMOVE PADDING)
    # -----------------------------------------------------

    nh, nw = shape

    img_crop = img_net[:nh, :nw]
    defect_crop = defect[:nh, :nw]
    solder_crop = solder[:nh, :nw]
    mask_crop = inspect_mask[:nh, :nw]

    # -----------------------------------------------------
    # OVERLAY — INDUSTRIAL COLORS
    # -----------------------------------------------------

    overlay = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2BGR)

    # 🟡 SOLDER
    overlay[solder_crop] = (0, 255, 255)

    # 🔴 DEFECT
    overlay[defect_crop] = (0, 0, 255)

    # convert for streamlit
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # mask preview
    mask_vis = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2BGR)
    mask_vis[mask_crop] = (0, 180, 0)
    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB)

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.image(original, use_column_width=True)
    col2.image(mask_vis, use_column_width=True)
    col3.image(overlay_rgb, use_column_width=True)

    if show_heatmap:
        heat = prob_defect[:nh, :nw]
        st.image(heat, clamp=True)

    st.dataframe(pd.DataFrame([metrics]))

    # -----------------------------------------------------
    # SAVE RESULT
    # -----------------------------------------------------

    if st.button("Save inspection"):

        st.session_state.results.append(metrics)

        _, buffer = cv2.imencode(".png", overlay)
        st.session_state.images.append(buffer.tobytes())

        st.success("Inspection saved ✔")


# =========================================================
# RESULTS TABLE
# =========================================================

if st.session_state.results:

    st.subheader("Inspection history")

    df = pd.DataFrame(st.session_state.results)
    st.dataframe(df)

    csv = df.to_csv(index=False).encode()

    st.download_button(
        "Download CSV",
        csv,
        "inspection_results.csv"
    )

    # ZIP
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


