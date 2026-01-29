import sys
import os
import io
import zipfile

import streamlit as st
import numpy as np
import cv2
import pandas as pd
import torch


# ---------------------------------------------------
# PATH FIX (Streamlit Cloud safe)
# ---------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.preprocessing import preprocess_rx, TARGET_SIZE
from engine.inference import load_model


# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "saved_images" not in st.session_state:
    st.session_state.saved_images = []


# ---------------------------------------------------
# PAGE
# ---------------------------------------------------

st.set_page_config(layout="wide")
st.title("RX Void Analyzer — Industrial Stable Build")


# ===================================================
# SIDEBAR
# ===================================================

with st.sidebar:

    st.header("Model")

    model_file = st.file_uploader("Load model (.pth)", type=["pth"])
    mask_file = st.file_uploader("Load inspection mask", type=["png","jpg"])

    st.divider()

    st.header("Detection")

    defect_th = st.slider(
        "Defect threshold",
        0.05,
        0.6,
        0.22,
        0.01
    )

    dominance = st.slider(
        "Defect vs solder dominance",
        0.6,
        1.5,
        0.85,
        0.05
    )

    st.divider()

    st.header("Mask alignment")

    tx = st.slider("Translate X", -250, 250, 0)
    ty = st.slider("Translate Y", -250, 250, 0)
    scale_mask = st.slider("Scale", 0.7, 1.3, 1.0)
    rot = st.slider("Rotation", -15, 15, 0)

    st.divider()

    st.header("Image preprocessing")

    contrast = st.slider("Contrast", 0.7, 1.6, 1.1)
    denoise = st.slider("Denoise", 0, 6, 2)

    show_heatmap = st.checkbox("Show defect heatmap")


if not model_file or not mask_file:
    st.info("Please load a model and a mask.")
    st.stop()


# ===================================================
# LOAD MODEL
# ===================================================

model = load_model(model_file)
model.eval()


# ===================================================
# IMAGE UPLOAD
# ===================================================

img_file = st.file_uploader("Upload RX image", type=["png","jpg","jpeg"])

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

    nh, nw = shape


    # ===================================================
    # MASK (LETTERBOX SAFE + ALIGNMENT)
    # ===================================================

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

    # rotation + scale
    M = cv2.getRotationMatrix2D(
        (new_w//2, new_h//2),
        rot,
        scale_mask
    )

    transformed = cv2.warpAffine(
        resized_mask,
        M,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderValue=(0,0,0)
    )

    canvas = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)

    x1 = max(tx, 0)
    y1 = max(ty, 0)

    x2 = min(TARGET_SIZE, tx + new_w)
    y2 = min(TARGET_SIZE, ty + new_h)

    canvas[y1:y2, x1:x2] = transformed[
        max(-ty,0):max(-ty,0)+(y2-y1),
        max(-tx,0):max(-tx,0)+(x2-x1)
    ]

    inspect_mask = (canvas[:,:,1] > 200) & valid_mask


    # ===================================================
    # INFERENCE
    # ===================================================

    img_tensor = img_net.astype("float32") / 255.0
    img_tensor = np.expand_dims(img_tensor, (0,1))

    with torch.no_grad():
        logits = model(torch.from_numpy(img_tensor))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    prob_solder = probs[1]
    prob_defect = probs[2]


    # ===================================================
    # INDUSTRIAL DECISION LOGIC
    # ===================================================

    defect = (
        (prob_defect > defect_th) &
        (prob_defect > prob_solder * dominance) &
        inspect_mask
    )

    solder = (
        (prob_solder > 0.30) &
        ~defect &
        inspect_mask
    )


    # remove padding
    img_crop = img_net[:nh, :nw]
    defect_crop = defect[:nh, :nw]
    solder_crop = solder[:nh, :nw]
    mask_crop = inspect_mask[:nh, :nw]


    # ===================================================
    # OVERLAY COLORS
    # ===================================================

    overlay = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2BGR)

    overlay[solder_crop] = (0,255,255)   # JAUNE
    overlay[defect_crop] = (0,0,255)     # ROUGE

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    mask_vis = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2BGR)
    mask_vis[mask_crop] = (0,180,0)
    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB)


    # ===================================================
    # METRICS
    # ===================================================

    red_pixels = int(np.sum(defect_crop))
    yellow_pixels = int(np.sum(solder_crop))

    metal_pixels = red_pixels + yellow_pixels

    lack_ratio = (
        red_pixels / metal_pixels * 100
        if metal_pixels > 0 else 0
    )

    metrics = {
        "manque_%": round(lack_ratio,2),
        "pixels_defaut": red_pixels,
        "pixels_soudure": yellow_pixels
    }


    # ===================================================
    # DISPLAY
    # ===================================================

    col1, col2, col3 = st.columns(3)

    col1.image(original, use_column_width=True)
    col2.image(mask_vis, use_column_width=True)
    col3.image(overlay_rgb, use_column_width=True)

    if show_heatmap:
        heat = prob_defect[:nh, :nw]
        st.image(heat, clamp=True)

    st.dataframe(pd.DataFrame([metrics]))


    # ===================================================
    # SAVE
    # ===================================================

    if st.button("Save inspection"):

        st.session_state.history.append(metrics)

        _, buffer = cv2.imencode(".png", overlay)
        st.session_state.saved_images.append(buffer.tobytes())

        st.success("Inspection saved ✔")


# ===================================================
# HISTORY + EXPORT
# ===================================================

if st.session_state.history:

    st.subheader("Inspection history")

    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    csv = df.to_csv(index=False).encode()

    st.download_button(
        "Download CSV",
        csv,
        "inspection_results.csv"
    )

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as z:

        for i, img in enumerate(st.session_state.saved_images):
            z.writestr(f"inspection_{i}.png", img)

        z.writestr("results.csv", csv)

    st.download_button(
        "Download FULL report (ZIP)",
        zip_buffer.getvalue(),
        "inspection_report.zip"
    )

    if st.button("RESET SESSION"):

        st.session_state.history = []
        st.session_state.saved_images = []

        st.success("Session cleared")

