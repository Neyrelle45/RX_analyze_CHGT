import streamlit as st
import numpy as np
import cv2
import pandas as pd
import zipfile
import io
from PIL import Image

from engine.preprocessing import preprocess_rx
from engine.inference import load_model, predict_mask, find_largest_void

st.set_page_config(layout="wide")
st.title("PAD VOID ENGINE — Industrial AI")

# =====================================================
# SESSION STATE
# =====================================================

if "results" not in st.session_state:
    st.session_state.results = []

if "images" not in st.session_state:
    st.session_state.images = []

# =====================================================
# SIDEBAR — MODEL
# =====================================================

st.sidebar.header("Model")

model_file = st.sidebar.file_uploader("Load model (.pth)", type=["pth"])
mask_file = st.sidebar.file_uploader("Inspection mask", type=["png", "jpg"])

model = load_model(model_file) if model_file else None

# =====================================================
# DETECTION
# =====================================================

st.sidebar.header("Detection")

threshold = st.sidebar.slider("Void threshold", 0.05, 0.6, 0.25, 0.01)
temperature = st.sidebar.slider("Softmax temperature", 0.7, 2.0, 1.2, 0.05)

# =====================================================
# MASK ALIGNMENT
# =====================================================

st.sidebar.header("Mask alignment")

tx = st.sidebar.slider("Translate X", -80, 80, 0, 1)
ty = st.sidebar.slider("Translate Y", -80, 80, 0, 1)
scale = st.sidebar.slider("Scale", 0.85, 1.15, 1.0, 0.005)
angle = st.sidebar.slider("Rotation", -3.0, 3.0, 0.0, 0.1)

# =====================================================
# RX PREPROCESSING
# =====================================================

st.sidebar.header("RX preprocessing")

contrast = st.sidebar.slider("Global contrast", 1.0, 2.2, 1.6, 0.05)
clahe = st.sidebar.slider("Local contrast", 1.0, 4.0, 2.2, 0.1)
gamma = st.sidebar.slider("Gamma", 0.8, 1.6, 1.1, 0.05)

show_heatmap = st.sidebar.checkbox("Show void probability heatmap", True)

# =====================================================
# IMAGE INPUT
# =====================================================

uploaded = st.file_uploader("RX image", type=["png", "jpg", "jpeg"])

if uploaded and model:

    original = np.array(Image.open(uploaded).convert("RGB"))
    h0, w0, _ = original.shape

    tensor, processed = preprocess_rx(original, contrast, clahe, gamma)

    pred_mask, heatmap = predict_mask(
        model, tensor, threshold, temperature
    )

    h, w = processed.shape

    # Inspection mask
    inspect_mask = np.ones((h, w), dtype=bool)

    if mask_file:
        m = np.array(Image.open(mask_file).convert("L"))
        m = cv2.resize(m, (w, h))

        M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)
        M[:, 2] += [tx, ty]
        m = cv2.warpAffine(m, M, (w, h))

        inspect_mask = m > 127
        pred_mask &= inspect_mask

    # Largest void
    largest_mask, largest_area, ai_conf = find_largest_void(
        pred_mask, heatmap, inspect_mask
    )

    # Overlay (ratio preserved)
    overlay = cv2.resize(original, (w, h))
    overlay[pred_mask] = [255, 0, 0]      # VOID
    overlay[inspect_mask & ~pred_mask] = [255, 255, 0]  # SOLDER

    if largest_mask is not None:
        contours, _ = cv2.findContours(
            largest_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (135, 206, 235), 3)

    # Metrics
    void_px = int(np.sum(pred_mask))
    solder_px = int(np.sum(inspect_mask & ~pred_mask))
    total_px = void_px + solder_px

    void_pct = (void_px / total_px * 100) if total_px else 0
    largest_void_pct = (largest_area / total_px * 100) if total_px else 0

    # DISPLAY
    c1, c2, c3 = st.columns(3)
    c1.image(original, caption="Original", use_container_width=True)
    c2.image(overlay, caption="Detection", use_container_width=True)

    if show_heatmap:
        hm = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        hm = cv2.resize(hm, (w, h))
        c3.image(hm, caption="Void probability heatmap", use_container_width=True)

    # TABLE
    row = {
        "void_%": round(void_pct, 2),
        "largest_void_%": round(largest_void_pct, 2),
        "IA_confidence_%": round(ai_conf * 100, 1),
        "void_pixels": void_px,
        "solder_pixels": solder_px,
    }

    if st.button("Save inspection"):
        st.session_state.results.append(row)
        _, buf = cv2.imencode(".png", overlay)
        st.session_state.images.append(buf.tobytes())
        st.success("Inspection saved")

    if st.session_state.results:
        df = pd.DataFrame(st.session_state.results)

        def highlight(r):
            styles = [""] * len(r)
            if r["void_%"] == df["void_%"].max():
                styles = ["background-color:#ff4b4b"] * len(r)
            if r["void_%"] == df["void_%"].min():
                styles = ["background-color:#4b8bff"] * len(r)
            return styles

        st.dataframe(df.style.apply(highlight, axis=1))

        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode(),
            "results.csv"
        )

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as z:
            for i, img in enumerate(st.session_state.images):
                z.writestr(f"inspection_{i}.png", img)

        st.download_button(
            "Download images ZIP",
            zip_buf.getvalue(),
            "images.zip"
        )

        if st.button("Clear results"):
            st.session_state.results = []
            st.session_state.images = []
