import streamlit as st
import numpy as np
import cv2
import pandas as pd
import io
import zipfile
from PIL import Image

from engine.preprocessing import preprocess_rx
from engine.inference import load_model, predict_mask, find_largest_void

# =====================================================
# PAGE
# =====================================================
st.set_page_config(layout="wide")
st.title("PAD VOID ENGINE — Industrial AI")

# =====================================================
# SESSION STATE
# =====================================================
if "results" not in st.session_state:
    st.session_state.results = []

if "saved_images" not in st.session_state:
    st.session_state.saved_images = []

# =====================================================
# SIDEBAR — MODEL
# =====================================================
st.sidebar.header("Model")

model_file = st.sidebar.file_uploader(
    "Load model (.pth)", type=["pth"]
)

model = load_model(model_file) if model_file else None

# =====================================================
# SIDEBAR — DETECTION
# =====================================================
st.sidebar.header("Detection")

threshold = st.sidebar.slider(
    "Void threshold", 0.05, 0.6, 0.25, 0.01
)

temperature = st.sidebar.slider(
    "Softmax temperature", 0.5, 2.0, 1.0, 0.05
)

# =====================================================
# SIDEBAR — RX PREPROCESSING
# =====================================================
st.sidebar.header("RX preprocessing")

contrast = st.sidebar.slider("Global contrast", 1.0, 2.2, 1.4, 0.05)
clahe = st.sidebar.slider("Local contrast", 1.0, 4.0, 2.2, 0.1)
gamma = st.sidebar.slider("Gamma", 0.8, 1.6, 1.1, 0.05)

show_heatmap = st.sidebar.checkbox("Show void probability heatmap", True)

# =====================================================
# IMAGE INPUT
# =====================================================
uploaded = st.file_uploader(
    "RX image", type=["png", "jpg", "jpeg"]
)

if uploaded and model:

    # =================================================
    # LOAD IMAGE
    # =================================================
    original = np.array(Image.open(uploaded).convert("RGB"))

    # =================================================
    # PREPROCESS
    # =================================================
    tensor, processed = preprocess_rx(
        original,
        contrast,
        clahe,
        gamma
    )

    # =================================================
    # INFERENCE
    # =================================================
    pred_mask, heatmap = predict_mask(
        model,
        tensor,
        threshold,
        temperature
    )

    h, w = pred_mask.shape

    # =================================================
    # LARGEST VOID
    # =================================================
    inspect_mask = np.ones_like(pred_mask, dtype=bool)

    largest_mask, largest_area, ai_conf = find_largest_void(
        pred_mask,
        heatmap,
        inspect_mask
    )

    # =================================================
    # OVERLAY (COLORS FIXED)
    # =================================================
    overlay = cv2.resize(original, (w, h))

    void_pixels = pred_mask.astype(bool)
    solder_pixels = ~void_pixels

    overlay[void_pixels] = (255, 0, 0)      # VOID = RED
    overlay[solder_pixels] = (0, 255, 255)  # SOLDER = YELLOW

    # Largest void contour (cyan)
    if largest_mask is not None and largest_area > 0:
        contours, _ = cv2.findContours(
            largest_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            overlay, contours, -1, (255, 255, 0), 3
        )

    # =================================================
    # METRICS
    # =================================================
    void_count = int(void_pixels.sum())
    solder_count = int(solder_pixels.sum())
    denom = void_count + solder_count

    void_pct = (void_count / denom * 100) if denom > 0 else 0
    largest_void_pct = (largest_area / denom * 100) if denom > 0 else 0

    # =================================================
    # DISPLAY
    # =================================================
    col1, col2, col3 = st.columns(3)

    col1.image(original, caption="Original", use_container_width=True)
    col2.image(overlay, caption="Detection", use_container_width=True)

    if show_heatmap:
        hm = heatmap.copy()
        hm = np.clip(hm, 0, 1)
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-6)

        heatmap_vis = (hm * 255).astype(np.uint8)
        heatmap_vis = cv2.applyColorMap(
            heatmap_vis, cv2.COLORMAP_JET
        )
        heatmap_vis = cv2.resize(
            heatmap_vis,
            (original.shape[1], original.shape[0])
        )

        col3.image(
            heatmap_vis,
            caption="Void probability heatmap",
            use_container_width=True
        )

    # =================================================
    # SAVE INSPECTION
    # =================================================
    if st.button("Save inspection"):

        st.session_state.results.append({
            "void_%": round(void_pct, 2),
            "largest_void_%": round(largest_void_pct, 2),
            "IA_confidence_%": round(ai_conf * 100, 1),
            "void_pixels": void_count,
            "solder_pixels": solder_count
        })

        _, buffer = cv2.imencode(".png", overlay)
        st.session_state.saved_images.append(buffer.tobytes())

        st.success("Inspection saved ✔")

    # =================================================
    # RESULTS TABLE
    # =================================================
    if st.session_state.results:

        df = pd.DataFrame(st.session_state.results)

        def highlight(row):
            if row["void_%"] == df["void_%"].max():
                return ["background-color:#ff4b4b"] * len(row)
            if row["void_%"] == df["void_%"].min():
                return ["background-color:#4b8bff"] * len(row)
            return [""] * len(row)

        st.subheader("Inspection results")
        st.dataframe(df.style.apply(highlight, axis=1))

        # EXPORTS
        csv = df.to_csv(index=False).encode()
        st.download_button(
            "Download CSV",
            csv,
            "inspection_results.csv"
        )

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as z:
            for i, img in enumerate(st.session_state.saved_images):
                z.writestr(f"inspection_{i}.png", img)

        st.download_button(
            "Download images ZIP",
            zip_buf.getvalue(),
            "inspections.zip"
        )

