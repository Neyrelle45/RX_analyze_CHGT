import streamlit as st
import numpy as np
import cv2
import pandas as pd
import zipfile
import io
from PIL import Image

from engine.preprocessing import preprocess_rx
from engine.inference import (
    load_model,
    predict_mask,
    find_largest_void
)

st.set_page_config(layout="wide")
st.title("PAD VOID ENGINE — Industrial AI")

# =========================
# SESSION STATE
# =========================

st.session_state.setdefault("results", [])
st.session_state.setdefault("saved_images", [])

# =========================
# SIDEBAR — MODEL
# =========================

st.sidebar.header("Model")

model_file = st.sidebar.file_uploader("Load model (.pth)", type=["pth"])
mask_file = st.sidebar.file_uploader("Inspection mask", type=["png", "jpg"])

model = load_model(model_file) if model_file else None

# =========================
# DETECTION
# =========================

st.sidebar.header("Detection")

threshold = st.sidebar.slider("Void threshold", 0.05, 0.6, 0.25, 0.01)
temperature = st.sidebar.slider("Softmax temperature", 1.0, 3.0, 1.6, 0.05)

# =========================
# MASK ALIGNMENT
# =========================

st.sidebar.header("Mask alignment")

tx = st.sidebar.slider("Translate X", -80, 80, 0, 1)
ty = st.sidebar.slider("Translate Y", -80, 80, 0, 1)
scale = st.sidebar.slider("Scale", 0.95, 1.05, 1.0, 0.001)
angle = st.sidebar.slider("Rotation", -3.0, 3.0, 0.0, 0.1)

# =========================
# PREPROCESSING
# =========================

st.sidebar.header("RX preprocessing")

contrast = st.sidebar.slider("Global contrast", 1.0, 2.2, 1.6, 0.05)
clahe = st.sidebar.slider("Local contrast", 1.0, 4.0, 2.2, 0.1)
gamma = st.sidebar.slider("Gamma", 0.8, 1.6, 1.1, 0.05)

show_heatmap = st.sidebar.checkbox("Show defect heatmap", True)

# =========================
# IMAGE INPUT
# =========================

uploaded = st.file_uploader("RX image", type=["png", "jpg", "jpeg"])

if uploaded and model:

    original = np.array(Image.open(uploaded).convert("RGB"))

    tensor, processed = preprocess_rx(
        original, contrast, clahe, gamma
    )

    pred_mask, heatmap = predict_mask(
        model, tensor, threshold, temperature
    )

    h, w = processed.shape
    inspect_mask = np.ones((h, w), dtype=bool)

    # ---- optional inspection mask ----
    if mask_file:
        mask_img = np.array(Image.open(mask_file).convert("L"))
        mask_img = cv2.resize(mask_img, (w, h))

        M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)
        M[:, 2] += [tx, ty]

        aligned = cv2.warpAffine(mask_img, M, (w, h))
        inspect_mask = aligned > 127
        pred_mask &= inspect_mask

    # ---- largest real void ----
    largest_mask, largest_area, ai_conf = find_largest_void(
        pred_mask, heatmap, inspect_mask
    )

    # =========================
    # OVERLAY
    # =========================

    overlay = cv2.resize(original, (w, h))
    overlay[pred_mask] = [255, 0, 0]          # void = red
    overlay[inspect_mask & ~pred_mask] = [255, 255, 0]  # solder = yellow

    if largest_mask is not None:
        y, x = np.mean(np.column_stack(np.where(largest_mask)), axis=0).astype(int)
        r = int(np.sqrt(largest_area / np.pi))
        cv2.circle(overlay, (x, y), r, (135, 206, 235), 4)

    # =========================
    # METRICS
    # =========================

    void_px = int(np.sum(pred_mask))
    solder_px = int(np.sum(inspect_mask) - void_px)

    void_pct = void_px / max(1, void_px + solder_px) * 100
    largest_void_pct = largest_area / max(1, np.sum(inspect_mask)) * 100

    df = pd.DataFrame([{
        "void_%": round(void_pct, 2),
        "largest_void_%": round(largest_void_pct, 2),
        "IA_confidence_%": round(ai_conf * 100, 1),
        "void_pixels": void_px,
        "solder_pixels": solder_px
    }])

    # =========================
    # DISPLAY
    # =========================

    c1, c2, c3 = st.columns(3)
    c1.image(original, caption="Original", use_container_width=True)
    c2.image(overlay, caption="Detection", use_container_width=True)

    if show_heatmap:
        hm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        c3.image(hm, caption="Void probability heatmap", use_container_width=True)

    # =========================
    # SAVE / TABLE
    # =========================

    if st.button("Save inspection"):
        st.session_state.results.append(df.iloc[0])
        _, buf = cv2.imencode(".png", overlay)
        st.session_state.saved_images.append(buf.tobytes())

    if st.session_state.results:
        results_df = pd.DataFrame(st.session_state.results)

        def highlight(row):
            if row["void_%"] == results_df["void_%"].max():
                return ["background-color:#ff4b4b"] * len(row)
            if row["void_%"] == results_df["void_%"].min():
                return ["background-color:#4b8bff"] * len(row)
            return [""] * len(row)

        st.dataframe(results_df.style.apply(highlight, axis=1))

        csv = results_df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "inspection_results.csv")

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as z:
            for i, img in enumerate(st.session_state.saved_images):
                z.writestr(f"inspection_{i}.png", img)

        st.download_button("Download images ZIP", zip_buf.getvalue(), "images.zip")

        if st.button("Clear history"):
            st.session_state.results.clear()
            st.session_state.saved_images.clear()

