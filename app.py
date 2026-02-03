import streamlit as st
import numpy as np
import cv2
import pandas as pd
import zipfile
import io
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
    "Load model (.pth)",
    type=["pth"]
)

mask_file = st.sidebar.file_uploader(
    "Inspection mask",
    type=["png", "jpg", "jpeg"]
)

model = load_model(model_file) if model_file else None


# =====================================================
# SIDEBAR — DETECTION
# =====================================================

st.sidebar.header("Detection")

threshold = st.sidebar.slider(
    "Void threshold",
    0.05, 0.6, 0.25, 0.01
)

temperature = st.sidebar.slider(
    "Softmax temperature",
    0.7, 2.0, 1.3, 0.05
)


# =====================================================
# SIDEBAR — MASK ALIGNMENT
# =====================================================

st.sidebar.header("Mask alignment")

tx = st.sidebar.slider("Translate X", -80, 80, 0, 1)
ty = st.sidebar.slider("Translate Y", -80, 80, 0, 1)

scale = st.sidebar.slider(
    "Scale",
    0.85, 1.15, 1.0, 0.002
)

angle = st.sidebar.slider(
    "Rotation",
    -3.0, 3.0, 0.0, 0.1
)


# =====================================================
# SIDEBAR — RX PREPROCESSING
# =====================================================

st.sidebar.header("RX preprocessing")

contrast = st.sidebar.slider(
    "Global contrast",
    1.0, 2.2, 1.55, 0.05
)

clahe = st.sidebar.slider(
    "Local contrast",
    1.0, 4.0, 2.2, 0.1
)

gamma = st.sidebar.slider(
    "Gamma",
    0.8, 1.6, 1.1, 0.05
)

show_heatmap = st.sidebar.checkbox(
    "Show void probability heatmap",
    True
)


# =====================================================
# IMAGE INPUT
# =====================================================

uploaded = st.file_uploader(
    "RX image",
    type=["png", "jpg", "jpeg"]
)

if uploaded and model:

    original = np.array(Image.open(uploaded).convert("RGB"))

    # ================= PREPROCESS =================

    tensor, processed = preprocess_rx(
        original,
        contrast,
        clahe,
        gamma
    )

    h, w = processed.shape

    # ================= PREDICTION =================

    pred_mask, heatmap = predict_mask(
        model,
        tensor,
        threshold,
        temperature
    )

    # ================= INSPECTION MASK =================

    inspect_mask = np.ones((h, w), dtype=bool)

    if mask_file:

        mask_img = np.array(
            Image.open(mask_file).convert("L")
        )

        mask_img = cv2.resize(mask_img, (w, h))

        M = cv2.getRotationMatrix2D(
            (w // 2, h // 2),
            angle,
            scale
        )

        M[:, 2] += [tx, ty]

        aligned = cv2.warpAffine(mask_img, M, (w, h))
        inspect_mask = aligned > 127

        pred_mask = pred_mask & inspect_mask

    # ================= LARGEST VOID =================

    largest_mask, largest_area, ai_conf = find_largest_void(
        pred_mask,
        heatmap,
        inspect_mask
    )

    # ================= OVERLAY =================

    overlay = cv2.resize(original, (w, h))

    void_pixels = pred_mask
    solder_pixels = (~pred_mask) & inspect_mask

    overlay[void_pixels] = [255, 0, 0]       # RED = VOID
    overlay[solder_pixels] = [255, 255, 0]   # YELLOW = SOLDER

    if largest_mask is not None:
        contours, _ = cv2.findContours(
            largest_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            overlay,
            contours,
            -1,
            (135, 206, 235),  # bleu ciel
            3
        )

    # ================= METRICS =================

    void_px = int(void_pixels.sum())
    solder_px = int(solder_pixels.sum())
    total_px = max(void_px + solder_px, 1)

    void_pct = void_px / total_px * 100
    largest_void_pct = largest_area / total_px * 100

    # ================= DISPLAY (RATIO SAFE) =================

    IMG_W = 360

    st.subheader("Inspection views")

    cols = st.columns([1, 1, 1])

    cols[0].image(original, caption="Original", width=IMG_W)
    cols[1].image(overlay, caption="Detection", width=IMG_W)

    if show_heatmap:
        hm = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        cols[2].image(
            cv2.resize(hm, (w, h)),
            caption="Void probability heatmap",
            width=IMG_W
        )

    # ================= SAVE =================

    if st.button("Save inspection"):

        st.session_state.results.append({
            "void_%": round(void_pct, 2),
            "largest_void_%": round(largest_void_pct, 2),
            "IA_confidence_%": round(ai_conf * 100, 1),
            "void_pixels": void_px,
            "solder_pixels": solder_px
        })

        _, buf = cv2.imencode(".png", overlay)
        st.session_state.saved_images.append(buf.tobytes())

        st.success("Inspection saved ✔")

    # ================= RESULTS TABLE =================

    if st.session_state.results:

        st.markdown("---")
        st.subheader("Inspection history")

        df = pd.DataFrame(st.session_state.results)

        max_void = df["void_%"].max()
        min_void = df["void_%"].min()

        def highlight(row):
            if row["void_%"] == max_void:
                return ["background-color:#ffcccc"] * len(row)
            if row["void_%"] == min_void:
                return ["background-color:#cce5ff"] * len(row)
            return [""] * len(row)

        st.dataframe(df.style.apply(highlight, axis=1))

        # ================= EXPORT =================

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
            "inspection_images.zip"
        )
