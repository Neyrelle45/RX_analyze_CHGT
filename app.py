import streamlit as st
import numpy as np
import cv2
import pandas as pd
import zipfile
import io
from PIL import Image

from engine.preprocessing import preprocess_rx
from engine.inference import load_model, predict_mask


st.set_page_config(layout="wide")
st.title("RX Void Analyzer — Industrial Edition")


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
    "Load inspection mask",
    type=["png", "jpg"]
)


model = None
if model_file:
    model = load_model(model_file)


# =====================================================
# SIDEBAR — DETECTION
# =====================================================

st.sidebar.header("Detection")

threshold = st.sidebar.slider(
    "Defect threshold",
    0.05,
    0.6,
    0.25,
    0.01
)

dominance = st.sidebar.slider(
    "Defect vs solder dominance",
    0.5,
    2.0,
    1.0,
    0.05
)


# =====================================================
# SIDEBAR — MASK ALIGNMENT (FINESSE)
# =====================================================

st.sidebar.header("Mask alignment")

tx = st.sidebar.slider("Translate X", -10, 10, 0, 1)
ty = st.sidebar.slider("Translate Y", -10, 10, 0, 1)

scale = st.sidebar.slider(
    "Scale",
    0.97,
    1.03,
    1.0,
    0.002
)

angle = st.sidebar.slider(
    "Rotation",
    -2.0,
    2.0,
    0.0,
    0.1
)


# =====================================================
# SIDEBAR — RX PREPROCESSING
# =====================================================

st.sidebar.header("RX preprocessing")

contrast = st.sidebar.slider(
    "Global contrast",
    0.8,
    2.2,
    1.4,
    0.05
)

clahe = st.sidebar.slider(
    "Local contrast (CLAHE)",
    1.0,
    4.0,
    2.5,
    0.1
)

void_boost = st.sidebar.slider(
    "Void enhancer",
    0.0,
    4.0,
    2.0,
    0.1
)

gamma = st.sidebar.slider(
    "Gamma micro-contrast",
    0.8,
    1.6,
    1.15,
    0.05
)

show_heatmap = st.sidebar.checkbox("Show defect heatmap", value=True)


# =====================================================
# IMAGE INPUT
# =====================================================

uploaded = st.file_uploader(
    "RX image",
    type=["png", "jpg", "jpeg"]
)

if uploaded and model:

    original = np.array(Image.open(uploaded).convert("RGB"))

    # PREPROCESS
    tensor, processed = preprocess_rx(
        original,
        contrast,
        clahe,
        void_boost,
        gamma
    )

    # PREDICT
    pred_mask, heatmap = predict_mask(
        model,
        tensor,
        threshold
    )

    # =====================================================
    # MASK ALIGNMENT
    # =====================================================

    inspect_mask = None

    if mask_file:
        mask_img = np.array(Image.open(mask_file).convert("L"))

        h, w = processed.shape

        mask_img = cv2.resize(mask_img, (w, h))

        M = cv2.getRotationMatrix2D(
            (w//2, h//2),
            angle,
            scale
        )

        M[:, 2] += [tx, ty]

        aligned = cv2.warpAffine(
            mask_img,
            M,
            (w, h)
        )

        inspect_mask = aligned > 127

        pred_mask = pred_mask & inspect_mask


    # =====================================================
    # COLORS
    # =====================================================

    overlay = original.copy()

    overlay = cv2.resize(
        overlay,
        (processed.shape[1], processed.shape[0])
    )

    void_pixels = pred_mask
    solder_pixels = (~pred_mask)

    if inspect_mask is not None:
        solder_pixels &= inspect_mask

    # VOID = RED
    overlay[void_pixels] = [255, 0, 0]

    # SOLDER = YELLOW
    overlay[solder_pixels] = [255, 255, 0]


    # =====================================================
    # METRICS
    # =====================================================

    void_count = np.sum(void_pixels)
    solder_count = np.sum(solder_pixels)

    denom = void_count + solder_count

    manque_pct = (void_count / denom * 100) if denom > 0 else 0


    # =====================================================
    # DISPLAY
    # =====================================================

    col1, col2, col3 = st.columns(3)

    col1.image(original, caption="Original")

    if inspect_mask is not None:
        mask_vis = original.copy()
        mask_vis = cv2.resize(mask_vis, (processed.shape[1], processed.shape[0]))
        mask_vis[inspect_mask] = [0,255,0]

        col2.image(mask_vis, caption="Mask aligned")

    col3.image(overlay, caption="Detection — RED=void | YELLOW=solder")


    if show_heatmap:
        st.image(
            (heatmap * 255).astype(np.uint8),
            caption="Defect heatmap"
        )


    # =====================================================
    # TABLE
    # =====================================================

    df = pd.DataFrame([{
        "manque_%": round(manque_pct,2),
        "pixels_defaut": int(void_count),
        "pixels_soudure": int(solder_count)
    }])

    st.dataframe(df)


    # =====================================================
    # SAVE INSPECTION
    # =====================================================

    if st.button("Save inspection"):

        st.session_state.results.append(df.iloc[0])

        _, buffer = cv2.imencode(".png", overlay)
        st.session_state.saved_images.append(buffer.tobytes())

        st.success("Inspection saved ✔")


    # =====================================================
    # EXPORT CSV
    # =====================================================

    if st.session_state.results:

        results_df = pd.DataFrame(st.session_state.results)

        csv = results_df.to_csv(index=False).encode()

        st.download_button(
            "Download CSV",
            csv,
            "inspection_results.csv"
        )


        # ZIP images
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as z:
            for i, img in enumerate(st.session_state.saved_images):
                z.writestr(f"inspection_{i}.png", img)

        st.download_button(
            "Download images ZIP",
            zip_buffer.getvalue(),
            "inspections.zip"
        )

