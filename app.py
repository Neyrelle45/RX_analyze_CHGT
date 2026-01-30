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
# SIDEBAR — DETECTION (INDUSTRIAL)
# =====================================================

st.sidebar.header("Detection")

percentile = st.sidebar.slider(
    "Defect percentile (adaptive threshold)",
    60,
    95,
    82,
    1
)


# =====================================================
# SIDEBAR — MASK ALIGNMENT
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
# SIDEBAR — RX PREPROCESS (SAFE DEFAULTS)
# =====================================================

st.sidebar.header("RX preprocessing")

contrast = st.sidebar.slider(
    "Global contrast",
    0.9,
    1.8,
    1.35,
    0.05
)

clahe = st.sidebar.slider(
    "Local contrast (CLAHE)",
    1.0,
    2.8,
    2.2,
    0.1
)

void_boost = st.sidebar.slider(
    "Void enhancer",
    0.0,
    3.0,
    1.6,
    0.1
)

gamma = st.sidebar.slider(
    "Gamma micro-contrast",
    0.9,
    1.3,
    1.1,
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

    tensor, processed = preprocess_rx(
        original,
        contrast,
        clahe,
        void_boost,
        gamma
    )

    pred_mask, heatmap = predict_mask(
        model,
        tensor,
        percentile
    )

    h, w = processed.shape

    # =====================================================
    # MASK ALIGNMENT
    # =====================================================

    inspect_mask = np.ones((h, w), dtype=bool)

    if mask_file:

        mask_img = np.array(Image.open(mask_file).convert("L"))
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

    # appliquer zone inspection
    pred_mask = pred_mask & inspect_mask


    # =====================================================
    # METRICS (FIX INDUSTRIEL)
    # =====================================================

    void_pixels = np.sum(pred_mask)

    solder_pixels = np.sum(
        (~pred_mask) & inspect_mask
    )

    total = void_pixels + solder_pixels

    manque_pct = (void_pixels / total * 100) if total > 0 else 0


    # =====================================================
    # OVERLAY INDUSTRIEL
    # =====================================================

    overlay = cv2.resize(original, (w, h)).copy()

    # solder = jaune
    overlay[(~pred_mask) & inspect_mask] = [255, 230, 0]

    # void = rouge
    overlay[pred_mask] = [255, 0, 0]


    # =====================================================
    # DISPLAY (HEATMAP FIXED SIZE)
    # =====================================================

    col1, col2, col3 = st.columns(3)

    col1.image(original, caption="Original", use_container_width=True)

    mask_vis = original.copy()
    mask_vis = cv2.resize(mask_vis, (w, h))
    mask_vis[inspect_mask] = [0,255,0]

    col2.image(mask_vis, caption="Mask aligned", use_container_width=True)

    col3.image(
        overlay,
        caption="Detection — RED=void | YELLOW=solder",
        use_container_width=True
    )


    if show_heatmap:

        heatmap_vis = (heatmap * 255).astype(np.uint8)

        st.image(
            heatmap_vis,
            caption="Defect heatmap",
            use_container_width=True
        )


    # =====================================================
    # TABLE
    # =====================================================

    df = pd.DataFrame([{
        "manque_%": round(manque_pct,2),
        "pixels_defaut": int(void_pixels),
        "pixels_soudure": int(solder_pixels)
    }])

    st.dataframe(df)


    # =====================================================
    # SAVE
    # =====================================================

    if st.button("Save inspection"):

        st.session_state.results.append(df.iloc[0])

        _, buffer = cv2.imencode(".png", overlay)
        st.session_state.saved_images.append(buffer.tobytes())

        st.success("Inspection saved ✔")


    if st.session_state.results:

        results_df = pd.DataFrame(st.session_state.results)

        csv = results_df.to_csv(index=False).encode()

        st.download_button(
            "Download CSV",
            csv,
            "inspection_results.csv"
        )

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as z:
            for i, img in enumerate(st.session_state.saved_images):
                z.writestr(f"inspection_{i}.png", img)

        st.download_button(
            "Download images ZIP",
            zip_buffer.getvalue(),
            "inspections.zip"
        )
