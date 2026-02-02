import streamlit as st
import numpy as np
import cv2
import pandas as pd
import zipfile
import io
from PIL import Image

from streamlit_drawable_canvas import st_canvas

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
    type=["png", "jpg"]
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
    0.8, 2.5, 1.6, 0.05
)


# =====================================================
# SIDEBAR — MASK ALIGNMENT
# =====================================================

st.sidebar.header("Mask alignment")

tx = st.sidebar.slider("Translate X", -80, 80, 0, 1)
ty = st.sidebar.slider("Translate Y", -80, 80, 0, 1)

scale = st.sidebar.slider("Scale", 0.85, 1.15, 1.0, 0.002)
angle = st.sidebar.slider("Rotation", -3.0, 3.0, 0.0, 0.1)


# =====================================================
# SIDEBAR — RX PREPROCESSING
# =====================================================

st.sidebar.header("RX preprocessing")

contrast = st.sidebar.slider("Global contrast", 1.0, 2.2, 1.55, 0.05)
clahe = st.sidebar.slider("Local contrast", 1.0, 4.0, 2.2, 0.1)
gamma = st.sidebar.slider("Gamma", 0.8, 1.6, 1.1, 0.05)

show_heatmap = st.sidebar.checkbox("Show void probability heatmap", True)


# =====================================================
# SIDEBAR — MANUAL CORRECTION
# =====================================================

st.sidebar.header("Manual correction")

enable_eraser = st.sidebar.checkbox("🧽 Enable eraser", False)
eraser_radius = st.sidebar.slider("Eraser radius (px)", 5, 80, 25, 1)

enable_pipette = st.sidebar.checkbox("🎯 Enable pipette", False)
pipette_mode = st.sidebar.radio(
    "Pipette mode",
    ["Force VOID", "Force SOLDER"],
    horizontal=True
)
pipette_radius = st.sidebar.slider("Pipette radius (px)", 3, 40, 12, 1)


# =====================================================
# IMAGE INPUT
# =====================================================

uploaded = st.file_uploader(
    "RX image",
    type=["png", "jpg", "jpeg"]
)

if uploaded and model:

    original = np.array(Image.open(uploaded).convert("RGB"))
    H0, W0 = original.shape[:2]

    tensor, processed = preprocess_rx(
        original,
        contrast,
        clahe,
        gamma
    )

    pred_mask, heatmap = predict_mask(
        model,
        tensor,
        threshold,
        temperature
    )

    h, w = processed.shape
    inspect_mask = np.ones((h, w), dtype=bool)

    # =====================================================
    # MASK ALIGNMENT
    # =====================================================

    if mask_file:
        mask_img = np.array(Image.open(mask_file).convert("L"))
        mask_img = cv2.resize(mask_img, (w, h))

        M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)
        M[:, 2] += [tx, ty]

        aligned = cv2.warpAffine(mask_img, M, (w, h))
        inspect_mask = aligned > 127

        pred_mask &= inspect_mask


    # =====================================================
    # OVERLAY
    # =====================================================

    overlay = cv2.resize(original, (w, h))

    void_pixels = pred_mask
    solder_pixels = inspect_mask & (~pred_mask)

    overlay[void_pixels] = [255, 0, 0]      # VOID = RED
    overlay[solder_pixels] = [255, 255, 0]  # SOLDER = YELLOW


    # =====================================================
    # 🧽 ERASER
    # =====================================================

    if enable_eraser:

        canvas = st_canvas(
            fill_color="rgba(0,0,0,1)",
            stroke_width=eraser_radius,
            stroke_color="rgba(0,0,0,1)",
            background_image=Image.fromarray(overlay),
            drawing_mode="freedraw",
            height=h,
            width=w,
            key="eraser"
        )

        if canvas.image_data is not None:
            erase_mask = canvas.image_data[:, :, 3] > 0
            pred_mask &= (~erase_mask)


    # =====================================================
    # 🎯 PIPETTE
    # =====================================================

    if enable_pipette:

        canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=1,
            stroke_color="rgba(0,0,0,0)",
            background_image=Image.fromarray(overlay),
            drawing_mode="point",
            height=h,
            width=w,
            key="pipette"
        )

        if canvas.json_data is not None:
            for obj in canvas.json_data["objects"]:
                x, y = int(obj["left"]), int(obj["top"])

                rr, cc = np.ogrid[:h, :w]
                mask = (rr - y)**2 + (cc - x)**2 <= pipette_radius**2

                if pipette_mode == "Force VOID":
                    pred_mask[mask] = True
                else:
                    pred_mask[mask] = False


    # =====================================================
    # LARGEST VOID (CONTOUR)
    # =====================================================

    largest_mask, largest_area, ai_conf = find_largest_void(
        pred_mask, heatmap, inspect_mask
    )

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
            (135, 206, 235),
            4
        )


    # =====================================================
    # METRICS
    # =====================================================

    void_px = np.sum(pred_mask)
    solder_px = np.sum(inspect_mask) - void_px

    void_pct = (void_px / np.sum(inspect_mask)) * 100 if np.sum(inspect_mask) else 0
    largest_void_pct = (largest_area / np.sum(inspect_mask)) * 100 if largest_area else 0


    # =====================================================
    # DISPLAY
    # =====================================================

    col1, col2, col3 = st.columns(3)

    col1.image(original, caption="Original", use_container_width=True)

    mask_vis = cv2.resize(original, (w, h))
    mask_vis[inspect_mask] = [0, 255, 0]
    col2.image(mask_vis, caption="Inspection mask", use_container_width=True)

    col3.image(overlay, caption="Detection", use_container_width=True)

    if show_heatmap:
        heatmap_vis = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_TURBO
        )
        heatmap_vis = cv2.resize(heatmap_vis, (w, h))
        st.image(
            heatmap_vis,
            caption="Void probability heatmap",
            use_container_width=True
        )


    # =====================================================
    # TABLE
    # =====================================================

    row = {
        "void_%": round(void_pct, 2),
        "largest_void_%": round(largest_void_pct, 2),
        "IA_confidence_%": round(ai_conf * 100, 1),
        "void_pixels": int(void_px),
        "solder_pixels": int(solder_px)
    }

    df = pd.DataFrame([row])
    st.dataframe(df)


    # =====================================================
    # SAVE / EXPORT
    # =====================================================

    if st.button("Save inspection"):
        st.session_state.results.append(row)
        _, buf = cv2.imencode(".png", overlay)
        st.session_state.saved_images.append(buf.tobytes())
        st.success("Inspection saved ✔")

    if st.session_state.results:

        results_df = pd.DataFrame(st.session_state.results)

        def highlight(row):
            if row["void_%"] == results_df["void_%"].max():
                return ["background-color:#ff4b4b"] * len(row)
            if row["void_%"] == results_df["void_%"].min():
                return ["background-color:#4b8bff"] * len(row)
            return [""] * len(row)

        st.markdown("### Inspection history")
        st.dataframe(results_df.style.apply(highlight, axis=1))

        st.download_button(
            "Download CSV",
            results_df.to_csv(index=False).encode(),
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

        if st.button("🧹 Clear session"):
            st.session_state.results = []
            st.session_state.saved_images = []
            st.experimental_rerun()


