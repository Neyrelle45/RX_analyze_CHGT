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
st.title("RX Void Analyzer — Industrial Edition")


# =====================================================
# SESSION STATE
# =====================================================

if "results" not in st.session_state:
    st.session_state.results = []

if "saved_images" not in st.session_state:
    st.session_state.saved_images = []


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Model")

model_file = st.sidebar.file_uploader("Load model (.pth)", type=["pth"])
mask_file = st.sidebar.file_uploader("Load inspection mask", type=["png", "jpg"])

@st.cache_resource
def get_model(model_file):
    return load_model(model_file)

model = get_model(model_file) if model_file else None


st.sidebar.header("Detection")

threshold = st.sidebar.slider("Void threshold", 0.05, 0.6, 0.25, 0.01)


st.sidebar.header("Mask alignment")

tx = st.sidebar.slider("Translate X", -80, 80, 0, 1)
ty = st.sidebar.slider("Translate Y", -80, 80, 0, 1)

scale = st.sidebar.slider("Scale", 0.95, 1.05, 1.0, 0.001)
angle = st.sidebar.slider("Rotation", -3.0, 3.0, 0.0, 0.1)


st.sidebar.header("RX preprocessing")

contrast = st.sidebar.slider("Global contrast", 1.0, 2.2, 1.6, 0.05)
clahe = st.sidebar.slider("Local contrast", 1.0, 4.0, 2.2, 0.1)
gamma = st.sidebar.slider("Gamma", 0.8, 1.6, 1.1, 0.05)

show_heatmap = st.sidebar.checkbox("Show defect heatmap", True)


# =====================================================
# IMAGE INPUT
# =====================================================

uploaded = st.file_uploader("RX image", type=["png", "jpg", "jpeg"])


if uploaded and model:

    original = np.array(Image.open(uploaded).convert("RGB"))

    tensor, processed = preprocess_rx(original, contrast, clahe, gamma)
    pred_mask, heatmap = predict_mask(model, tensor, threshold)

    h, w = processed.shape

    # masque inspection par défaut
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
    # LARGEST VOID (APRES MASK !!)
    # =====================================================

    largest_void_mask, largest_void_area, ai_conf = find_largest_void(
        pred_mask,
        heatmap,
        inspect_mask
    )

    # =====================================================
    # OVERLAY (BGR SAFE)
    # =====================================================

    overlay = cv2.cvtColor(cv2.resize(original, (w, h)), cv2.COLOR_RGB2BGR)

    void_pixels = pred_mask
    solder_pixels = (~pred_mask) & inspect_mask

    overlay[void_pixels] = (0, 0, 255)       # rouge BGR
    overlay[solder_pixels] = (0, 255, 255)   # jaune BGR

    # cercle largest void
    if largest_void_mask is not None:

        coords = np.column_stack(np.where(largest_void_mask))
        y, x = coords.mean(axis=0).astype(int)
        radius = int(np.sqrt(largest_void_area / np.pi))

        cv2.circle(
            overlay,
            (x, y),
            radius,
            (255, 255, 0),   # bleu ciel BGR
            4
        )

    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # =====================================================
    # METRICS
    # =====================================================

    void_count = np.sum(void_pixels)
    solder_count = np.sum(solder_pixels)

    denom = void_count + solder_count
    manque_pct = (void_count / denom * 100) if denom > 0 else 0

    df = {
        "void_%": round(manque_pct, 2),
        "largest_void_px": int(largest_void_area),
        "IA_confidence_%": round(ai_conf * 100, 1),
        "void_pixels": int(void_count),
        "solder_pixels": int(solder_count)
    }

    # =====================================================
    # DISPLAY
    # =====================================================

    col1, col2, col3 = st.columns(3)

    col1.image(original, caption="Original", use_container_width=True)

    mask_vis = original.copy()
    mask_vis = cv2.resize(mask_vis, (w, h))
    mask_vis[inspect_mask] = [0,255,0]

    col2.image(mask_vis, caption="Mask aligned", use_container_width=True)

    col3.image(overlay, caption="Detection", use_container_width=True)

    if show_heatmap:

        heatmap_vis = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        st.image(heatmap_vis, use_container_width=True)

    # =====================================================
    # SAVE
    # =====================================================

    if st.button("Save inspection"):

        st.session_state.results.append(df)

        _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        st.session_state.saved_images.append(buffer.tobytes())

        st.success("Inspection saved ✔")


# =====================================================
# HISTORY (TOUJOURS EN BAS)
# =====================================================

st.divider()
st.subheader("Inspection History")

if len(st.session_state.results) == 0:

    st.info("No inspection saved yet.")

else:

    results_df = pd.DataFrame(st.session_state.results)

    def highlight_rows(row):

        if row["void_%"] == results_df["void_%"].max():
            return ['background-color: #ff4b4b'] * len(row)

        if row["void_%"] == results_df["void_%"].min():
            return ['background-color: #4b8bff'] * len(row)

        return [''] * len(row)

    st.dataframe(
        results_df.style.apply(highlight_rows, axis=1),
        use_container_width=True
    )

    csv = results_df.to_csv(index=False).encode()

    st.download_button("Download CSV", csv, "inspection_results.csv")

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as z:
        for i, img in enumerate(st.session_state.saved_images):
            z.writestr(f"inspection_{i}.png", img)

    st.download_button("Download images ZIP", zip_buffer.getvalue(), "inspections.zip")
