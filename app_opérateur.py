import streamlit as st
import numpy as np
import cv2
import pandas as pd
import zipfile
import io
import torch
import time
from PIL import Image

from engine.preprocessing import preprocess_rx
from engine.inference import load_model, find_largest_void

# =====================================================
# CONFIG
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

if "operator_corrections" not in st.session_state:
    st.session_state.operator_corrections = None

# =====================================================
# SIDEBAR — MODEL
# =====================================================
st.sidebar.header("Model")

model_file = st.sidebar.file_uploader("Load model (.pth)", type=["pth"])
mask_file  = st.sidebar.file_uploader("Inspection mask", type=["png","jpg","jpeg"])

model = load_model(model_file) if model_file else None
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# SIDEBAR — DETECTION
# =====================================================
st.sidebar.header("Detection")

threshold = st.sidebar.slider("Void threshold", 0.05, 0.6, 0.25, 0.01)
temperature = st.sidebar.slider("Softmax temperature", 0.8, 2.5, 1.6, 0.05)

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

contrast = st.sidebar.slider("Global contrast", 1.0, 2.2, 1.6, 0.05)
clahe    = st.sidebar.slider("Local contrast", 1.0, 4.0, 2.2, 0.1)
gamma    = st.sidebar.slider("Gamma", 0.8, 1.6, 1.1, 0.05)

show_heatmap = st.sidebar.checkbox("Show void probability heatmap", True)

# =====================================================
# SIDEBAR — OPERATOR REVIEW
# =====================================================
st.sidebar.header("Operator review")

review_mode = st.sidebar.checkbox("Enable operator review", False)
review_action = None

if review_mode:
    review_action = st.sidebar.radio(
        "Correction type",
        ["Force VOID", "Force SOLDER", "Ignore zone"]
    )

# =====================================================
# IMAGE INPUT
# =====================================================
uploaded = st.file_uploader("RX image", type=["png","jpg","jpeg"])

if uploaded and model:

    # -----------------------------
    # LOAD IMAGE
    # -----------------------------
    original = np.array(Image.open(uploaded).convert("RGB"))
    H0, W0 = original.shape[:2]

    # -----------------------------
    # PREPROCESS
    # -----------------------------
    tensor, processed = preprocess_rx(original, contrast, clahe, gamma)
    tensor = tensor.to(device)

    h, w = processed.shape

    # -----------------------------
    # INIT OPERATOR MASKS
    # -----------------------------
    if st.session_state.operator_corrections is None:
        st.session_state.operator_corrections = {
            "force_void": np.zeros((h, w), dtype=bool),
            "force_solder": np.zeros((h, w), dtype=bool),
            "ignore": np.zeros((h, w), dtype=bool),
        }

    # -----------------------------
    # MODEL INFERENCE (ONCE)
    # -----------------------------
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits / temperature, dim=1)
        pred = torch.argmax(probs, dim=1)[0].cpu().numpy()
        void_prob = probs[0, 2].cpu().numpy()

    # -----------------------------
    # BASE MASKS
    # -----------------------------
    solder_mask = (pred == 1)
    void_mask   = (pred == 2)
    copper_mask = (pred == 3)

    # -----------------------------
    # INSPECTION MASK
    # -----------------------------
    inspect_mask = np.ones((h, w), dtype=bool)

    if mask_file:
        mask_img = np.array(Image.open(mask_file).convert("L"))
        mask_img = cv2.resize(mask_img, (w, h))

        M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)
        M[:, 2] += [tx, ty]

        aligned = cv2.warpAffine(mask_img, M, (w, h))
        inspect_mask = aligned > 127

        solder_mask &= inspect_mask
        void_mask   &= inspect_mask
        copper_mask &= inspect_mask

    # -----------------------------
    # APPLY OPERATOR CORRECTIONS
    # -----------------------------
    oc = st.session_state.operator_corrections

    void_mask[oc["force_void"]] = True
    solder_mask[oc["force_void"]] = False

    solder_mask[oc["force_solder"]] = True
    void_mask[oc["force_solder"]] = False

    void_mask[oc["ignore"]] = False
    solder_mask[oc["ignore"]] = False

    # -----------------------------
    # LARGEST VOID
    # -----------------------------
    largest_mask, largest_area, ai_conf = find_largest_void(
        void_mask, void_prob, inspect_mask
    )

    inspect_area = inspect_mask.sum()
    largest_void_pct = (largest_area / inspect_area * 100) if inspect_area else 0

    # -----------------------------
    # OVERLAY
    # -----------------------------
    overlay = cv2.resize(original, (w, h))

    overlay[solder_mask] = [255, 255, 0]
    overlay[void_mask]   = [255, 0, 0]
    overlay[copper_mask] = [0, 0, 255]

    if largest_mask is not None:
        ys, xs = np.where(largest_mask)
        cy, cx = int(ys.mean()), int(xs.mean())
        r = int(np.sqrt(largest_area / np.pi))
        cv2.circle(overlay, (cx, cy), r, (135, 206, 235), 3)

    overlay_disp = cv2.resize(overlay, (W0, H0))

    # =====================================================
    # DISPLAY
    # =====================================================
    col1, col2, col3 = st.columns(3)

    col1.image(original, caption="Original", use_container_width=True)

    mask_vis = cv2.resize(original, (w, h))
    mask_vis[inspect_mask] = [0, 255, 0]
    col2.image(cv2.resize(mask_vis, (W0, H0)), caption="Mask aligned", use_container_width=True)

    col3.image(
        overlay_disp,
        caption="Detection — YELLOW=solder | RED=void | BLUE=copper",
        use_container_width=True
    )

    if show_heatmap:
        hm = (void_prob * 255).astype(np.uint8)
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        hm_disp = cv2.resize(hm, (int(W0 * 0.6), int(H0 * 0.6)))
        st.image(hm_disp, caption="Void probability heatmap")

    # =====================================================
    # METRICS (LIVE)
    # =====================================================
    void_count   = int(void_mask.sum())
    solder_count = int(solder_mask.sum())

    manque_pct = (void_count / (void_count + solder_count) * 100) if (void_count + solder_count) else 0

    metrics = {
        "void_%": round(manque_pct, 2),
        "largest_void_%": round(largest_void_pct, 2),
        "IA_confidence_%": round(ai_conf * 100, 1),
        "void_pixels": void_count,
        "solder_pixels": solder_count
    }

    st.subheader("Live metrics")
    st.json(metrics)

    # =====================================================
    # OPERATOR CLICK CORRECTION
    # =====================================================
    if review_mode:
        click = st.image(
            overlay_disp,
            caption="Click to correct (operator review)",
            use_container_width=True
        )

        if click is not None and review_action is not None:
            cx = int(click["x"] * w / W0)
            cy = int(click["y"] * h / H0)

            r = 12
            yy, xx = np.ogrid[:h, :w]
            disk = (xx - cx)**2 + (yy - cy)**2 <= r*r

            if review_action == "Force VOID":
                oc["force_void"] |= disk
            elif review_action == "Force SOLDER":
                oc["force_solder"] |= disk
            elif review_action == "Ignore zone":
                oc["ignore"] |= disk

    # =====================================================
    # SAVE INSPECTION
    # =====================================================
    if st.button("Save inspection"):
        st.session_state.results.append(metrics)
        _, buf = cv2.imencode(".png", overlay_disp)
        st.session_state.saved_images.append(buf.tobytes())
        st.success("Inspection saved ✔")

# =====================================================
# HISTORY + EXPORT
# =====================================================
if st.session_state.results:
    st.subheader("Inspection history")
    df = pd.DataFrame(st.session_state.results)
    st.dataframe(df)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode(),
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

    if st.button("Clear history"):
        st.session_state.results.clear()
        st.session_state.saved_images.clear()
        st.session_state.operator_corrections = None
        st.success("History cleared ✔")

