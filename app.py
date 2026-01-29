import streamlit as st
import cv2
import yaml
import numpy as np
import pandas as pd

from engine.preprocessing import preprocess_rx, TARGET_SIZE
from engine.inference import load_model, predict
from engine.postprocessing import classify_defects
from engine.metrics import compute_metrics

st.set_page_config(layout="wide")
st.title("RX Void Analyzer")

# ---------------- SIDEBAR ----------------

with st.sidebar:

    model_file = st.file_uploader("Model (.pth)")
    cfg_file = st.file_uploader("config.yaml")
    mask_file = st.file_uploader("Mask")

    st.subheader("Detection")

    defect_th = st.slider("Defect threshold", 0.1, 0.9, 0.35)

    st.subheader("Mask alignment")

    tx = st.slider("Translate X", -200, 200, 0)
    ty = st.slider("Translate Y", -200, 200, 0)
    scale = st.slider("Scale", 0.5, 1.5, 1.0)
    angle = st.slider("Rotation", -10, 10, 0)

    st.subheader("Preprocess")

    contrast = st.slider("Contrast", 0.5, 3.0, 1.0)
    denoise = st.slider("Denoise", 0, 20, 5)

    show_heatmap = st.checkbox("Show heatmap")

if not(model_file and cfg_file and mask_file):
    st.stop()

cfg = yaml.safe_load(cfg_file)
model = load_model(model_file)

# ---------------- MASK ----------------

mask = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), 1)
mask = cv2.resize(mask, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)

center = (TARGET_SIZE//2, TARGET_SIZE//2)

M = cv2.getRotationMatrix2D(center, angle, scale)
M[:,2] += [tx, ty]

mask = cv2.warpAffine(mask, M, (TARGET_SIZE, TARGET_SIZE), flags=cv2.INTER_NEAREST)
inspect_mask = mask[:,:,1] > 200

# ---------------- IMAGE ----------------

img_file = st.file_uploader("RX image")

if img_file:

    original = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), 0)

    img_p, scale, shape = preprocess_rx(original, contrast, denoise)

    pred, heat = predict(model, img_p, defect_th)

    solder = (pred==1) & inspect_mask
    defect = (pred==2) & inspect_mask

    voids, lacks = classify_defects(defect, solder, inspect_mask, cfg)

    metrics = compute_metrics(voids, lacks, inspect_mask.sum())

    # overlay
    overlay = cv2.cvtColor(img_p, cv2.COLOR_GRAY2BGR)

    blue = np.zeros_like(overlay)
    blue[solder] = [180,0,0]
    overlay = cv2.addWeighted(overlay,1,blue,0.35,0)

    red = np.zeros_like(overlay)
    red[defect] = [0,0,255]
    overlay = cv2.addWeighted(overlay,1,red,0.85,0)

    col1, col2 = st.columns(2)

    col1.image(original, caption="Original (ratio preserved)")
    col2.image(overlay, caption="Analysis")

    if show_heatmap:
        st.image(heat, clamp=True, caption="Defect heatmap")

    st.dataframe(pd.DataFrame([metrics]))

