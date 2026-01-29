import streamlit as st
import cv2
import yaml
import numpy as np
import pandas as pd

from engine.preprocessing import preprocess_rx, TARGET_SIZE
from engine.inference import load_model, predict
from engine.postprocessing import classify_defects

st.set_page_config(layout="wide")
st.title("RX Void Analyzer")

# ---------------- SIDEBAR ----------------

with st.sidebar:

    model_file = st.file_uploader("Model (.pth)")
    cfg_file = st.file_uploader("config.yaml")
    mask_file = st.file_uploader("Mask")

    st.subheader("Detection")

    defect_th = st.slider("Defect threshold", 0.10, 0.60, 0.30)

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

# ---------------- LOAD IMAGE ----------------

img_file = st.file_uploader("RX image")

if img_file:

    original = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), 0)

    # preprocessing réseau
    img_net, scale_factor, shape = preprocess_rx(original, contrast, denoise)

    # ---------------- MASK (SUR IMAGE RESEAU !) ----------------

    mask = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), 1)
    mask = cv2.resize(mask, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)

    center = (TARGET_SIZE//2, TARGET_SIZE//2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[:,2] += [tx, ty]

    mask = cv2.warpAffine(mask, M, (TARGET_SIZE, TARGET_SIZE), flags=cv2.INTER_NEAREST)
    inspect_mask = mask[:,:,1] > 200

    # ---------------- IA ----------------

    pred, heat = predict(model, img_net, defect_th)

    solder = (pred==1) & inspect_mask
    defect = (pred==2) & inspect_mask

    # ---------------- METRICS PRO ----------------

    red_pixels = np.sum(defect)
    blue_pixels = np.sum(solder)

    metal = red_pixels + blue_pixels
    lack_ratio = (red_pixels / metal * 100) if metal > 0 else 0

    metrics = {
        "manque_%": round(lack_ratio,2),
        "pixels_rouges": int(red_pixels),
        "pixels_bleus": int(blue_pixels)
    }

    # ---------------- VISU MASQUE ----------------

    mask_overlay = cv2.cvtColor(img_net, cv2.COLOR_GRAY2BGR)

    green = np.zeros_like(mask_overlay)
    green[inspect_mask] = [0,255,0]

    mask_overlay = cv2.addWeighted(mask_overlay,1,green,0.35,0)

    # ---------------- VISU ANALYSE ----------------

    overlay = cv2.cvtColor(img_net, cv2.COLOR_GRAY2BGR)

    blue_layer = np.zeros_like(overlay)
    blue_layer[solder] = [180,0,0]   # BLEU

    red_layer = np.zeros_like(overlay)
    red_layer[defect] = [0,0,255]    # ROUGE

    overlay = cv2.addWeighted(overlay,1,blue_layer,0.35,0)
    overlay = cv2.addWeighted(overlay,1,red_layer,0.85,0)

    # ---------------- DISPLAY 3 IMAGES ----------------

    col1, col2, col3 = st.columns(3)

    col1.image(original, caption="Original (ratio conservé)")

    col2.image(mask_overlay, caption="Masque ajusté (image réseau)")

    col3.image(overlay, caption="Analyse : bleu=soudure | rouge=manque")

    if show_heatmap:
        st.image(heat, clamp=True, caption="Heatmap défaut")

    st.dataframe(pd.DataFrame([metrics]))


