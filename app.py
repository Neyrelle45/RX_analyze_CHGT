import streamlit as st
import cv2
import yaml
import torch
import numpy as np
import pandas as pd
import io
import zipfile
from engine.model import UNetLite
from engine.preprocessing import preprocess_rx
from engine.postprocessing import classify_defects
from engine.metrics import compute_metrics

# =========================
# CONFIG
# =========================
DEVICE = "cpu"
st.set_page_config(layout="wide")
st.title("🔍 Analyse RX – Générique (Void & Manque)")

# =========================
# SESSION STATE
# =========================
if "results" not in st.session_state:
    st.session_state.results = []

if "images" not in st.session_state:
    st.session_state.images = []

if "overlays" not in st.session_state:
    st.session_state.overlays = []

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Configuration")

    model_file = st.file_uploader("📦 Modèle IA (.pth)", type=["pth"])
    config_file = st.file_uploader("📄 Config métier (.yaml)", type=["yaml","yml"])
    mask_file = st.file_uploader("🎯 Masque inspection (unique)", type=["png"])

    contrast = st.slider("Contraste", 0.5, 3.0, 1.0, 0.1)
    denoise = st.slider("Réduction bruit", 0, 20, 5)

    if st.button("🔄 RESET analyse"):
        st.session_state.results = []
        st.session_state.images = []
        st.session_state.overlays = []
        st.experimental_rerun()

# =========================
# CHARGEMENT MODELE
# =========================
@st.cache_resource
def load_model(weights):
    model = UNetLite()
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.eval()
    return model

# =========================
# VERIFICATIONS
# =========================
if not (model_file and config_file and mask_file):
    st.info("⬅️ Charge un modèle, un config.yaml et un masque pour commencer.")
    st.stop()

# =========================
# LECTURE CONFIG & MASQUE
# =========================
cfg = yaml.safe_load(config_file)

mask_bytes = np.frombuffer(mask_file.read(), np.uint8)
mask_img = cv2.imdecode(mask_bytes, cv2.IMREAD_COLOR)

if mask_img is None:
    st.error("❌ Impossible de lire le masque.")
    st.stop()

inspect_mask = mask_img[:,:,1] > 200

if inspect_mask.sum() == 0:
    st.error("❌ Le masque ne contient aucune zone verte exploitable.")
    st.stop()

model = load_model(model_file)

# =========================
# UPLOAD IMAGE RX
# =========================
st.subheader("📥 Charger une image RX")
img_file = st.file_uploader("Image RX (.png / .jpg)", type=["png","jpg","jpeg"])

if img_file:
    # --- Lecture image ---
    img_bytes = np.frombuffer(img_file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        st.error("❌ Impossible de lire l’image RX.")
        st.stop()

    # --- Prétraitement ---
    img_p = preprocess_rx(img, contrast, denoise)
    img_n = img_p.astype("float32") / 255.0
    t = torch.from_numpy(img_n).unsqueeze(0).unsqueeze(0)

    # --- Inférence ---
    with torch.no_grad():
        pred = torch.argmax(model(t), dim=1)[0].cpu().numpy()

    # --- Masques ---
    solder_mask = (pred == 1) & inspect_mask
    defect_mask = (pred == 2) & inspect_mask

    # --- Post-processing ---
    voids, lacks = classify_defects(defect_mask, solder_mask, inspect_mask, cfg)
    metrics = compute_metrics(voids, lacks, solder_mask.sum())
    metrics["image"] = img_file.name

    # --- Overlay ---
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay[solder_mask] = [120, 0, 0]
    overlay[defect_mask] = [0, 0, 255]

    if voids:
        v = max(voids, key=lambda r: r.area)
        y, x = v.centroid
        cv2.circle(
            overlay,
            (int(x), int(y)),
            int(v.equivalent_diameter / 2),
            (255, 200, 0),
            4
        )

    # --- Sauvegarde session ---
    st.session_state.results.append(metrics)
    st.session_state.images.append(img)
    st.session_state.overlays.append(overlay)

# =========================
# AFFICHAGE PRINCIPAL
# =========================
if st.session_state.images:
    idx = len(st.session_state.images) - 1

    col1, col2 = st.columns(2)
    col1.image(st.session_state.images[idx], caption="Image originale", clamp=True)
    col2.image(st.session_state.overlays[idx], caption="Image analysée", clamp=True)

# =========================
# TABLE RESULTATS
# =========================
if st.session_state.results:
    st.subheader("📊 Résultats cumulés")
    df = pd.DataFrame(st.session_state.results)
    st.dataframe(df)

# =========================
# VIGNETTES
# =========================
if st.session_state.overlays:
    st.subheader("🖼️ Images analysées")
    cols = st.columns(min(5, len(st.session_state.overlays)))
    for i, col in enumerate(cols):
        col.image(st.session_state.overlays[i], caption=st.session_state.results[i]["image"], width=150)

# =========================
# EXPORT ZIP / CSV
# =========================
if st.session_state.results:
    st.subheader("📦 Export")

    # CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger résultats CSV", csv_bytes, "resultats.csv")

    # ZIP images
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as z:
        for i, overlay in enumerate(st.session_state.overlays):
            _, buf = cv2.imencode(".png", overlay)
            z.writestr(f"image_{i+1}.png", buf.tobytes())

    st.download_button("📥 Télécharger images analysées (ZIP)", zip_buffer.getvalue(), "images_analysees.zip")

