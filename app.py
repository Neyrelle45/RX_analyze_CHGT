import streamlit as st
import cv2
import yaml
import torch
import numpy as np

from engine.model import UNetLite
from engine.preprocessing import preprocess_rx
from engine.postprocessing import classify_defects
from engine.metrics import compute_metrics

# =========================
# CONFIG GLOBALE
# =========================
DEVICE = "cpu"   # Streamlit Cloud = CPU
st.set_page_config(layout="wide")
st.title("🔍 Analyse RX – Générique (BTC / BGA / QFN / PADS)")

# =========================
# UPLOADS
# =========================
model_file  = st.file_uploader("📦 Modèle IA (.pth)", type=["pth"])
config_file = st.file_uploader("⚙️ Configuration (.yaml)", type=["yaml","yml"])
img_file    = st.file_uploader("🖼️ Image RX (.png / .jpg)", type=["png","jpg","jpeg"])
mask_file   = st.file_uploader("🎯 Masque inspection (.png)", type=["png"])

st.divider()

# =========================
# PARAMETRES RX
# =========================
contrast = st.slider("Contraste", 0.5, 3.0, 1.0, 0.1)
denoise  = st.slider("Réduction bruit", 0, 20, 5)

# =========================
# CHARGEMENT MODELE
# =========================
@st.cache_resource
def load_model(weights_bytes):
    model = UNetLite()
    model.load_state_dict(torch.load(weights_bytes, map_location=DEVICE))
    model.eval()
    return model

# =========================
# PIPELINE
# =========================
if model_file and config_file and img_file and mask_file:

    # --- Chargement config ---
    try:
        cfg = yaml.safe_load(config_file)
    except Exception as e:
        st.error(f"Erreur lecture config.yaml : {e}")
        st.stop()

    # --- Chargement modèle ---
    try:
        model = load_model(model_file)
    except Exception as e:
        st.error(f"Erreur chargement modèle : {e}")
        st.stop()

    # --- Lecture image RX ---
    img_bytes = np.frombuffer(img_file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        st.error("Impossible de lire l’image RX.")
        st.stop()

    # --- Lecture masque ---
    mask_bytes = np.frombuffer(mask_file.read(), np.uint8)
    mask = cv2.imdecode(mask_bytes, cv2.IMREAD_COLOR)

    if mask is None:
        st.error("Impossible de lire le masque.")
        st.stop()

    # --- Zone inspection (vert) ---
    inspect_mask = mask[:,:,1] > 200

    if inspect_mask.sum() == 0:
        st.error("Le masque ne contient aucune zone verte exploitable.")
        st.stop()

    # =========================
    # PRETRAITEMENT
    # =========================
    img_p = preprocess_rx(img, contrast, denoise)

    # sécurité type
    img_n = img_p.astype("float32") / 255.0

    # [H,W] -> [1,1,H,W]
    t = torch.from_numpy(img_n).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # =========================
    # INFERENCE IA
    # =========================
    with torch.no_grad():
        out = model(t)
        pred = torch.argmax(out, dim=1)[0].cpu().numpy()

    # =========================
    # POST-PROCESSING
    # =========================
    solder_mask = (pred == 1) & inspect_mask
    defect_mask = (pred == 2) & inspect_mask

    voids, lacks = classify_defects(defect_mask, solder_mask, inspect_mask, cfg)
    metrics = compute_metrics(voids, lacks, solder_mask.sum())

    # =========================
    # RENDU VISUEL
    # =========================
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    overlay[solder_mask] = [120, 0, 0]   # bleu foncé
    overlay[defect_mask] = [0, 0, 255]   # rouge

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

    # =========================
    # AFFICHAGE
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image originale")
        st.image(img, clamp=True)

    with col2:
        st.subheader("Image analysée")
        st.image(overlay, clamp=True)

    st.subheader("📊 Résultats")
    st.table(metrics)

else:
    st.info("⬆️ Charge un modèle, un config.yaml, une image RX et un masque pour démarrer.")

