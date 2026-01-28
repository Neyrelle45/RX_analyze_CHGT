import streamlit as st
import cv2
import yaml
import torch
import numpy as np
import pandas as pd
import io
import zipfile

from engine.model import UNet
from engine.preprocessing import preprocess_rx
from engine.postprocessing import classify_defects
from engine.metrics import compute_metrics

# =========================
# CONFIG GLOBALE
# =========================
DEVICE = "cpu"
st.set_page_config(layout="wide")
st.title("🔍 Analyse RX – Void & Manques (IA Générique)")

# =========================
# SESSION STATE
# =========================
for k in ["results", "images", "overlays", "names"]:
    if k not in st.session_state:
        st.session_state[k] = []

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Configuration")

    model_file = st.file_uploader("📦 Modèle IA (.pth)", type=["pth"])
    config_file = st.file_uploader("📄 Config métier (.yaml)", type=["yaml","yml"])
    mask_file  = st.file_uploader("🎯 Masque inspection (unique)", type=["png"])

    st.subheader("🔧 Ajustement masque")
    tx = st.slider("Translation X (px)", -500, 500, 0)
    ty = st.slider("Translation Y (px)", -500, 500, 0)
    angle = st.slider("Rotation (°)", -180, 180, 0)
    sx = st.slider("Échelle X", 0.3, 2.0, 1.0, 0.01)
    sy = st.slider("Échelle Y", 0.3, 2.0, 1.0, 0.01)

    st.subheader("🎚️ Prétraitement RX")
    contrast = st.slider("Contraste", 0.5, 3.0, 1.0, 0.1)
    denoise  = st.slider("Réduction bruit", 0, 20, 5)

    if st.button("🔄 RESET COMPLET"):
        for k in st.session_state:
            st.session_state[k] = []
        st.experimental_rerun()

# =========================
# VERIFICATIONS
# =========================
if not (model_file and config_file and mask_file):
    st.info("⬅️ Charge un modèle, un config.yaml et un masque pour démarrer.")
    st.stop()

# =========================
# LOAD CONFIG & MODEL
# =========================
cfg = yaml.safe_load(config_file)

@st.cache_resource
def load_model(weights):
    model = UNet()
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.eval()
    return model

model = load_model(model_file)

# =========================
# LOAD & TRANSFORM MASK
# =========================
mask_raw = cv2.imdecode(
    np.frombuffer(mask_file.read(), np.uint8),
    cv2.IMREAD_COLOR
)

h, w = mask_raw.shape[:2]
center = (w // 2, h // 2)

M = cv2.getRotationMatrix2D(center, angle, 1.0)
M[0, 0] *= sx
M[1, 1] *= sy
M[0, 2] += tx
M[1, 2] += ty

mask_warp = cv2.warpAffine(mask_raw, M, (w, h))
inspect_mask = mask_warp[:, :, 1] > 200

if inspect_mask.sum() == 0:
    st.error("❌ Le masque ajusté ne contient aucune zone verte exploitable.")
    st.stop()

# =========================
# UPLOAD IMAGE RX
# =========================
st.subheader("📥 Charger une image RX")
img_file = st.file_uploader("Image RX (.png / .jpg)", type=["png","jpg","jpeg"])

if img_file:
    img = cv2.imdecode(
        np.frombuffer(img_file.read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        st.error("❌ Impossible de lire l’image RX.")
        st.stop()

    # --- Prétraitement ---
    img_p = preprocess_rx(img, contrast, denoise)
    img_n = img_p.astype("float32") / 255.0
    t = torch.from_numpy(img_n).unsqueeze(0).unsqueeze(0)

    # --- Inférence ---
    with torch.no_grad():
        out = model(t)
        pred = torch.argmax(out, dim=1)[0].cpu().numpy()

    # --- DEBUG IA ---
    st.write("🧪 Pixels par classe :", {
        "fond": int((pred == 0).sum()),
        "soudure": int((pred == 1).sum()),
        "defaut": int((pred == 2).sum())
    })

    # --- Masques ---
    solder_mask = (pred == 1) & inspect_mask
    defect_mask = (pred == 2) & inspect_mask

    # --- Post-processing ---
    voids, lacks = classify_defects(defect_mask, solder_mask, inspect_mask, cfg)
    metrics = compute_metrics(voids, lacks, solder_mask.sum())
    metrics["image"] = img_file.name

    # --- Overlay ---
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay[solder_mask] = [120, 0, 0]   # soudure
    overlay[defect_mask] = [0, 0, 255]   # défaut

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
    st.session_state.images.append(img)
    st.session_state.overlays.append(overlay)
    st.session_state.results.append(metrics)
    st.session_state.names.append(img_file.name)

# =========================
# AFFICHAGE PRINCIPAL
# =========================
if st.session_state.images:
    idx = len(st.session_state.images) - 1

    col1, col2, col3 = st.columns(3)
    col1.image(st.session_state.images[idx], caption="Original", clamp=True)
# --- VISU masque en transparence ---
mask_overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

alpha = 0.4
green = np.zeros_like(mask_overlay)
green[inspect_mask] = [0, 255, 0]

mask_overlay = cv2.addWeighted(mask_overlay, 1.0, green, alpha, 0)

col2.image(mask_overlay, caption="Masque (overlay)", clamp=True)
    col3.image(st.session_state.overlays[idx], caption="Analyse IA", clamp=True)

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
    st.subheader("🖼️ Historique analyses")
    cols = st.columns(min(5, len(st.session_state.overlays)))
    for i, col in enumerate(cols):
        col.image(
            st.session_state.overlays[i],
            caption=st.session_state.names[i],
            width=150
        )

# =========================
# EXPORT
# =========================
if st.session_state.results:
    st.subheader("📦 Export")

    # CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Télécharger résultats CSV",
        csv_bytes,
        "resultats_rx.csv"
    )

    # ZIP images
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as z:
        for i, overlay in enumerate(st.session_state.overlays):
            _, buf = cv2.imencode(".png", overlay)
            z.writestr(f"{st.session_state.names[i]}", buf.tobytes())

    st.download_button(
        "📥 Télécharger images analysées (ZIP)",
        zip_buffer.getvalue(),
        "images_analysees.zip"
    )

