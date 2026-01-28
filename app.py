import streamlit as st
import cv2
import yaml
import torch
import numpy as np
import pandas as pd
import io, zipfile
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
for k in ["results", "images", "overlays"]:
    if k not in st.session_state:
        st.session_state[k] = []

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Configuration")

    model_file = st.file_uploader("📦 Modèle IA (.pth)", type=["pth"])
    config_file = st.file_uploader("📄 Config métier (.yaml)", type=["yaml","yml"])
    mask_file = st.file_uploader("🎯 Masque inspection", type=["png"])

    st.subheader("🔧 Ajustement masque")
    tx = st.slider("Translation X (px)", -300, 300, 0)
    ty = st.slider("Translation Y (px)", -300, 300, 0)
    angle = st.slider("Rotation (°)", -180, 180, 0)
    sx = st.slider("Échelle X", 0.5, 2.0, 1.0, 0.01)
    sy = st.slider("Échelle Y", 0.5, 2.0, 1.0, 0.01)

    contrast = st.slider("Contraste", 0.5, 3.0, 1.0, 0.1)
    denoise = st.slider("Réduction bruit", 0, 20, 5)

    if st.button("🔄 RESET"):
        for k in ["results", "images", "overlays"]:
            st.session_state[k] = []
        st.experimental_rerun()

# =========================
# LOAD MODEL / CONFIG
# =========================
if not (model_file and config_file and mask_file):
    st.info("⬅️ Charge modèle, config et masque.")
    st.stop()

cfg = yaml.safe_load(config_file)

@st.cache_resource
def load_model(weights):
    m = UNetLite()
    m.load_state_dict(torch.load(weights, map_location=DEVICE))
    m.eval()
    return m

model = load_model(model_file)

# =========================
# MASK TRANSFORMATION
# =========================
mask_raw = cv2.imdecode(
    np.frombuffer(mask_file.read(), np.uint8),
    cv2.IMREAD_COLOR
)

h, w = mask_raw.shape[:2]
center = (w // 2, h // 2)

# rotation + scale
M = cv2.getRotationMatrix2D(center, angle, 1.0)
M[0,0] *= sx
M[1,1] *= sy
M[0,2] += tx
M[1,2] += ty

mask_warp = cv2.warpAffine(mask_raw, M, (w, h))
inspect_mask = mask_warp[:,:,1] > 200

# =========================
# UPLOAD IMAGE RX
# =========================
st.subheader("📥 Image RX")
img_file = st.file_uploader("Image RX", type=["png","jpg","jpeg"])

if img_file:
    img = cv2.imdecode(
        np.frombuffer(img_file.read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    img_p = preprocess_rx(img, contrast, denoise)
    img_n = img_p.astype("float32") / 255.0
    t = torch.from_numpy(img_n).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        out = model(t)
        pred = torch.argmax(out, 1)[0].numpy()

    # DEBUG CLASSES
    st.write("🧪 Pixels par classe :", {
        "fond": int((pred==0).sum()),
        "soudure": int((pred==1).sum()),
        "defaut": int((pred==2).sum())
    })

    solder = (pred==1) & inspect_mask
    defect = (pred==2) & inspect_mask

    voids, lacks = classify_defects(defect, solder, inspect_mask, cfg)
    metrics = compute_metrics(voids, lacks, solder.sum())
    metrics["image"] = img_file.name

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # VISU MASQUE
    overlay_mask = overlay.copy()
    overlay_mask[inspect_mask] = [0,255,0]

    overlay[solder] = [120,0,0]
    overlay[defect] = [0,0,255]

    if voids:
        v = max(voids, key=lambda r:r.area)
        y,x = v.centroid
        cv2.circle(
            overlay,
            (int(x),int(y)),
            int(v.equivalent_diameter/2),
            (255,200,0),4
        )

    st.session_state.images.append(img)
    st.session_state.overlays.append(overlay)
    st.session_state.results.append(metrics)

    col1, col2, col3 = st.columns(3)
    col1.image(img, caption="Original", clamp=True)
    col2.image(overlay_mask, caption="Masque appliqué", clamp=True)
    col3.image(overlay, caption="Analyse", clamp=True)

# =========================
# RESULTS TABLE
# =========================
if st.session_state.results:
    st.subheader("📊 Résultats cumulés")
    df = pd.DataFrame(st.session_state.results)
    st.dataframe(df)


