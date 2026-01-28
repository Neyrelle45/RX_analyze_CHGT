import streamlit as st
import cv2, yaml, torch
import numpy as np
from engine.model import UNetLite
from engine.preprocessing import preprocess_rx
from engine.postprocessing import classify_defects
from engine.metrics import compute_metrics

DEVICE = "cpu"

st.set_page_config(layout="wide")
st.title("🔍 Analyse RX – Générique")

# === Uploads ===
model_file = st.file_uploader("Modèle IA (.pth)", type=["pth"])
config_file = st.file_uploader("Config modèle (.yaml)", type=["yaml","yml"])
img_file    = st.file_uploader("Image RX", type=["png","jpg"])
mask_file   = st.file_uploader("Masque inspection", type=["png"])

contrast = st.slider("Contraste", 0.5, 3.0, 1.0, 0.1)
denoise  = st.slider("Réduction bruit", 0, 20, 5)

@st.cache_resource
def load_model(weights):
    model = UNetLite()
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.eval()
    return model

if model_file and config_file and img_file and mask_file:
    cfg = yaml.safe_load(config_file)

    model = load_model(model_file)

    img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), 0)
    mask = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_COLOR)
    inspect = mask[:,:,1] > 200

    img_p = preprocess_rx(img, contrast, denoise)
    img_n = img_p / 255.0
    t = torch.tensor(img_n).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        pred = torch.argmax(model(t),1)[0].numpy()

    solder = (pred==1) & inspect
    defect = (pred==2) & inspect

    voids, lacks = classify_defects(defect, solder, inspect, cfg)
    metrics = compute_metrics(voids, lacks, np.sum(solder))

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay[solder] = [120,0,0]
    overlay[defect] = [0,0,255]

    if voids:
        v = max(voids, key=lambda r:r.area)
        y,x = v.centroid
        cv2.circle(overlay,(int(x),int(y)),int(v.equivalent_diameter/2),(255,200,0),4)

    c1,c2 = st.columns(2)
    c1.image(img, caption="Original", clamp=True)
    c2.image(overlay, caption="Analysé", clamp=True)

    st.subheader("📊 Résultats")
    st.table(metrics)

