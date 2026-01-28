import streamlit as st
import cv2, yaml, torch
import numpy as np
import pandas as pd
from engine.model import UNetLite
from engine.preprocessing import preprocess_rx
from engine.postprocessing import classify_defects
from engine.metrics import compute_metrics
from zipfile import ZipFile
import tempfile, os

DEVICE = "cpu"
st.title("📁 Analyse RX – Batch")

model_file = st.file_uploader("Modèle IA (.pth)", type=["pth"])
config_file = st.file_uploader("Config modèle (.yaml)", type=["yaml","yml"])
zip_file   = st.file_uploader("ZIP images RX", type=["zip"])
mask_file  = st.file_uploader("Masque inspection (.png)", type=["png"])

@st.cache_resource
def load_model(weights):
    model = UNetLite()
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.eval()
    return model

if model_file and config_file and zip_file and mask_file:
    cfg = yaml.safe_load(config_file)
    model = load_model(model_file)

    with tempfile.TemporaryDirectory() as tmp:
        ZipFile(zip_file).extractall(tmp)
        imgs = [f for f in os.listdir(tmp) if f.lower().endswith(('.png','.jpg'))]

        mask = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_COLOR)
        inspect = mask[:,:,1] > 200

        results = []

        for name in imgs:
            img = cv2.imread(os.path.join(tmp,name),0)
            img_p = preprocess_rx(img)
            img_n = img_p / 255.0
            t = torch.tensor(img_n).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                pred = torch.argmax(model(t),1)[0].numpy()

            solder = (pred==1) & inspect
            defect = (pred==2) & inspect

            voids, lacks = classify_defects(defect, solder, inspect, cfg)
            metrics = compute_metrics(voids, lacks, np.sum(solder))
            metrics["image"] = name
            results.append(metrics)

        df = pd.DataFrame(results)
        st.dataframe(df)
