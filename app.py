import streamlit as st
import cv2
import yaml
import numpy as np
import pandas as pd

from engine.preprocessing import preprocess_rx
from engine.inference import load_model, predict
from engine.postprocessing import classify_defects
from engine.metrics import compute_metrics

st.set_page_config(layout="wide")
st.title("RX Void Analyzer")

with st.sidebar:

    model_file=st.file_uploader("Model (.pth)")
    cfg_file=st.file_uploader("config.yaml")
    mask_file=st.file_uploader("Mask")

    heatmap_toggle=st.checkbox("Show defect heatmap")

    contrast=st.slider("Contrast",0.5,3.0,1.0)
    denoise=st.slider("Denoise",0,20,5)

if not(model_file and cfg_file and mask_file):
    st.stop()

cfg=yaml.safe_load(cfg_file)
model=load_model(model_file)

mask=cv2.imdecode(np.frombuffer(mask_file.read(),np.uint8),1)
inspect_mask=mask[:,:,1]>200

img_file=st.file_uploader("RX image")

if img_file:

    img=cv2.imdecode(np.frombuffer(img_file.read(),np.uint8),0)
    img_p=preprocess_rx(img,contrast,denoise)

    pred,heat=predict(model,img_p)

    solder=(pred==1)&inspect_mask
    defect=(pred==2)&inspect_mask

    voids,lacks=classify_defects(defect,solder,inspect_mask,cfg)
    metrics=compute_metrics(voids,lacks,solder.sum())

    overlay=cv2.cvtColor(img_p,cv2.COLOR_GRAY2BGR)

    blue=np.zeros_like(overlay)
    blue[solder]=[180,0,0]
    overlay=cv2.addWeighted(overlay,1,blue,0.35,0)

    red=np.zeros_like(overlay)
    red[defect]=[0,0,255]
    overlay=cv2.addWeighted(overlay,1,red,0.9,0)

    if heatmap_toggle:
        st.image(heat,cmap="hot")

    col1,col2=st.columns(2)
    col1.image(img_p,caption="Original")
    col2.image(overlay,caption="Analysis")

    st.dataframe(pd.DataFrame([metrics]))

