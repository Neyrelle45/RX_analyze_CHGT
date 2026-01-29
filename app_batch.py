import streamlit as st
import cv2
import yaml
import numpy as np
import pandas as pd

from engine.preprocessing import preprocess_rx
from engine.inference import load_model, predict
from engine.postprocessing import classify_defects
from engine.metrics import compute_metrics

st.title("RX Batch Analyzer")

model_file=st.file_uploader("Model")
cfg_file=st.file_uploader("Config")
mask_file=st.file_uploader("Mask")

if not(model_file and cfg_file and mask_file):
    st.stop()

cfg=yaml.safe_load(cfg_file)
model=load_model(model_file)

mask=cv2.imdecode(np.frombuffer(mask_file.read(),np.uint8),1)
inspect_mask=mask[:,:,1]>200

files=st.file_uploader("RX images",accept_multiple_files=True)

results=[]

if files:

    for f in files:

        img=cv2.imdecode(np.frombuffer(f.read(),np.uint8),0)
        img=preprocess_rx(img)

        pred,_=predict(model,img)

        solder=(pred==1)&inspect_mask
        defect=(pred==2)&inspect_mask

        voids,lacks=classify_defects(defect,solder,inspect_mask,cfg)
        metrics=compute_metrics(voids,lacks,solder.sum())

        metrics["image"]=f.name
        results.append(metrics)

    df=pd.DataFrame(results)
    st.dataframe(df)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        "rx_batch_results.csv"
    )
