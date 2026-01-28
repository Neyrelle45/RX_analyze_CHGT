import streamlit as st
import cv2, os
import numpy as np
import torch
import pandas as pd
from skimage.measure import label, regionprops

ROOT_PATH = "Analyze_RX"
IMG_DIR = f"{ROOT_PATH}/rx_images"
MASK_DIR = f"{ROOT_PATH}/masks"
OUT_DIR = f"{ROOT_PATH}/resultats/images"
MODEL_PATH = f"{ROOT_PATH}/models/BTC/model.pth"

os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SimpleUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = torch.nn.Sequential(torch.nn.Conv2d(1,16,3,padding=1), torch.nn.ReLU())
        self.enc2 = torch.nn.Sequential(torch.nn.Conv2d(16,32,3,padding=1), torch.nn.ReLU())
        self.pool = torch.nn.MaxPool2d(2)
        self.dec1 = torch.nn.Sequential(torch.nn.Conv2d(32,16,3,padding=1), torch.nn.ReLU())
        self.out  = torch.nn.Conv2d(16,3,1)
    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = torch.nn.functional.interpolate(e2, scale_factor=2)
        d1 = self.dec1(d1)
        return self.out(d1)

@st.cache_resource
def load_model():
    m = SimpleUNet().to(DEVICE)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.eval()
    return m

model = load_model()

def circularity(a,p): return 0 if p==0 else 4*np.pi*a/(p*p)

st.title("📁 Analyse RX BTC – Batch")

mask_name = st.selectbox("Masque fixe", os.listdir(MASK_DIR))
run = st.button("🚀 Lancer l’analyse batch")

if run:
    mask = cv2.imread(os.path.join(MASK_DIR, mask_name))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    inspect = mask[:,:,1] > 200

    results = []

    for name in os.listdir(IMG_DIR):
        img = cv2.imread(os.path.join(IMG_DIR,name),0)
        img_n = cv2.normalize(img,None,0,1,cv2.NORM_MINMAX)

        timg = torch.tensor(img_n,dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = torch.argmax(model(timg),1)[0].cpu().numpy()

        solder = (pred==1)&inspect
        defect = (pred==2)&inspect

        regions = regionprops(label(defect))
        surf = np.sum(solder)
        surf_def = sum([r.area for r in regions if r.area>20])

        results.append({
            "image":name,
            "taux_defaut_%":round(surf_def/surf*100,2) if surf else 0
        })

    df = pd.DataFrame(results)
    st.dataframe(df)
    df.to_csv(f"{ROOT_PATH}/resultats/BTC_batch.csv", index=False)
    st.success("✅ Analyse batch terminée")
