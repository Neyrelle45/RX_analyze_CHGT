import streamlit as st
import cv2
import numpy as np
import torch
import pandas as pd
from skimage.measure import label, regionprops

# =========================
# CONFIG
# =========================
ROOT_PATH = "Analyze_RX"
MODEL_PATH = f"{ROOT_PATH}/models/BTC/model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# MODELE (identique notebook)
# =========================
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
    model = SimpleUNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# =========================
# OUTILS
# =========================
def circularity(area, perimeter):
    return 0 if perimeter == 0 else 4*np.pi*area/(perimeter**2)

def is_natural_blob(r):
    if circularity(r.area, r.perimeter) > 0.95:
        return False
    if r.major_axis_length/(r.minor_axis_length+1e-5) > 4:
        return False
    if r.area < 20:
        return False
    return True

def classify_defects(defect, solder, inspect):
    labeled = label(defect)
    voids, lacks = [], []

    for r in regionprops(labeled):
        if not is_natural_blob(r):
            continue

        circ = circularity(r.area, r.perimeter)
        obj = labeled == r.label

        touches_border = np.any(obj & (~inspect))
        fully_in_solder = np.all(obj <= solder)

        if circ >= 0.75 and fully_in_solder and not touches_border:
            voids.append(r)
        else:
            lacks.append(r)

    return voids, lacks

# =========================
# UI
# =========================
st.set_page_config(layout="wide")
st.title("🔍 Analyse RX BTC – Image par image")

img_file = st.file_uploader("Image RX (.png / .jpg)", type=["png","jpg","jpeg"])
mask_file = st.file_uploader("Masque inspection (.png)", type=["png"])

contrast = st.slider("Contraste", 0.5, 3.0, 1.0, 0.1)
denoise = st.slider("Réduction bruit", 0, 20, 5, 1)

if img_file and mask_file:
    img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), 0)
    mask = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_COLOR)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    inspect = mask[:,:,1] > 200

    # prétraitement
    img = cv2.convertScaleAbs(img, alpha=contrast)
    img = cv2.fastNlMeansDenoising(img, None, denoise)
    img_n = cv2.normalize(img,None,0,1,cv2.NORM_MINMAX)

    timg = torch.tensor(img_n,dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = torch.argmax(model(timg),1)[0].cpu().numpy()

    solder = (pred==1) & inspect
    defect = (pred==2) & inspect

    voids, lacks = classify_defects(defect, solder, inspect)
    big_void = max(voids, key=lambda r:r.area) if voids else None

    surf = np.sum(solder)
    surf_def = sum([r.area for r in voids+lacks])

    # rendu
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay[solder] = [120,0,0]
    overlay[defect] = [0,0,255]

    if big_void:
        y,x = big_void.centroid
        cv2.circle(overlay,(int(x),int(y)),
                   int(big_void.equivalent_diameter/2),
                   (255,200,0),4)

    c1,c2 = st.columns(2)
    c1.image(img, caption="Image originale", clamp=True)
    c2.image(overlay, caption="Image analysée", clamp=True)

    st.subheader("📊 Résultats")
    st.table(pd.DataFrame([{
        "Taux défaut (%)": round(surf_def/surf*100,2) if surf else 0,
        "Plus gros void (%)": round(big_void.area/surf*100,2) if big_void else 0,
        "Nb voids": len(voids),
        "Nb manques": len(lacks)
    }]))
