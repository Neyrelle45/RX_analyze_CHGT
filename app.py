import streamlit as st
import numpy as np
import cv2
import pandas as pd
import zipfile
import io
from PIL import Image

from engine.preprocessing import preprocess_rx
from engine.inference import load_model, predict_mask, find_largest_void


st.set_page_config(layout="wide")
st.title("PAD VOID ENGINE — Industrial AI")


# ------------------------------------------------
# SESSION
# ------------------------------------------------

if "results" not in st.session_state:
    st.session_state.results = []

if "saved_images" not in st.session_state:
    st.session_state.saved_images = []


# ------------------------------------------------
# CACHE MODEL
# ------------------------------------------------

@st.cache_resource
def get_model(file):
    return load_model(file)


@st.cache_data
def cached_preprocess(img, contrast, clahe, gamma):
    return preprocess_rx(img, contrast, clahe, gamma)


# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

model_file = st.sidebar.file_uploader("Model (.pth)", type="pth")
mask_file = st.sidebar.file_uploader("Inspection mask", type=["png","jpg"])

threshold = st.sidebar.slider("Void threshold",0.05,0.6,0.25,0.01)

tx = st.sidebar.slider("Translate X",-80,80,0)
ty = st.sidebar.slider("Translate Y",-80,80,0)
scale = st.sidebar.slider("Scale",0.95,1.05,1.0,0.001)
angle = st.sidebar.slider("Rotation",-3.0,3.0,0.0,0.1)

contrast = st.sidebar.slider("Global contrast",1.0,2.2,1.5,0.05)
clahe = st.sidebar.slider("Local contrast",1.0,4.0,2.0,0.1)
gamma = st.sidebar.slider("Gamma",0.8,1.6,1.1,0.05)

show_heatmap = st.sidebar.checkbox("Show heatmap", True)

if st.sidebar.button("RESET results"):
    st.session_state.results.clear()
    st.session_state.saved_images.clear()


model = get_model(model_file) if model_file else None


uploaded = st.file_uploader("RX image", type=["png","jpg","jpeg"])

if uploaded and model:

    original = np.array(Image.open(uploaded).convert("RGB"))

    tensor, processed = cached_preprocess(
        original,
        contrast,
        clahe,
        gamma
    )

    pred_mask, heatmap = predict_mask(
        model,
        tensor,
        threshold
    )

    h, w = processed.shape

    inspect_mask = np.ones((h,w),dtype=bool)

    if mask_file:

        mask = np.array(Image.open(mask_file).convert("L"))
        mask = cv2.resize(mask,(w,h))

        M = cv2.getRotationMatrix2D((w//2,h//2),angle,scale)
        M[:,2] += [tx,ty]

        aligned = cv2.warpAffine(mask,M,(w,h))
        inspect_mask = aligned > 127

        pred_mask &= inspect_mask


    largest, area, conf = find_largest_void(
        pred_mask,
        heatmap,
        inspect_mask
    )


    overlay = cv2.resize(original,(w,h))

    overlay[pred_mask] = [255,0,0]
    overlay[(~pred_mask)&inspect_mask] = [255,255,0]

    if largest is not None:

        coords = np.column_stack(np.where(largest))
        y,x = coords.mean(axis=0).astype(int)
        radius = int(np.sqrt(area/np.pi))

        cv2.circle(
            overlay,
            (x,y),
            radius,
            (235,206,135),
            4
        )


    void_px = int(pred_mask.sum())
    solder_px = int(((~pred_mask)&inspect_mask).sum())

    ratio = void_px / (void_px + solder_px + 1e-6) * 100


    col1,col2,col3 = st.columns(3)

    col1.image(original,use_container_width=True)

    mask_vis = original.copy()
    mask_vis = cv2.resize(mask_vis,(w,h))
    mask_vis[inspect_mask] = [0,255,0]

    col2.image(mask_vis,use_container_width=True)

    col3.image(
        overlay,
        caption="RED=void | YELLOW=solder",
        use_container_width=True
    )


    if show_heatmap:

        heat = (heatmap*255).astype(np.uint8)
        heat = cv2.applyColorMap(heat,cv2.COLORMAP_TURBO)

        heat = cv2.resize(
            heat,
            (w//2,h//2),
            interpolation=cv2.INTER_AREA
        )

        st.image(heat)


    df = pd.DataFrame([{
        "void_%":round(ratio,2),
        "largest_void_px":area,
        "AI_confidence_%":round(conf*100,1),
        "void_pixels":void_px,
        "solder_pixels":solder_px
    }])

    st.dataframe(df)


    if st.button("Save inspection"):

        st.session_state.results.append(df.iloc[0])

        _,buf = cv2.imencode(".png",overlay)
        st.session_state.saved_images.append(buf.tobytes())


if st.session_state.results:

    hist = pd.DataFrame(st.session_state.results)
    st.dataframe(hist)

    csv = hist.to_csv(index=False).encode()

    st.download_button(
        "Download CSV",
        csv,
        "void_results.csv"
    )

    zbuf = io.BytesIO()

    with zipfile.ZipFile(zbuf,"w") as z:
        for i,img in enumerate(st.session_state.saved_images):
            z.writestr(f"inspection_{i}.png",img)

    st.download_button(
        "Download images ZIP",
        zbuf.getvalue(),
        "void_images.zip"
    )

