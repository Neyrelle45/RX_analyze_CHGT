def preprocess_rx(
    img,
    contrast=1.1,
    denoise=3,
    clahe_strength=0,
    blackhat_strength=0
):

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- NORMALISATION DOUCE ---
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)

    img = ((img - p1) / (p99 - p1) * 255).astype(np.uint8)

    # --- CONTRASTE LEGER ---
    img = cv2.convertScaleAbs(img, alpha=contrast)

    # --- DENOISE DOUX ---
    if denoise > 0:
        img = cv2.GaussianBlur(img, (3,3), 0)

    img_net, valid_mask, scale, shape = letterbox(img)

    return img_net, valid_mask, scale, shape


