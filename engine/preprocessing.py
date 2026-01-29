def preprocess_rx(
    img,
    contrast=1.0,
    denoise=5,
    clahe_strength=0.0,
    tophat_strength=0
):

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------
    # CLAHE (LOCAL CONTRAST)
    # ---------------------------------------------------

    if clahe_strength > 0:

        clip = 2.0 + clahe_strength * 4

        clahe = cv2.createCLAHE(
            clipLimit=clip,
            tileGridSize=(8, 8)
        )

        img = clahe.apply(img)

    # ---------------------------------------------------
    # TOP HAT (VOID BOOSTER)
    # ---------------------------------------------------

    if tophat_strength > 0:

        kernel_size = int(3 + tophat_strength * 6)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )

        tophat = cv2.morphologyEx(
            img,
            cv2.MORPH_BLACKHAT,
            kernel
        )

        img = cv2.add(img, tophat)

    # ---------------------------------------------------

    img_net, valid_mask, scale, shape = letterbox(img)

    if contrast != 1.0:
        img_net = cv2.convertScaleAbs(img_net, alpha=contrast)

    if denoise > 0:
        img_net = cv2.fastNlMeansDenoising(img_net, None, denoise)

    return img_net, valid_mask, scale, shape

