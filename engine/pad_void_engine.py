import cv2
import numpy as np


def detect_pads(inspect_mask):
    """
    Détecte automatiquement les pads depuis le mask.

    Retourne une liste de masks binaires.
    """

    mask = inspect_mask.astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    pads = []

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        # filtre bruit
        if area < 200:
            continue

        pad = labels == i

        pads.append(pad)

    return pads


# ---------------------------------------------------


def analyze_pad_voids(void_mask, inspect_mask, ipc_limit=25):
    """
    Analyse void par pad.

    ipc_limit = seuil FAIL (%)
    """

    pads = detect_pads(inspect_mask)

    pad_results = []

    failing_pads = 0
    total_void = 0
    total_solder = 0

    for i, pad in enumerate(pads):

        pad_area = np.sum(pad)

        void_in_pad = np.sum(void_mask & pad)

        solder_in_pad = pad_area - void_in_pad

        ratio = (void_in_pad / pad_area * 100) if pad_area > 0 else 0

        fail = ratio > ipc_limit

        if fail:
            failing_pads += 1

        total_void += void_in_pad
        total_solder += solder_in_pad

        pad_results.append({
            "pad_id": i,
            "void_%": round(ratio, 2),
            "void_pixels": int(void_in_pad),
            "pad_pixels": int(pad_area),
            "FAIL": fail
        })

    global_ratio = (
        total_void / (total_void + total_solder) * 100
        if (total_void + total_solder) > 0
        else 0
    )

    summary = {
        "global_void_%": round(global_ratio, 2),
        "pads_total": len(pads),
        "pads_fail": failing_pads
    }

    return pad_results, summary, pads
