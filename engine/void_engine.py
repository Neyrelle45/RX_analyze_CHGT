import cv2
import numpy as np


def analyze_voids(defect_mask, inspect_mask):
    """
    Returns:
        filtered_mask
        stats dict
    """

    # limiter à zone inspection
    defect_mask = defect_mask & inspect_mask

    defect_mask = defect_mask.astype(np.uint8) * 255

    # -------------------------------------------------
    # CONNECTED COMPONENTS
    # -------------------------------------------------

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        defect_mask,
        connectivity=8
    )

    filtered = np.zeros_like(defect_mask)

    void_areas = []

    H, W = defect_mask.shape
    image_area = H * W

    # seuil industriel typique
    min_void_pixels = max(12, image_area * 0.00005)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_void_pixels:
            continue

        component = (labels == i).astype(np.uint8)

        # -------------------------------------------------
        # CIRCULARITY
        # -------------------------------------------------

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        cnt = contours[0]

        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # -------------------------------------------------
        # REJET FORMES NON VOID
        # -------------------------------------------------

        if circularity < 0.25:
            continue

        # bounding box
        x, y, w, h = cv2.boundingRect(cnt)

        aspect_ratio = w / h if h != 0 else 0

        if aspect_ratio > 2.5 or aspect_ratio < 0.4:
            continue

        filtered[labels == i] = 255
        void_areas.append(area)

    filtered_mask = filtered.astype(bool)

    total_void = np.sum(filtered_mask)

    largest_void = max(void_areas) if void_areas else 0

    return filtered_mask, {
        "void_pixels": int(total_void),
        "largest_void_pixels": int(largest_void),
        "void_count": len(void_areas)
    }
