import numpy as np
from skimage.measure import label, regionprops
import math

def circularity(area, perimeter):
    if perimeter == 0:
        return 0
    return 4 * math.pi * area / (perimeter**2)

def classify_defects(defect_mask, solder_mask, inspect_mask, cfg):
    labeled = label(defect_mask)
    voids, lacks = [], []

    for r in regionprops(labeled):
        if r.area < cfg["min_area_px"]:
            continue

        circ = circularity(r.area, r.perimeter)
        aspect = r.major_axis_length / (r.minor_axis_length + 1e-6)
        obj = labeled == r.label

        if circ > cfg["exclude"]["max_circularity"]:
            continue
        if aspect > cfg["exclude"]["max_aspect_ratio"]:
            continue

        touches_border = np.any(obj & (~inspect_mask))
        fully_in_solder = np.all(obj <= solder_mask)

        if circ >= cfg["void"]["circularity_min"] and fully_in_solder and not touches_border:
            voids.append(r)
        else:
            lacks.append(r)

    return voids, lacks
