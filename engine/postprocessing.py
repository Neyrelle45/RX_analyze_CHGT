from skimage.measure import label, regionprops
import numpy as np
import math

def circularity(area, perimeter):
    if perimeter == 0:
        return 0
    return 4 * math.pi * area / (perimeter**2)

def classify_defects(defect_mask, solder_mask, inspect_mask, cfg):

    labeled = label(defect_mask)

    voids=[]
    lacks=[]

    for r in regionprops(labeled):

        if r.area < cfg["min_area_px"]:
            continue

        circ = circularity(r.area, r.perimeter)

        touches_border = np.any((labeled==r.label) & (~inspect_mask))

        if circ >= cfg["void"]["circularity_min"] and not touches_border:
            voids.append(r)
        else:
            lacks.append(r)

    return voids,lacks

