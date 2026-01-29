def compute_metrics(voids, lacks, inspect_area):

    defect_area = sum(r.area for r in voids + lacks)

    if inspect_area == 0:
        inspect_area = 1

    biggest = max(voids, key=lambda r: r.area) if voids else None

    return {
        "taux_defaut_%": round(min(100, defect_area / inspect_area * 100), 2),
        "plus_gros_void_%": round(biggest.area / inspect_area * 100, 2) if biggest else 0,
        "nb_void": len(voids),
        "nb_manque": len(lacks)
    }
