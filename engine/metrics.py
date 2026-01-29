def compute_metrics(voids, lacks, solder_area):

    defect_area=sum(r.area for r in voids+lacks)

    biggest=max(voids,key=lambda r:r.area) if voids else None

    return {
        "taux_defaut_%": round(defect_area/solder_area*100,2) if solder_area else 0,
        "plus_gros_void_%": round(biggest.area/solder_area*100,2) if biggest else 0,
        "nb_void":len(voids),
        "nb_manque":len(lacks)
    }
