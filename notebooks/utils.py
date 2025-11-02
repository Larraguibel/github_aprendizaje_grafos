# utilidades para geometrías espaciales

import pandas as pd
import geopandas as gpd
import ee


def networkx_graph_to_gdf(G, crs='EPSG:4326'):
    nodes_data = []
    for node_id, attrs in G.nodes(data=True):
        lat = attrs.get("lat", None)
        lon = attrs.get("lon", None)
        if lat is None or lon is None:
            continue  # saltar nodos sin coords

        nodes_data.append({
            "node_id": node_id,
            "lat": float(lat),
            "lon": float(lon),
            "tipo": attrs.get("tipo", None),
            "nombre": attrs.get("nombre", None),
        })
    nodes_df = pd.DataFrame(nodes_data)
    gdf_nodes = gpd.GeoDataFrame(
        nodes_df,
        geometry=gpd.points_from_xy(nodes_df["lon"], nodes_df["lat"]),
        crs=crs  # lat/lon WGS84
    )
    return gdf_nodes


def subdivide_bbox(min_lon, min_lat, max_lon, max_lat, depth=1, eps=1e-6):
    """
    Divide recursivamente un bounding box en 4 sub-rectángulos (NW, NE, SW, SE),
    agregando un pequeño solapamiento ('eps') entre las cajas para evitar huecos
    por errores numéricos o límites exactos.

    Parámetros
    ----------
    min_lon, min_lat, max_lon, max_lat : float
        Bounding box original.
    depth : int
        Nivel de recursión.
        depth=1 -> 4 sub-bboxes
        depth=2 -> 16 sub-bboxes
        depth=3 -> 64 sub-bboxes
        etc.
    eps : float
        Pequeño solapamiento en grados. Evita dejar franjas vacías al muestrear.

    Retorna
    -------
    list[ee.Geometry.Rectangle]
        Lista con todas las subcajas como ee.Geometry.Rectangle.
    """

    mid_lon = (min_lon + max_lon) / 2.0
    mid_lat = (min_lat + max_lat) / 2.0

    # Sub-rectángulos con leve solapamiento en los bordes compartidos
    boxes_coords = [
        # NW
        (min_lon,
         mid_lat - eps,
         mid_lon + eps,
         max_lat),

        # NE
        (mid_lon - eps,
         mid_lat - eps,
         max_lon,
         max_lat),

        # SW
        (min_lon,
         min_lat,
         mid_lon + eps,
         mid_lat + eps),

        # SE
        (mid_lon - eps,
         min_lat,
         max_lon,
         mid_lat + eps),
    ]

    if depth == 1:
        # caso base: devolver estas 4 como Rectangles
        return [ee.Geometry.Rectangle(b) for b in boxes_coords]

    # caso recursivo: subdividir cada sub-box nuevamente
    out = []
    for (sub_min_lon,
         sub_min_lat,
         sub_max_lon,
         sub_max_lat) in boxes_coords:

        out.extend(
            subdivide_bbox(
                sub_min_lon,
                sub_min_lat,
                sub_max_lon,
                sub_max_lat,
                depth=depth - 1,
                eps=eps
            )
        )

    return out
