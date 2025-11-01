import pandas as pd
import geopandas as gpd

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