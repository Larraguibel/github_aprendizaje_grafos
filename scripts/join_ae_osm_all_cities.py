import re
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.errors import ShapelyDeprecationWarning
from tqdm.auto import tqdm
import warnings

from utils import networkx_graph_to_gdf, try_int


MAX_DISTANCE_METERS = 200.0
METRIC_CRS = "EPSG:3857"  # CRS métrico global (Web Mercator)


def discover_cities(ae_dir: Path, graphs_dir: Path) -> list[str]:
    """Encuentra las ciudades que tienen embeddings y grafo OSM."""
    ae_cities = set()
    for gpkg in ae_dir.glob("embeddings_*.gpkg"):
        m = re.match(r"embeddings_(.+)\.gpkg", gpkg.name)
        if m:
            ae_cities.add(m.group(1))

    graph_cities = set()
    for gexf in graphs_dir.glob("grafo_*.gexf"):
        m = re.match(r"grafo_(.+)\.gexf", gexf.name)
        if m:
            graph_cities.add(m.group(1))

    common = sorted(ae_cities & graph_cities)
    return common


def ensure_lat_lon(poi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Asegura que poi_gdf tenga columnas 'lat' y 'lon' en EPSG:4326."""
    if "lat" in poi_gdf.columns and "lon" in poi_gdf.columns:
        return poi_gdf

    poi_ll = poi_gdf.to_crs(epsg=4326)
    if "lat" not in poi_ll.columns:
        poi_ll["lat"] = poi_ll.geometry.y
    if "lon" not in poi_ll.columns:
        poi_ll["lon"] = poi_ll.geometry.x
    return poi_ll


def build_enriched_graph_for_city(
    city: str,
    ae_path: Path,
    graph_path: Path,
    out_path: Path,
) -> None:
    """Crea un grafo OSM enriquecido con embeddings de AlphaEarth para una ciudad."""
    # 1) Cargar embeddings
    ae_emb = gpd.read_file(ae_path)
    if ae_emb.geometry is None or ae_emb.geometry.is_empty.all():
        raise ValueError(f"Embeddings file for {city} has no geometry: {ae_path}")
    if ae_emb.crs is None:
        # Fall back a WGS84 si falta CRS
        ae_emb = ae_emb.set_crs("EPSG:4326")

    # 2) Cargar grafo y convertir a GeoDataFrame de POIs
    poi_graph = nx.read_gexf(graph_path)
    poi = networkx_graph_to_gdf(poi_graph)
    if poi.crs is None:
        poi = poi.set_crs("EPSG:4326")

    # 3) Same CRS antes de proyectar
    if ae_emb.crs != poi.crs:
        poi = poi.to_crs(ae_emb.crs)

    # 4) Proyectar ambos a CRS métrico
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
        poi_m = poi.to_crs(METRIC_CRS)
        ae_m = (
            ae_emb.to_crs(METRIC_CRS)
            .reset_index()
            .rename(columns={"index": "ae_index"})
        )

    # 5) Buffer de 200 m alrededor de cada nodo
    poi_buffer = poi_m.copy()
    poi_buffer["geometry"] = poi_buffer.geometry.buffer(MAX_DISTANCE_METERS)

    # 6) Spatial join: embeddings dentro del buffer de cada POI
    matches = gpd.sjoin(
        ae_m,
        poi_buffer[["node_id", "geometry"]],
        how="inner",
        predicate="within",
    )

    if matches.empty:
        print(f"[warn] {city}: no matches between embeddings and nodes. Skipping.")
        return

    # 7) Lista de índices de embeddings por node_id
    if "ae_index" not in matches.columns:
        raise KeyError(f"'ae_index' column missing in matches for {city}")
    
    # 8) Promedio de embeddings por nodo
    embedding_cols = [c for c in matches.columns if re.fullmatch(r"A\d+", c)]
    if not embedding_cols:
        raise ValueError(f"No embedding columns 'Axx' found in embeddings for {city}")

    poi_emb_mean = (
        matches.groupby("node_id")[embedding_cols]
        .mean()
        .reset_index()
    )

    # 9) Metadatos del POI (asegurando lat/lon)
    poi_ll = ensure_lat_lon(poi)
    meta_cols = [
        c for c in ["node_id", "nombre", "tipo", "lat", "lon", "geometry"]
        if c in poi_ll.columns
    ]
    poi_meta = poi_ll[meta_cols].drop_duplicates(subset="node_id")

    poi_emb_enriched = poi_emb_mean.merge(
        poi_meta,
        on="node_id",
        how="inner",
    )

    poi_emb_enriched_gdf = gpd.GeoDataFrame(
        poi_emb_enriched,
        geometry="geometry",
        crs=poi_ll.crs,
    )

    # 10) Normalizar IDs de nodos y preparar atributos
    mapping = {n: try_int(n) for n in list(poi_graph.nodes)}
    if any(k != v for k, v in mapping.items()):
        poi_graph = nx.relabel_nodes(poi_graph, mapping, copy=True)

    poi_nodes = poi_emb_enriched_gdf.copy()
    poi_nodes["node_id_norm"] = poi_nodes["node_id"].apply(try_int)

    embedding_cols = [c for c in poi_nodes.columns if re.fullmatch(r"A\d+", c)]

    def row_to_attrs(row):
        attrs = {}
        if "nombre" in row.index:
            attrs["nombre"] = row.get("nombre")
        if "tipo" in row.index:
            attrs["tipo"] = row.get("tipo")
        if "lat" in row.index:
            attrs["lat"] = float(row["lat"]) if pd.notnull(row.get("lat")) else None
        if "lon" in row.index:
            attrs["lon"] = float(row["lon"]) if pd.notnull(row.get("lon")) else None
        for c in embedding_cols:
            val = row[c]
            if pd.notnull(val):
                attrs[c] = float(val)
        return attrs

    attrs_dict = {
        row["node_id_norm"]: row_to_attrs(row)
        for _, row in poi_nodes.iterrows()
    }

    missing_in_graph = []
    for nid, attrs in attrs_dict.items():
        if nid in poi_graph:
            poi_graph.nodes[nid].update(
                {k: v for k, v in attrs.items() if v is not None}
            )
        else:
            missing_in_graph.append(nid)

    print(
        f"[info] {city}: {len(poi_nodes)} nodes with embeddings. "
        f"{len(missing_in_graph)} IDs missing in graph."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(poi_graph, out_path)
    print(f"[ok] {city}: enriched graph written to {out_path}")


def main():
    # Root del repo: notebooks/utils/data_pipeline -> subir 3 niveles
    root = Path(__file__).resolve().parents[3]
    data_dir = root / "data"
    ae_dir = data_dir / "ae_embeddings"
    graphs_dir = data_dir / "city_graphs"
    out_dir = data_dir / "city_graphs_enriched"

    cities = discover_cities(ae_dir, graphs_dir)
    if not cities:
        print("[error] No common cities found between ae_embeddings and city_graphs.")
        return

    print(f"Found {len(cities)} cities: {', '.join(cities)}")
    print(f"Output directory: {out_dir}")

    for city in tqdm(cities, desc="Cities"):
        ae_path = ae_dir / f"embeddings_{city}.gpkg"
        graph_path = graphs_dir / f"grafo_{city}.gexf"
        out_path = out_dir / f"grafo_{city}_con_ae_embeddings.gexf"

        if not ae_path.exists():
            print(f"[skip] {city}: embeddings file not found: {ae_path}")
            continue
        if not graph_path.exists():
            print(f"[skip] {city}: graph file not found: {graph_path}")
            continue

        try:
            build_enriched_graph_for_city(city, ae_path, graph_path, out_path)
        except Exception as e:
            print(f"[error] {city}: {e}")


if __name__ == "__main__":
    main()
