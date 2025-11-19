import time
import sys
from pathlib import Path

import ee
import geopandas as gpd
from shapely.geometry import Point
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1] 
sys.path.append(str(ROOT))
from utils import subdivide_bbox


OUT_PATH_TEMPLATE = "data/ae_embeddings/embeddings_{}.gpkg"

_CITY_BBOXES = {
    # ==== SUDAMÉRICA (sin Santiago) ====
    "buenos_aires": {
        "min_lat": -34.7143027,
        "min_lon": -58.5539916,
        "max_lat": -34.5205861,
        "max_lon": -58.3227,
    },
    "lima": {
        "min_lat": -12.2796,
        "min_lon": -77.422,
        "max_lat": -11.7249,
        "max_lon": -76.619,
    },
    "bogota": {
        "min_lat": 3.722,
        "min_lon": -74.482,
        "max_lat": 4.851,
        "max_lon": -73.973,
    },
    "quito": {
        "min_lat": -0.23534,
        "min_lon": -78.52529,
        "max_lat": -0.2026,
        "max_lon": -78.49422,
    },

    # ==== EUROPA (sin Madrid) ====
    "paris": {
        "min_lat": 48.8,
        "min_lon": 2.2125,
        "max_lat": 48.9125,
        "max_lon": 2.475,
    },
    "greater_london": {
        "min_lat": 51.25,
        "min_lon": -0.57,
        "max_lat": 51.72,
        "max_lon": 0.37,
    },
    "rome": {
        "min_lat": 41.792,
        "min_lon": 12.308,
        "max_lat": 41.993,
        "max_lon": 12.685,
    },
    "berlin": {
        "min_lat": 52.327157,
        "min_lon": 13.066864,
        "max_lat": 52.684707,
        "max_lon": 13.781318,
    },
    "johannesburg": {
        "min_lat": -26.4,       # top = -25.9, bottom = -26.4
        "min_lon": 27.65,       # left = 27.65
        "max_lat": -25.9,
        "max_lon": 28.5,        # right = 28.5
    },

    # ==== USA ====
    "washington_dc": {
        "min_lat": 38.8591,
        "min_lon": -77.0886,
        "max_lat": 38.9375,
        "max_lon": -76.9902,
    },
}

def sample_box(geom, n_samples, embeddings):
    fc = embeddings.sample(
        region=geom,
        scale=10,
        numPixels=n_samples,
        geometries=True
    )
    fc_dict = fc.getInfo()
    rows = []
    for feat in fc_dict["features"]:
        props = dict(feat["properties"])
        lon, lat = feat["geometry"]["coordinates"]
        props["lon"] = lon
        props["lat"] = lat
        rows.append(props)
    return rows


def download_city_embeddings(
    city_name: str,
    bbox: dict,
    embeddings,
    depth: int = 3,
    n_samples_per_tile: int = 5000,
    overwrite: bool = False,
):
    """
    Descarga embeddings AlphaEarth para una ciudad, subdividiendo el bbox
    y guardando el resultado como GPKG en OUT_PATH_TEMPLATE.
    """
    out_path = Path(OUT_PATH_TEMPLATE.format(city_name))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        print(f"[skip] {city_name}: {out_path} ya existe, no se sobrescribe.")
        return gpd.read_file(out_path)

    boxes = subdivide_bbox(
        bbox["min_lon"],
        bbox["min_lat"],
        bbox["max_lon"],
        bbox["max_lat"],
        depth=depth,
    )

    all_rows = []
    start = time.time()

    # tqdm para ver progreso por tiles dentro de la ciudad
    for geom in tqdm(boxes, desc=f"{city_name}: tiles", leave=False):
        rows = sample_box(geom, n_samples=n_samples_per_tile, embeddings=embeddings)
        all_rows.extend(rows)

    # Pasamos a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        all_rows,
        geometry=[Point(r["lon"], r["lat"]) for r in all_rows],
        crs="EPSG:4326",
    )

    gdf_3857 = gdf.to_crs(epsg=3857)
    gdf_3857.to_file(out_path, driver="GPKG")

    elapsed = time.time() - start
    print(
        f"[done] {city_name}: {len(gdf_3857)} puntos -> {out_path} "
        f"({elapsed/60:.1f} min aprox.)"
    )
    return gdf

def save_all_cities_embeddings(
    depth: int = 3,
    n_samples_per_tile: int = 5000,
    overwrite: bool = False,
):
    """
    Descarga embeddings AlphaEarth para todas las ciudades en _CITY_BBOXES
    y guarda un GPKG por ciudad usando OUT_PATH_TEMPLATE.
    """
    # Colección de embeddings AlphaEarth (la creamos solo una vez)
    emb_col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    embeddings = emb_col.filterDate("2024-01-01", "2025-01-01").mosaic()

    resultados = {}

    # tqdm sobre las ciudades
    for city_name, bbox in tqdm(_CITY_BBOXES.items(), desc="Ciudades"):
        gdf_city = download_city_embeddings(
            city_name=city_name,
            bbox=bbox,
            embeddings=embeddings,
            depth=depth,
            n_samples_per_tile=n_samples_per_tile,
            overwrite=overwrite,
        )
        resultados[city_name] = gdf_city

    return resultados


if __name__ == "__main__":
    import ee

    print("Inicializando Google Earth Engine...")
    try:
        ee.Initialize(project="ee-dlarraguibel")
    except Exception as e:
        print("Fallo en ee.Initialize(). Intentando ee.Authenticate()...")
        ee.Authenticate()
        ee.Initialize(project="ee-dlarraguibel")

    print("\nDescargando embeddings AlphaEarth para todas las ciudades...\n")

    resultados = save_all_cities_embeddings(
        depth=3,
        n_samples_per_tile=4000,
        overwrite=False,   # Cambia a True si quieres regenerar archivos existentes
    )

    print("\nProceso completado.")
    for city, gdf in resultados.items():
        print(f" - {city}: {len(gdf)} puntos")
