import json
from pathlib import Path
import requests

from math import radians, cos, sin, asin, sqrt
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import networkx as nx

from shapely.geometry import shape, Point, LineString
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import contextily as cx
from tqdm import tqdm
import os



# Función para obtener datos de Overpass API
def get_overpass_data(query):
    """
    Consulta la API de Overpass y retorna los datos en formato JSON
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    response = requests.post(overpass_url, data={'data': query})
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error en la consulta: {response.status_code}")

# Función para clasificar el tipo de nodo
def classify_node(tags):
    """
    Clasifica un nodo según sus tags de OSM
    """
    if 'amenity' in tags:
        amenity = tags['amenity']
        if amenity in ['school', 'university', 'college', 'kindergarten']:
            return 'educacion'
        elif amenity in ['restaurant', 'cafe']:
            return 'gastronomia'
        elif amenity in ['hospital', 'clinic', 'doctors']:
            return 'salud'
        elif amenity in ['bank', 'pharmacy']:
            return 'servicios'
        elif amenity in ['police', 'fire_station']:
            return 'seguridad'
        elif amenity == 'library':
            return 'cultura'
    
    if 'shop' in tags:
        return 'comercio'
    
    if 'public_transport' in tags or 'highway' in tags:
        return 'transporte'
    
    if 'leisure' in tags:
        return 'recreacion'
    
    return 'otro'
    

def query_f(min_lon, min_lat, max_lon, max_lat):
    overpass_query = f"""
    [out:json][timeout:180][bbox:{min_lat},{min_lon},{max_lat},{max_lon}];
    (
    node["amenity"="school"];
    node["amenity"="university"];
    node["amenity"="college"];
    node["amenity"="kindergarten"];
    node["shop"];
    node["amenity"="restaurant"];
    node["amenity"="cafe"];
    node["amenity"="bank"];
    node["amenity"="pharmacy"];
    node["amenity"="hospital"];
    node["amenity"="clinic"];
    node["amenity"="doctors"];
    node["public_transport"="stop_position"];
    node["highway"="bus_stop"];
    node["leisure"="park"];
    node["leisure"="playground"];
    node["amenity"="police"];
    node["amenity"="fire_station"];
    node["amenity"="library"];
    );

    out body;
    """
    return overpass_query



def obtain_capital(min_lon, min_lat, max_lon, max_lat, nombre):
    overpass_query = query_f(min_lon, min_lat, max_lon, max_lat)
    data = get_overpass_data(overpass_query)

    # Procesar los nodos
    nodes_data = []
    for element in data['elements']:
        if element['type'] == 'node':
            node_info = {
                'id': element['id'],
                'lat': element['lat'],
                'lon': element['lon'],
                'tipo': classify_node(element.get('tags', {})),
                'nombre': element.get('tags', {}).get('name', 'Sin nombre'),
                'tags': element.get('tags', {})
            }
            nodes_data.append(node_info)

    print(f"Total de nodos obtenidos: {len(nodes_data)}")

    # Crear DataFrame
    df_nodes = pd.DataFrame(nodes_data)
    g_nodes = gpd.GeoDataFrame(df_nodes.copy(), geometry=gpd.points_from_xy(df_nodes["lon"], df_nodes["lat"]), crs="EPSG:4326")
    g_nodes_m = g_nodes.to_crs("EPSG:32719")
    coords = np.c_[g_nodes_m.geometry.x.values, g_nodes_m.geometry.y.values]
    tree = cKDTree(coords)


    MAX_DISTANCE = 200  # metros
    neighbors = tree.query_ball_tree(tree, r=MAX_DISTANCE)

    edges_list = []
    for i, neigh_list in enumerate(neighbors):
        for j in neigh_list:
            if i < j:
                id_i = g_nodes.iloc[i]["id"]
                id_j = g_nodes.iloc[j]["id"]
                edges_list.append((id_i, id_j))

    # Construir grafo
    G = nx.Graph()
    for _, node in df_nodes.iterrows():
        G.add_node(node['id'], 
                lat=node['lat'], 
                lon=node['lon'], 
                tipo=node['tipo'],
                nombre=node['nombre'])
    G.add_edges_from(edges_list)
    if not nx.is_connected(G):
        componentes = list(nx.connected_components(G))
        print(f"Número de componentes conexas para {nombre}: {len(componentes)}")
        print(f"Tamaño de la componente más grande para {nombre}: {len(max(componentes, key=len))}")
    
    nx.write_gexf(G, f'../data/grafo_OSM/grafo_{nombre}.gexf')

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

if __name__ == "__main__":
    for ciudad in tqdm(_CITY_BBOXES.keys()):
        obtain_capital(_CITY_BBOXES[ciudad]["min_lon"], _CITY_BBOXES[ciudad]["min_lat"],  _CITY_BBOXES[ciudad]["max_lon"], _CITY_BBOXES[ciudad]["max_lat"], ciudad)
        print(f"Ciudad {ciudad} procesada y guardada")
