import re
from pathlib import Path
import networkx as nx


def extract_city_name_from_filename(path: Path) -> str:
    """
    Extrae el nombre de ciudad desde algo tipo:
    grafo_bogota_con_ae_embeddings.gexf -> bogota
    """
    m = re.match(r"grafo_(.+)_con_ae_embeddings\.gexf", path.name)
    if m:
        return m.group(1)
    return path.stem  # fallback raro, pero por si cambia el nombre


def graph_stats_from_file(path: Path) -> dict:
    """Carga un grafo GEXF y calcula estadísticas básicas."""
    city = extract_city_name_from_filename(path)

    G = nx.read_gexf(path)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    is_directed = G.is_directed()

    # Para conteo de componentes "débiles" (ignorando dirección)
    if is_directed:
        G_und = G.to_undirected()
    else:
        G_und = G

    # Componentes conexas (weak)
    components = list(nx.connected_components(G_und))
    n_weak_components = len(components)

    # Componente gigante
    if components:
        giant = max(components, key=len)
        giant_n_nodes = len(giant)
        giant_frac_nodes = giant_n_nodes / n_nodes if n_nodes > 0 else 0.0
    else:
        giant_n_nodes = 0
        giant_frac_nodes = 0.0

    # Componentes fuertemente conexas (si es dirigido)
    if is_directed:
        sccs = list(nx.strongly_connected_components(G))
        n_strong_components = len(sccs)
    else:
        n_strong_components = None

    # Densidad
    density = nx.density(G)

    # Grado medio (en grafo no dirigido; si es dirigido usamos grado total)
    if is_directed:
        degrees = dict(G.degree())
    else:
        degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / n_nodes if n_nodes > 0 else 0.0

    # % de nodos con embeddings (asumimos columnas A00..A63, o al menos una Axx)
    # Tomamos atributos del primer nodo como referencia de nombres
    embedding_pattern = re.compile(r"^A\d+$")
    embedding_cols = set()
    for _, data in G.nodes(data=True):
        for k in data.keys():
            if embedding_pattern.fullmatch(k):
                embedding_cols.add(k)
        if embedding_cols:
            break  # ya encontramos al menos una, basta

    n_with_embeddings = 0
    if embedding_cols:
        for _, data in G.nodes(data=True):
            # consideramos que tiene embeddings si al menos una columna Axx NO es None
            if any(data.get(col) is not None for col in embedding_cols):
                n_with_embeddings += 1

    frac_with_embeddings = (
        n_with_embeddings / n_nodes if n_nodes > 0 else 0.0
    )

    return {
        "city": city,
        "file": path.name,
        "is_directed": is_directed,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": density,
        "avg_degree": avg_degree,
        "n_weak_components": n_weak_components,
        "n_strong_components": n_strong_components,
        "giant_n_nodes": giant_n_nodes,
        "giant_frac_nodes": giant_frac_nodes,
        "n_nodes_with_embeddings": n_with_embeddings,
        "frac_nodes_with_embeddings": frac_with_embeddings,
    }
