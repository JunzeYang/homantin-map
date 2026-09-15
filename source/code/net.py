# -*- coding: utf-8 -*-
# @Time   : 2026/9/5
# @Author : Junze Yang
# @File   : net.py

"""Build the full directed motor-vehicle road network used by all analyses.

This revised copy intentionally keeps the original extraction/filtering logic
and link/node schema so that OD, criticality, robustness, and accessibility
scripts remain ID-compatible. The accidental duplicated second copy of the
script in the previous file has been removed. No Ho Man Tin analysis bbox is
introduced here: the complete retained largest weakly connected road network
defines the downstream computational domain.
"""

from pathlib import Path
import bz2
import math
import shutil

import networkx as nx
import pandas as pd
import geopandas as gpd
import osmnx as ox


# ============================================================
# 0. Paths and basic settings
# ============================================================

# Directory structure:
#
# resilience_homantin/
# ├── homantin.osm
# ├── node.csv
# ├── link.csv
# ├── shp/
# │   ├── node.shp
# │   └── link.shp
# └── code/
#     └── net.py
#
# net.py is in "code", so BASE_DIR is one level above it.
BASE_DIR = Path(__file__).resolve().parent.parent

OSM_FILE = BASE_DIR / "homantin.osm"
NODE_CSV = BASE_DIR / "node.csv"
LINK_CSV = BASE_DIR / "link.csv"

SHP_DIR = BASE_DIR / "shp"
NODE_SHP = SHP_DIR / "node.shp"
LINK_SHP = SHP_DIR / "link.shp"

SHP_DIR.mkdir(parents=True, exist_ok=True)

# Output encoding
OUTPUT_ENCODING = "gbk"

# Network options
INCLUDE_SERVICE_ROADS = False
KEEP_LARGEST_COMPONENT = True


# ============================================================
# 1. Road classes retained
# ============================================================

DRIVE_HIGHWAYS = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    # "unclassified",
    "residential",
    # "living_street",
}

if INCLUDE_SERVICE_ROADS:
    DRIVE_HIGHWAYS.add("service")


# ============================================================
# 2. Helper functions
# ============================================================

def as_list(value):
    """Convert an OSM attribute to a Python list."""
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return list(value)

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    return [value]


def value_to_string(value):
    """Convert an OSM attribute to a plain string."""
    if value is None:
        return None

    if isinstance(value, (list, tuple, set)):
        return ";".join(str(v) for v in value)

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def gbk_safe_text(value):
    """
    Convert text to a GBK-safe string.
    Characters unavailable in GBK are replaced instead of raising
    UnicodeEncodeError during CSV/SHP export.
    """
    text = value_to_string(value)
    if text is None:
        return None
    return text.encode(OUTPUT_ENCODING, errors="replace").decode(
        OUTPUT_ENCODING, errors="replace"
    )


def is_drive_edge(data):
    """Return True if an OSM edge should be retained as a driving road."""
    highways = {str(v) for v in as_list(data.get("highway"))}

    if not highways:
        return False

    if not highways.intersection(DRIVE_HIGHWAYS):
        return False

    access_values = {
        str(v).strip().lower()
        for v in as_list(data.get("access"))
    }

    # Explicitly inaccessible roads are removed.
    if access_values.intersection({"no", "private"}):
        return False

    return True


def ensure_useful_way_tags():
    """Ensure required OSM way tags survive graph construction."""
    required_tags = [
        "highway",
        "name",
        "ref",
        "oneway",
        "lanes",
        "maxspeed",
        "access",
        "bridge",
        "tunnel",
        "service",
        "junction",
    ]

    current = list(ox.settings.useful_tags_way)

    for tag in required_tags:
        if tag not in current:
            current.append(tag)

    ox.settings.useful_tags_way = current


def get_graph_from_xml_function():
    """
    Return graph_from_xml across OSMnx 1.x/2.x API layouts.
    """
    if hasattr(ox, "graph_from_xml"):
        return ox.graph_from_xml

    if hasattr(ox, "graph") and hasattr(ox.graph, "graph_from_xml"):
        return ox.graph.graph_from_xml

    raise AttributeError(
        "Cannot find OSMnx graph_from_xml(). "
        "Please check the installed OSMnx package."
    )


def load_osm_xml_compat(osm_path):
    """
    Load an OSM XML file in a way that is compatible with older OSMnx
    versions on Chinese Windows.

    Older OSMnx versions open a normal .osm file as text without specifying
    encoding. On Windows this can make Python use GBK even when the OSM XML
    itself is UTF-8, causing UnicodeDecodeError.

    Workaround:
    - read the original OSM as raw bytes;
    - compress the bytes to a temporary .osm.bz2 file;
    - let old OSMnx parse the bz2 stream in binary mode;
    - XML parser then respects the encoding declared by the XML itself.

    The original homantin.osm is never modified.
    """
    if not osm_path.exists():
        raise FileNotFoundError(f"Cannot find OSM file: {osm_path}")

    temp_bz2 = BASE_DIR / "_homantin_osmnx_temp.osm.bz2"

    # Remove any stale temporary file from a previous interrupted run.
    if temp_bz2.exists():
        temp_bz2.unlink()

    graph_from_xml = get_graph_from_xml_function()

    try:
        with osm_path.open("rb") as src, bz2.open(temp_bz2, "wb") as dst:
            shutil.copyfileobj(src, dst)

        # Do NOT pass encoding=... here:
        # the installed old OSMnx version does not support that argument.
        G = graph_from_xml(
            str(temp_bz2),
            bidirectional=False,
            simplify=False,
            retain_all=True,
        )

    finally:
        if temp_bz2.exists():
            try:
                temp_bz2.unlink()
            except PermissionError:
                # Non-fatal: the graph has already been loaded.
                pass

    return G


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in meters."""
    r = 6371009.0

    lon1 = math.radians(float(lon1))
    lat1 = math.radians(float(lat1))
    lon2 = math.radians(float(lon2))
    lat2 = math.radians(float(lat2))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2.0) ** 2
    )

    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def ensure_edge_lengths(G):
    """
    Ensure every pre-simplified edge has a numeric length attribute.
    Normally graph_from_xml already provides this, but this fallback
    avoids version-specific failures.
    """
    for u, v, key, data in G.edges(keys=True, data=True):
        length = data.get("length")

        valid = False
        try:
            if length is not None and not pd.isna(length):
                float(length)
                valid = True
        except (TypeError, ValueError):
            valid = False

        if not valid:
            x1 = G.nodes[u].get("x")
            y1 = G.nodes[u].get("y")
            x2 = G.nodes[v].get("x")
            y2 = G.nodes[v].get("y")

            if None in (x1, y1, x2, y2):
                raise ValueError(
                    f"Cannot calculate length for edge {(u, v, key)} "
                    "because node coordinates are missing."
                )

            data["length"] = haversine_m(x1, y1, x2, y2)
        else:
            data["length"] = float(length)

    return G


def simplify_graph_compat(G):
    """Simplify topology across different OSMnx API layouts."""
    if hasattr(ox, "simplification") and hasattr(
        ox.simplification, "simplify_graph"
    ):
        return ox.simplification.simplify_graph(G)

    if hasattr(ox, "simplify_graph"):
        return ox.simplify_graph(G)

    raise AttributeError(
        "Cannot find OSMnx simplify_graph(). "
        "Please check the installed OSMnx package."
    )


def graph_to_gdfs_compat(G):
    """Convert graph to node/edge GeoDataFrames across OSMnx versions."""
    kwargs = dict(
        nodes=True,
        edges=True,
        node_geometry=True,
        fill_edge_geometry=True,
    )

    if hasattr(ox, "graph_to_gdfs"):
        return ox.graph_to_gdfs(G, **kwargs)

    if hasattr(ox, "convert") and hasattr(ox.convert, "graph_to_gdfs"):
        return ox.convert.graph_to_gdfs(G, **kwargs)

    if hasattr(ox, "utils_graph") and hasattr(
        ox.utils_graph, "graph_to_gdfs"
    ):
        return ox.utils_graph.graph_to_gdfs(G, **kwargs)

    raise AttributeError(
        "Cannot find OSMnx graph_to_gdfs(). "
        "Please check the installed OSMnx package."
    )


def clean_dataframe_text_for_gbk(df):
    """Make all object/string columns safe for GBK output."""
    df = df.copy()

    for col in df.columns:
        if col == "geometry":
            continue

        if (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        ):
            df[col] = df[col].apply(gbk_safe_text)

    return df


def remove_existing_shapefile(path):
    """
    Remove an existing shapefile and its common sidecar files so that
    an old schema does not interfere with a new export.
    """
    extensions = [
        ".shp",
        ".shx",
        ".dbf",
        ".prj",
        ".cpg",
        ".qix",
        ".sbn",
        ".sbx",
    ]

    for ext in extensions:
        p = path.with_suffix(ext)
        if p.exists():
            p.unlink()


# ============================================================
# 3. Start
# ============================================================

print("=" * 72)
print("Ho Man Tin OSM road-network extraction")
print("=" * 72)
print(f"OSMnx version : {getattr(ox, '__version__', 'unknown')}")
print(f"OSM file      : {OSM_FILE}")
print(f"Base directory: {BASE_DIR}")
print()

ensure_useful_way_tags()


# ============================================================
# 4. Read original OSM network
# ============================================================

print("[1/7] Reading homantin.osm ...")

G = load_osm_xml_compat(OSM_FILE)

print(
    f"      Raw network: "
    f"{G.number_of_nodes():,} nodes, "
    f"{G.number_of_edges():,} directed edges"
)


# ============================================================
# 5. Filter to motor-vehicle roads
# ============================================================

print("[2/7] Filtering motor-vehicle roads ...")

edges_to_remove = [
    (u, v, key)
    for u, v, key, data in G.edges(keys=True, data=True)
    if not is_drive_edge(data)
]

G.remove_edges_from(edges_to_remove)

# Remove nodes with no remaining incident edge.
G.remove_nodes_from(list(nx.isolates(G)))

if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
    raise RuntimeError(
        "No driving road remains after filtering. "
        "Check the highway tags in homantin.osm or set "
        "INCLUDE_SERVICE_ROADS = True."
    )

print(
    f"      After filtering: "
    f"{G.number_of_nodes():,} nodes, "
    f"{G.number_of_edges():,} directed edges"
)


# ============================================================
# 6. Keep largest weakly connected component
# ============================================================

print("[3/7] Processing connected component ...")

if KEEP_LARGEST_COMPONENT:
    largest_nodes = max(
        nx.weakly_connected_components(G),
        key=len,
    )
    G = G.subgraph(largest_nodes).copy()

print(
    f"      Retained network: "
    f"{G.number_of_nodes():,} nodes, "
    f"{G.number_of_edges():,} directed edges"
)


# ============================================================
# 7. Ensure lengths and simplify topology
# ============================================================

print("[4/7] Simplifying topology ...")

G = ensure_edge_lengths(G)
G = simplify_graph_compat(G)

if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
    raise RuntimeError("The graph became empty after simplification.")

print(
    f"      Simplified network: "
    f"{G.number_of_nodes():,} nodes, "
    f"{G.number_of_edges():,} directed links"
)


# ============================================================
# 8. Convert graph to GeoDataFrames
# ============================================================

print("[5/7] Building node/link tables ...")

nodes_gdf, edges_gdf = graph_to_gdfs_compat(G)

# Preserve graph node identifiers directly from the GeoDataFrame index.
osm_node_ids = list(nodes_gdf.index)

# New compact IDs used by all subsequent analysis scripts.
osm_to_node_id = {
    osm_id: node_id
    for node_id, osm_id in enumerate(osm_node_ids)
}

# Directed degree statistics on the simplified graph.
in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())

# Unique-neighbour degree after ignoring direction and parallel links.
G_undirected_simple = nx.Graph(G)
undirected_degree = dict(G_undirected_simple.degree())


# ============================================================
# 9. Node output
# ============================================================

nodes_output = gpd.GeoDataFrame(
    {
        "node_id": [
            osm_to_node_id[osm_id]
            for osm_id in osm_node_ids
        ],
        "osm_id": [
            gbk_safe_text(osm_id)
            for osm_id in osm_node_ids
        ],
        "lon": pd.to_numeric(
            nodes_gdf["x"], errors="coerce"
        ).to_numpy(),
        "lat": pd.to_numeric(
            nodes_gdf["y"], errors="coerce"
        ).to_numpy(),
        "in_degree": [
            int(in_degree.get(osm_id, 0))
            for osm_id in osm_node_ids
        ],
        "out_degree": [
            int(out_degree.get(osm_id, 0))
            for osm_id in osm_node_ids
        ],
        "undir_deg": [
            int(undirected_degree.get(osm_id, 0))
            for osm_id in osm_node_ids
        ],
    },
    geometry=list(nodes_gdf.geometry),
    crs=nodes_gdf.crs,
)


# ============================================================
# 10. Link output
# ============================================================

# graph_to_gdfs returns a MultiIndex (u, v, key).
edge_indices = list(edges_gdf.index)

u_values = [idx[0] for idx in edge_indices]
v_values = [idx[1] for idx in edge_indices]
key_values = [idx[2] for idx in edge_indices]


def edge_attr(name):
    """Read an edge attribute, returning None when absent."""
    if name in edges_gdf.columns:
        return list(edges_gdf[name])
    return [None] * len(edges_gdf)


osm_way_values = edge_attr("osmid")
name_values = edge_attr("name")
highway_values = edge_attr("highway")
oneway_values = edge_attr("oneway")
length_values = edge_attr("length")
lanes_values = edge_attr("lanes")
maxspeed_values = edge_attr("maxspeed")
ref_values = edge_attr("ref")
access_values = edge_attr("access")
bridge_values = edge_attr("bridge")
tunnel_values = edge_attr("tunnel")


links_output = gpd.GeoDataFrame(
    {
        "link_id": range(len(edges_gdf)),
        "from_node": [
            osm_to_node_id[u]
            for u in u_values
        ],
        "to_node": [
            osm_to_node_id[v]
            for v in v_values
        ],
        "u_osm": [
            gbk_safe_text(v)
            for v in u_values
        ],
        "v_osm": [
            gbk_safe_text(v)
            for v in v_values
        ],
        "edge_key": [
            int(k) if isinstance(k, (int, float)) else gbk_safe_text(k)
            for k in key_values
        ],
        "osm_way": [
            gbk_safe_text(v)
            for v in osm_way_values
        ],
        "name": [
            gbk_safe_text(v)
            for v in name_values
        ],
        "highway": [
            gbk_safe_text(v)
            for v in highway_values
        ],
        "oneway": [
            gbk_safe_text(v)
            for v in oneway_values
        ],
        "length_m": pd.to_numeric(
            pd.Series(length_values),
            errors="coerce",
        ).to_numpy(),
        "lanes": [
            gbk_safe_text(v)
            for v in lanes_values
        ],
        "maxspeed": [
            gbk_safe_text(v)
            for v in maxspeed_values
        ],
        "ref": [
            gbk_safe_text(v)
            for v in ref_values
        ],
        "access": [
            gbk_safe_text(v)
            for v in access_values
        ],
        "bridge": [
            gbk_safe_text(v)
            for v in bridge_values
        ],
        "tunnel": [
            gbk_safe_text(v)
            for v in tunnel_values
        ],
    },
    geometry=list(edges_gdf.geometry),
    crs=edges_gdf.crs,
)


# ============================================================
# 11. Export CSV
# ============================================================

print("[6/7] Saving CSV files in GBK ...")

node_csv = pd.DataFrame(
    nodes_output.drop(columns="geometry")
)
node_csv["geometry"] = nodes_output.geometry.to_wkt()

link_csv = pd.DataFrame(
    links_output.drop(columns="geometry")
)
link_csv["geometry"] = links_output.geometry.to_wkt()

node_csv = clean_dataframe_text_for_gbk(node_csv)
link_csv = clean_dataframe_text_for_gbk(link_csv)

node_csv.to_csv(
    NODE_CSV,
    index=False,
    encoding=OUTPUT_ENCODING,
)

link_csv.to_csv(
    LINK_CSV,
    index=False,
    encoding=OUTPUT_ENCODING,
)


# ============================================================
# 12. Export Shapefiles
# ============================================================

print("[7/7] Saving Shapefiles in GBK ...")

# Shapefile field names should remain <= 10 characters.
node_shp = nodes_output.rename(
    columns={
        "in_degree": "in_deg",
        "out_degree": "out_deg",
    }
).copy()

link_shp = links_output.copy()

node_shp = clean_dataframe_text_for_gbk(node_shp)
link_shp = clean_dataframe_text_for_gbk(link_shp)

remove_existing_shapefile(NODE_SHP)
remove_existing_shapefile(LINK_SHP)

node_shp.to_file(
    NODE_SHP,
    driver="ESRI Shapefile",
    encoding="GBK",
)

link_shp.to_file(
    LINK_SHP,
    driver="ESRI Shapefile",
    encoding="GBK",
)


# ============================================================
# 13. Final checks and summary
# ============================================================

required_node_cols = {
    "node_id",
    "osm_id",
    "lon",
    "lat",
}

required_link_cols = {
    "link_id",
    "from_node",
    "to_node",
    "length_m",
}

if not required_node_cols.issubset(node_csv.columns):
    raise RuntimeError("node.csv is missing required columns.")

if not required_link_cols.issubset(link_csv.columns):
    raise RuntimeError("link.csv is missing required columns.")

if link_csv["from_node"].isna().any():
    raise RuntimeError("link.csv contains missing from_node values.")

if link_csv["to_node"].isna().any():
    raise RuntimeError("link.csv contains missing to_node values.")


print()
print("=" * 72)
print("Completed successfully")
print("=" * 72)
print(f"Nodes               : {len(nodes_output):,}")
print(f"Directed links      : {len(links_output):,}")
print(f"CRS                 : {nodes_output.crs}")
print()
print("Generated files:")
print(f"  {NODE_CSV}")
print(f"  {LINK_CSV}")
print(f"  {NODE_SHP}")
print(f"  {LINK_SHP}")
print()

if links_output["length_m"].notna().any():
    print("Link length statistics (m):")
    print(links_output["length_m"].describe())

print("=" * 72)
