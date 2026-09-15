
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import geopandas as gpd
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import requests
from PIL import Image, ImageOps
from pyproj import Transformer
from shapely.geometry import Polygon, box


# ---------------------------------------------------------------------------
# Project paths and reproducible spatial settings
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "res"
TOPO_DIR = RESULT_DIR / "topology"
FIG_DIR = RESULT_DIR / "fig"
CACHE_DIR = PROJECT_ROOT / ".cache" / "topo"
TILE_CACHE_DIR = CACHE_DIR / "tiles"
SHP_DIR = RESULT_DIR / "shp"

WGS84 = "EPSG:4326"
WEB_MERCATOR = "EPSG:3857"
HK80 = "EPSG:2326"

# This extent was matched to homantin.png.  It contains the central Kowloon
# road system around Ho Man Tin, not Hong Kong's rail/pedestrian/cycle layers.
CONTEXT_WEST = 114.1580
CONTEXT_SOUTH = 22.3015
CONTEXT_EAST = 114.2055
CONTEXT_NORTH = 22.3420

# Town Planning Board: Ho Man Tin (KPA 6 & 7) OZP, current at 2026-09-05.
TARGET_PLAN_NO = "S/K7/26"
TPB_DATASET_ID = "f5334b37-85de-557c-bcb6-fe7061b332bb"
CSDI_REST_BASE = (
    "https://portal.csdi.gov.hk/server/rest/services/common/"
    f"{TPB_DATASET_ID}"
)

# Approximate fallback traced from the official textual limits: Boundary
# Street (north); East Rail Line and Princess Margaret Road (west); Chatham
# Road North (south); Lo Lung Road, Tin Kwong Road, the eastern slopes of Ho
# Man Tin Hill and Shun Yung Street (east).  Its area is about 2.05 km2, close
# to the statutory planning-scheme area reported by TPB documents.
FALLBACK_BOUNDARY_COORDS = [
    (114.1720, 22.3270),
    (114.1860, 22.3270),
    (114.1870, 22.3230),
    (114.1875, 22.3190),
    (114.1875, 22.3150),
    (114.1860, 22.3120),
    (114.1840, 22.3090),
    (114.1810, 22.3090),
    (114.1795, 22.3120),
    (114.1780, 22.3160),
    (114.1765, 22.3200),
    (114.1735, 22.3230),
]

BOUNDARY_PATH = TOPO_DIR / "homantin_boundary.geojson"
CONTEXT_GRAPH_PATH = TOPO_DIR / "kowloon_context_drive.graphml"
CONTEXT_GPKG_PATH = TOPO_DIR / "kowloon_context_drive.gpkg"
METADATA_PATH = TOPO_DIR / "topology_metadata.json"
NODE_CSV_PATH = TOPO_DIR / "node.csv"
LINK_CSV_PATH = TOPO_DIR / "link.csv"
NODE_SHP_PATH = SHP_DIR / "node.shp"
LINK_SHP_PATH = SHP_DIR / "link.shp"

FIG_CONTEXT_PATH = FIG_DIR / "01_context_network_with_homantin.jpg"

# Public OSM raster tiles are converted to a pale greyscale background before
# plotting, keeping labels/landmarks as context without competing with the
# analytical road-class colours.
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_PROVIDER_CACHE = "osm_standard_greyscale"

HTTP_HEADERS = {
    "User-Agent": "HoManTin-resilience-research/1.0 (academic network analysis)",
    "Accept": "application/json,application/geo+json,image/png,*/*",
    "Connection": "close",
}


ROAD_COLORS = {
    "motorway": "#8F1D20",
    "motorway_link": "#B53A37",
    "trunk": "#C04B21",
    "trunk_link": "#D96C3B",
    "primary": "#D9822B",
    "primary_link": "#E9A260",
    "secondary": "#2676A8",
    "secondary_link": "#559CC6",
    "tertiary": "#168A76",
    "tertiary_link": "#55AA96",
    "unclassified": "#65737E",
    "residential": "#78858D",
    "living_street": "#939DA3",
    "service": "#AAB2B6",
}

ROAD_WIDTHS = {
    "motorway": 2.0,
    "motorway_link": 1.6,
    "trunk": 1.8,
    "trunk_link": 1.45,
    "primary": 1.55,
    "primary_link": 1.25,
    "secondary": 1.15,
    "secondary_link": 0.95,
    "tertiary": 0.90,
    "tertiary_link": 0.75,
    "unclassified": 0.60,
    "residential": 0.50,
    "living_street": 0.45,
    "service": 0.34,
}

EXCLUDED_HIGHWAYS = {
    "bridleway",
    "corridor",
    "cycleway",
    "footway",
    "path",
    "pedestrian",
    "platform",
    "proposed",
    "raceway",
    "steps",
    "track",
}

# OSMnx 1.9's stock ``drive`` filter excludes every ``highway=service`` way.
# That is unnecessarily restrictive here: publicly motor-accessible estate and
# facility roads are part of the road topology.  This filter retains them while
# still rejecting parking aisles, driveways, private access and non-motor ways.
MOTOR_ROAD_FILTER = (
    '["highway"]'
    '["area"!~"yes"]'
    '["access"!~"private|no"]'
    '["highway"!~"abandoned|bridleway|bus_guideway|construction|corridor|'
    'cycleway|elevator|escalator|footway|no|path|pedestrian|planned|platform|'
    'proposed|raceway|razed|steps|track"]'
    '["motor_vehicle"!~"no"]'
    '["motorcar"!~"no"]'
    '["service"!~"alley|driveway|emergency_access|parking|parking_aisle|private"]'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="redownload the official boundary, OSM graph and map tiles",
    )
    parser.add_argument(
        "--no-basemap",
        action="store_true",
        help="make figures without web-map tiles",
    )
    return parser.parse_args()


def configure() -> None:
    for directory in (TOPO_DIR, FIG_DIR, CACHE_DIR, TILE_CACHE_DIR, SHP_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # The only text drawn on the maps is English, so this enforces Arial for
    # titles, annotations, scale bars and attribution alike.
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "axes.unicode_minus": False,
            "figure.facecolor": "#F7F5F0",
            "savefig.facecolor": "#F7F5F0",
        }
    )
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(CACHE_DIR / "osmnx")
    ox.settings.requests_timeout = 300
    ox.settings.log_console = True


def request_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                params=params,
                headers=HTTP_HEADERS,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(f"ArcGIS error: {result['error']}")
            return result
        except Exception as exc:  # network endpoints occasionally reset TLS
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Request failed for {url}: {last_error}")


def discover_csdi_service() -> Tuple[str, Dict[str, Any]]:
    errors = []
    for service_type in ("FeatureServer", "MapServer"):
        service_url = f"{CSDI_REST_BASE}/{service_type}"
        try:
            metadata = request_json(service_url, params={"f": "json"})
            if metadata.get("layers"):
                return service_url, metadata
            errors.append(f"{service_type}: no layers")
        except Exception as exc:
            errors.append(f"{service_type}: {exc}")
    raise RuntimeError("; ".join(errors))


def find_plan_field(fields: Iterable[Dict[str, Any]]) -> Optional[str]:
    names = [str(field.get("name", "")) for field in fields]
    normalized = {
        name.replace("_", "").replace(" ", "").upper(): name for name in names
    }
    for candidate in ("PLANNO", "PLANNUMBER", "PLANREF"):
        if candidate in normalized:
            return normalized[candidate]
    for name in names:
        normalized_name = name.replace("_", "").replace(" ", "").upper()
        if "PLAN" in normalized_name and (
            "NO" in normalized_name or "NUMBER" in normalized_name
        ):
            return name
    return None


def fetch_official_boundary() -> gpd.GeoDataFrame:
    """Query every CSDI polygon layer for the exact current OZP number."""
    service_url, service_meta = discover_csdi_service()
    diagnostics = []
    for layer in service_meta["layers"]:
        layer_url = f"{service_url}/{layer['id']}"
        try:
            layer_meta = request_json(layer_url, params={"f": "json"})
            if "polygon" not in str(layer_meta.get("geometryType", "")).lower():
                continue
            plan_field = find_plan_field(layer_meta.get("fields", []))
            if not plan_field:
                continue
            plan_value = TARGET_PLAN_NO.replace("'", "''")
            payload = request_json(
                f"{layer_url}/query",
                params={
                    "where": f"{plan_field} = '{plan_value}'",
                    "outFields": "*",
                    "returnGeometry": "true",
                    "outSR": "4326",
                    "f": "geojson",
                },
                timeout=90,
            )
            if payload.get("features"):
                boundary = gpd.GeoDataFrame.from_features(
                    payload["features"], crs=WGS84
                )
                boundary = boundary[boundary.geometry.notna()].copy()
                try:
                    boundary["geometry"] = boundary.geometry.make_valid()
                except Exception:
                    boundary["geometry"] = boundary.buffer(0)
                geometry = boundary.geometry.unary_union
                return gpd.GeoDataFrame(
                    {
                        "name": ["Ho Man Tin OZP"],
                        "plan_no": [TARGET_PLAN_NO],
                        "boundary_source": ["official_cSDI_TPB"],
                        "source_url": [service_url],
                        "layer_id": [int(layer["id"])],
                    },
                    geometry=[geometry],
                    crs=WGS84,
                )
            diagnostics.append(f"layer {layer['id']}: no exact match")
        except Exception as exc:
            diagnostics.append(f"layer {layer['id']}: {exc}")
    raise RuntimeError("No matching official polygon. " + "; ".join(diagnostics))


def fallback_boundary() -> gpd.GeoDataFrame:
    polygon = Polygon(FALLBACK_BOUNDARY_COORDS)
    return gpd.GeoDataFrame(
        {
            "name": ["Ho Man Tin OZP (approximate)"],
            "plan_no": [TARGET_PLAN_NO],
            "boundary_source": ["approximate_from_TPB_text_description"],
            "source_url": [
                "https://www.tpb.gov.hk/en/list_of_plans/plan_schd_ozp.html"
            ],
            "layer_id": [None],
        },
        geometry=[polygon],
        crs=WGS84,
    )


def load_or_fetch_boundary(refresh: bool) -> Tuple[gpd.GeoDataFrame, str]:
    cached: Optional[gpd.GeoDataFrame] = None
    if BOUNDARY_PATH.exists() and not refresh:
        cached = gpd.read_file(BOUNDARY_PATH).to_crs(WGS84)
        source = str(cached.iloc[0].get("boundary_source", ""))
        print(f"Using cached boundary ({source}): {BOUNDARY_PATH}")
        return cached, source

    try:
        boundary = fetch_official_boundary()
        source = str(boundary.iloc[0]["boundary_source"])
        print(f"Downloaded official TPB boundary for {TARGET_PLAN_NO}.")
    except Exception as exc:
        warnings.warn(
            "The official CSDI service is unavailable; using the documented "
            f"approximate boundary. Details: {exc}"
        )
        boundary = cached if cached is not None else fallback_boundary()
        source = str(boundary.iloc[0]["boundary_source"])

    boundary.to_file(BOUNDARY_PATH, driver="GeoJSON", encoding="utf-8")
    return boundary, source


def as_string_set(value: Any) -> set:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).lower() for item in value}
    return {str(value).lower()}


def clean_motor_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Apply an explicit final safeguard against non-motor transport ways."""
    cleaned = graph.copy()
    to_remove = []
    for u, v, key, data in cleaned.edges(keys=True, data=True):
        highways = as_string_set(data.get("highway"))
        access = str(data.get("access", "")).lower()
        motor_vehicle = str(data.get("motor_vehicle", "")).lower()
        railway = data.get("railway")
        if (
            highways & EXCLUDED_HIGHWAYS
            or access in {"no", "private"}
            or motor_vehicle == "no"
            or railway not in {None, "", "no"}
        ):
            to_remove.append((u, v, key))
    cleaned.remove_edges_from(to_remove)
    cleaned.remove_nodes_from(list(nx.isolates(cleaned)))
    return cleaned


def retain_largest_weak_component(
    graph: nx.MultiDiGraph,
) -> nx.MultiDiGraph:
    """Remove every detached road component after the spatial crop.

    Weak connectivity is the appropriate cropping check for a directed road
    graph: it tests whether all roads belong to one physical network while
    preserving genuine one-way restrictions. Requiring strong connectivity
    here would incorrectly delete valid boundary approaches and cul-de-sacs.
    """
    if graph.number_of_nodes() == 0:
        raise RuntimeError("The cropped motor-road graph is empty.")
    components = list(nx.weakly_connected_components(graph))
    largest = max(components, key=len)
    removed_nodes = graph.number_of_nodes() - len(largest)
    connected = graph.subgraph(largest).copy()
    connected.remove_nodes_from(list(nx.isolates(connected)))
    if not nx.is_weakly_connected(connected):
        raise RuntimeError("Failed to produce a weakly connected road graph.")
    connected.graph["connectivity_rule"] = "largest weakly connected component"
    connected.graph["detached_nodes_removed"] = int(removed_nodes)
    if removed_nodes:
        print(
            f"Removed {removed_nodes} nodes in {len(components) - 1} detached "
            "road component(s)."
        )
    return connected


def validate_topology(graph: nx.MultiDiGraph) -> None:
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise RuntimeError("The analysis graph is empty.")
    if nx.number_of_isolates(graph) != 0:
        raise RuntimeError("The analysis graph still contains isolated nodes.")
    if not nx.is_weakly_connected(graph):
        raise RuntimeError("The analysis graph is not weakly connected.")


def load_or_download_context_graph(refresh: bool) -> nx.MultiDiGraph:
    if CONTEXT_GRAPH_PATH.exists() and not refresh:
        cached = ox.load_graphml(CONTEXT_GRAPH_PATH)
        expected_bbox = (
            f"{CONTEXT_WEST},{CONTEXT_SOUTH},{CONTEXT_EAST},{CONTEXT_NORTH}"
        )
        if cached.graph.get("context_bbox_wgs84") == expected_bbox:
            cached = retain_largest_weak_component(clean_motor_graph(cached))
            validate_topology(cached)
            ox.save_graphml(cached, filepath=CONTEXT_GRAPH_PATH)
            print(f"Using cached OSM graph: {CONTEXT_GRAPH_PATH}")
            return cached
        print("Cached graph extent differs from the configured map; redownloading.")

    print("Downloading the wider central-Kowloon driving network from OSM...")
    graph = ox.graph_from_bbox(
        north=CONTEXT_NORTH,
        south=CONTEXT_SOUTH,
        east=CONTEXT_EAST,
        west=CONTEXT_WEST,
        network_type="drive",
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
        custom_filter=MOTOR_ROAD_FILTER,
    )
    graph = retain_largest_weak_component(clean_motor_graph(graph))
    validate_topology(graph)
    graph.graph["study_area"] = "homantin.png Kowloon context"
    graph.graph["transport_modes"] = "motor roads only"
    graph.graph["context_bbox_wgs84"] = (
        f"{CONTEXT_WEST},{CONTEXT_SOUTH},{CONTEXT_EAST},{CONTEXT_NORTH}"
    )
    ox.save_graphml(graph, filepath=CONTEXT_GRAPH_PATH)
    return graph


def first_road_class(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else "unclassified"
    value = str(value).lower()
    return value if value in ROAD_COLORS else "unclassified"


def _text_attr(value: Any, max_len: int = 250) -> str:
    """Serialize OSM attributes safely for CSV/Shapefile output."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        text = "|".join(str(item) for item in value)
    else:
        text = str(value)
    return text[:max_len]


def _bool_attr(value: Any) -> int:
    if isinstance(value, str):
        return int(value.strip().lower() in {"true", "1", "yes"})
    return int(bool(value))


def export_link_node_files(
    graph: nx.MultiDiGraph,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Export the final simplified directed graph to CSV and Shapefile.

    GraphML remains the authoritative computational topology.  The CSV files are
    convenient tabular exports, while the Shapefiles are the canonical plotting
    layers used by this script and later GIS figures.
    """
    nodes, edges = ox.graph_to_gdfs(
        graph,
        nodes=True,
        edges=True,
        fill_edge_geometry=True,
    )
    nodes = nodes.to_crs(WGS84).copy()
    edges = edges.to_crs(WGS84).copy()

    # ---- nodes -------------------------------------------------------------
    node_rows = []
    for node_id, row in nodes.iterrows():
        data = graph.nodes[node_id]
        node_rows.append(
            {
                "node_id": str(node_id),
                "lon": float(row.geometry.x),
                "lat": float(row.geometry.y),
                "in_deg": int(graph.in_degree(node_id)),
                "out_deg": int(graph.out_degree(node_id)),
                "degree": int(graph.degree(node_id)),
                "street_cnt": int(data.get("street_count", 0) or 0),
                "geometry": row.geometry,
            }
        )
    node_gdf = gpd.GeoDataFrame(node_rows, geometry="geometry", crs=WGS84)

    node_gdf.drop(columns="geometry").to_csv(
        NODE_CSV_PATH, index=False, encoding="utf-8-sig"
    )
    node_gdf.to_file(
        NODE_SHP_PATH,
        driver="ESRI Shapefile",
        encoding="utf-8",
        index=False,
    )

    # ---- directed links ----------------------------------------------------
    edge_rows = []
    for link_id, ((u, v, key), row) in enumerate(edges.iterrows(), start=1):
        data = graph.get_edge_data(u, v, key) or {}
        geometry = row.geometry
        edge_rows.append(
            {
                "link_id": int(link_id),
                "u": str(u),
                "v": str(v),
                "key": int(key),
                "osmid": _text_attr(data.get("osmid")),
                "name": _text_attr(data.get("name")),
                "highway": _text_attr(data.get("highway")),
                "length_m": float(data.get("length", 0.0) or 0.0),
                "oneway": _bool_attr(data.get("oneway", False)),
                "lanes": _text_attr(data.get("lanes")),
                "maxspeed": _text_attr(data.get("maxspeed")),
                "geometry": geometry,
            }
        )
    link_gdf = gpd.GeoDataFrame(edge_rows, geometry="geometry", crs=WGS84)

    # CSV keeps WKT so it is still spatially interpretable without the SHP.
    link_csv = pd.DataFrame(link_gdf.drop(columns="geometry"))
    link_csv["wkt"] = link_gdf.geometry.to_wkt()
    link_csv.to_csv(LINK_CSV_PATH, index=False, encoding="utf-8-sig")

    # Shapefile field names are deliberately <=10 chars where possible.
    link_shp = link_gdf.rename(columns={"length_m": "len_m"}).copy()
    link_shp.to_file(
        LINK_SHP_PATH,
        driver="ESRI Shapefile",
        encoding="utf-8",
        index=False,
    )

    print(f"Saved node CSV: {NODE_CSV_PATH}")
    print(f"Saved link CSV: {LINK_CSV_PATH}")
    print(f"Saved node SHP: {NODE_SHP_PATH}")
    print(f"Saved link SHP: {LINK_SHP_PATH}")
    return node_gdf, link_gdf


def load_plot_layers() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Read plotting layers from exported Shapefiles, never from the live graph."""
    if not NODE_SHP_PATH.exists() or not LINK_SHP_PATH.exists():
        raise FileNotFoundError(
            "node.shp/link.shp are missing. Run export_link_node_files() first."
        )
    nodes = gpd.read_file(NODE_SHP_PATH).to_crs(WEB_MERCATOR)
    links = gpd.read_file(LINK_SHP_PATH).to_crs(WEB_MERCATOR)
    return nodes, links


def graph_edges(graph: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """Backward-compatible helper; analytical plotting now reads link.shp."""
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True, fill_edge_geometry=True)
    edges = edges.to_crs(WEB_MERCATOR).copy()
    edges["road_class"] = edges["highway"].map(first_road_class)
    return edges

def save_network(context_graph: nx.MultiDiGraph) -> None:
    # GeoPackage is convenient for visual inspection in QGIS; GraphML remains
    # the authoritative topology because it preserves direction and parallel
    # edges directly.
    ox.save_graph_geopackage(
        context_graph, filepath=CONTEXT_GPKG_PATH, directed=True
    )


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    lat_rad = math.radians(lat_deg)
    scale = 2.0**zoom
    x_tile = int((lon_deg + 180.0) / 360.0 * scale)
    y_tile = int(
        (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * scale
    )
    return x_tile, y_tile


def num2deg(x_tile: int, y_tile: int, zoom: int) -> Tuple[float, float]:
    scale = 2.0**zoom
    lon_deg = x_tile / scale * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_tile / scale)))
    return math.degrees(lat_rad), lon_deg


def download_tile(x_tile: int, y_tile: int, zoom: int, refresh: bool) -> Image.Image:
    tile_path = (
        TILE_CACHE_DIR
        / TILE_PROVIDER_CACHE
        / str(zoom)
        / str(x_tile)
        / f"{y_tile}.png"
    )
    if tile_path.exists() and not refresh:
        return Image.open(tile_path).convert("RGB")
    tile_path.parent.mkdir(parents=True, exist_ok=True)
    url = TILE_URL.format(z=zoom, x=x_tile, y=y_tile)
    last_error: Optional[Exception] = None
    response = None
    for attempt in range(3):
        try:
            response = requests.get(url, headers=HTTP_HEADERS, timeout=45)
            response.raise_for_status()
            break
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
    if response is None or not response.ok:
        raise RuntimeError(f"Tile download failed for {url}: {last_error}")
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    image.save(tile_path)
    return image


def basemap_mosaic(
    bounds_wgs84: Sequence[float], zoom: int, refresh: bool
) -> Tuple[Image.Image, Sequence[float]]:
    min_lon, min_lat, max_lon, max_lat = bounds_wgs84
    x_min, y_min = deg2num(max_lat, min_lon, zoom)
    x_max, y_max = deg2num(min_lat, max_lon, zoom)
    mosaic = Image.new(
        "RGB", (256 * (x_max - x_min + 1), 256 * (y_max - y_min + 1))
    )
    for x_tile in range(x_min, x_max + 1):
        for y_tile in range(y_min, y_max + 1):
            tile = download_tile(x_tile, y_tile, zoom, refresh)
            mosaic.paste(tile, ((x_tile - x_min) * 256, (y_tile - y_min) * 256))

    lat_top, lon_left = num2deg(x_min, y_min, zoom)
    lat_bottom, lon_right = num2deg(x_max + 1, y_max + 1, zoom)
    transformer = Transformer.from_crs(WGS84, WEB_MERCATOR, always_xy=True)
    x_left, y_top = transformer.transform(lon_left, lat_top)
    x_right, y_bottom = transformer.transform(lon_right, lat_bottom)
    return mosaic, [x_left, x_right, y_bottom, y_top]


def padded_bounds(gdf: gpd.GeoDataFrame, ratio: float) -> Tuple[float, ...]:
    x_min, y_min, x_max, y_max = gdf.total_bounds
    dx, dy = x_max - x_min, y_max - y_min
    return (
        x_min - ratio * dx,
        y_min - ratio * dy,
        x_max + ratio * dx,
        y_max + ratio * dy,
    )


def mercator_bounds_to_wgs84(bounds: Sequence[float]) -> Tuple[float, ...]:
    inverse = Transformer.from_crs(WEB_MERCATOR, WGS84, always_xy=True)
    lon_min, lat_min = inverse.transform(bounds[0], bounds[1])
    lon_max, lat_max = inverse.transform(bounds[2], bounds[3])
    return lon_min, lat_min, lon_max, lat_max


def add_map_furniture(
    ax: plt.Axes,
    bounds: Sequence[float],
    scale_length_m: float,
) -> None:
    x_min, y_min, x_max, y_max = bounds
    width, height = x_max - x_min, y_max - y_min
    x0 = x_max - 0.055 * width - scale_length_m
    y0 = y_min + 0.060 * height
    ax.plot(
        [x0, x0 + scale_length_m],
        [y0, y0],
        color="#24323A",
        linewidth=2.6,
        solid_capstyle="butt",
        zorder=20,
    )
    ax.plot(
        [x0, x0, x0 + scale_length_m, x0 + scale_length_m],
        [y0 - 0.008 * height, y0 + 0.008 * height] * 2,
        color="#24323A",
        linewidth=1.2,
        zorder=20,
    )
    label = f"{int(scale_length_m / 1000)} km" if scale_length_m >= 1000 else f"{int(scale_length_m)} m"
    ax.text(
        x0 + scale_length_m / 2,
        y0 + 0.018 * height,
        label,
        ha="center",
        va="bottom",
        fontsize=11,
        color="#24323A",
        zorder=20,
    )
    ax.annotate(
        "N",
        xy=(x_max - 0.055 * width, y_max - 0.055 * height),
        xytext=(x_max - 0.055 * width, y_max - 0.145 * height),
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#24323A",
        arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": "#24323A"},
        zorder=20,
    )


def add_basemap(
    ax: plt.Axes,
    bounds: Sequence[float],
    zoom: int,
    refresh: bool,
    enabled: bool,
) -> bool:
    if not enabled:
        ax.set_facecolor("#F7F5F0")
        return False
    try:
        image, extent = basemap_mosaic(
            mercator_bounds_to_wgs84(bounds), zoom=zoom, refresh=refresh
        )
        pale = ImageOps.colorize(
            ImageOps.grayscale(image), black="#9AA5A8", white="#FAF9F5"
        )
        ax.imshow(pale, extent=extent, zorder=0, alpha=0.72)
        return True
    except Exception as exc:
        warnings.warn(f"Basemap download failed; continuing without it: {exc}")
        ax.set_facecolor("#F7F5F0")
        return False


def plot_edges(
    ax: plt.Axes,
    edges: gpd.GeoDataFrame,
    width_factor: float = 1.0,
    alpha: float = 0.86,
) -> None:
    # A single restrained line style deliberately mirrors homantin.png. Major
    # divided roads remain visually stronger because their carriageways are
    # represented by multiple parallel OSM edges, not because of class colour.
    edges.plot(
        ax=ax,
        color="#414648",
        linewidth=0.52 * width_factor,
        alpha=alpha,
        zorder=3,
    )


def figure_for_bounds(
    bounds: Sequence[float], width_inches: float = 9.0
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a full-bleed canvas whose aspect exactly matches map bounds."""
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    fig = plt.figure(figsize=(width_inches, width_inches * height / width))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    return fig, ax


def save_figure(fig: plt.Figure, path: Path) -> None:
    # Full-canvas axes and zero padding guarantee no surrounding white border.
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(
        path,
        dpi=300,
        format="jpg",
        bbox_inches=None,
        pad_inches=0,
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(fig)


def make_context_figure(
    boundary: gpd.GeoDataFrame,
    refresh: bool,
    basemap_enabled: bool,
) -> None:
    """Plot only from exported node/link Shapefiles."""
    nodes, links = load_plot_layers()
    boundary_web = boundary.to_crs(WEB_MERCATOR)
    context = gpd.GeoDataFrame(
        geometry=[
            box(
                CONTEXT_WEST,
                CONTEXT_SOUTH,
                CONTEXT_EAST,
                CONTEXT_NORTH,
            )
        ],
        crs=WGS84,
    ).to_crs(WEB_MERCATOR)
    bounds = padded_bounds(context, 0.0)

    fig, ax = figure_for_bounds(bounds, width_inches=8.0)
    add_basemap(ax, bounds, 13, refresh, basemap_enabled)

    # Links and nodes are explicitly read back from disk.  This guarantees that
    # the map visualizes exactly the same exported GIS layers delivered to the
    # user rather than a separate in-memory representation.
    links.plot(
        ax=ax,
        color="#414648",
        linewidth=0.52,
        alpha=0.88,
        zorder=3,
    )
    nodes.plot(
        ax=ax,
        color="#303638",
        markersize=1.4,
        alpha=0.38,
        zorder=4,
    )

    boundary_web.plot(
        ax=ax,
        facecolor="#D9A441",
        edgecolor="none",
        alpha=0.24,
        zorder=8,
    )
    boundary_web.boundary.plot(
        ax=ax, color="#9E2A2B", linewidth=1.8, zorder=9
    )
    centroid = boundary_web.geometry.iloc[0].centroid
    ax.annotate(
        "Ho Man Tin",
        xy=(centroid.x, centroid.y),
        xytext=(centroid.x + 1350, centroid.y + 1600),
        fontsize=14,
        fontweight="bold",
        color="#711C20",
        ha="left",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#9E2A2B",
            "linewidth": 0.8,
            "alpha": 0.92,
        },
        arrowprops={"arrowstyle": "-|>", "color": "#9E2A2B", "lw": 1.2},
        zorder=12,
    )
    add_map_furniture(ax, bounds, 1000)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_axis_off()
    save_figure(fig, FIG_CONTEXT_PATH)

def graph_summary(graph: nx.MultiDiGraph) -> Dict[str, Any]:
    lengths = [
        float(data.get("length", 0.0))
        for _, _, _, data in graph.edges(keys=True, data=True)
    ]
    strong_sizes = sorted(
        (len(component) for component in nx.strongly_connected_components(graph)),
        reverse=True,
    )
    return {
        "nodes": int(graph.number_of_nodes()),
        "directed_edges": int(graph.number_of_edges()),
        "directed_edge_length_km": round(sum(lengths) / 1000.0, 3),
        "is_directed": bool(graph.is_directed()),
        "is_multigraph": bool(graph.is_multigraph()),
        "isolated_nodes": int(nx.number_of_isolates(graph)),
        "weakly_connected": bool(nx.is_weakly_connected(graph)),
        "weak_component_count": int(nx.number_weakly_connected_components(graph)),
        "strong_component_count": int(len(strong_sizes)),
        "largest_strong_component_nodes": int(strong_sizes[0]),
        "largest_strong_component_share": round(
            strong_sizes[0] / graph.number_of_nodes(), 6
        ),
    }


def write_metadata(
    context_graph: nx.MultiDiGraph,
    boundary: gpd.GeoDataFrame,
    boundary_source: str,
) -> None:
    area_km2 = float(boundary.to_crs(HK80).geometry.area.sum() / 1e6)
    metadata = {
        "boundary": {
            "plan_no": TARGET_PLAN_NO,
            "source": boundary_source,
            "area_km2": round(area_km2, 4),
            "official_dataset_id": TPB_DATASET_ID,
            "official_dataset_url": (
                "https://portal.csdi.gov.hk/geoportal/?datasetId="
                f"{TPB_DATASET_ID}&lang=en"
            ),
            "fallback_basis": (
                "TPB OZP explanatory statement: Boundary Street; East Rail "
                "Line and Princess Margaret Road; Chatham Road North; Lo Lung "
                "Road, Tin Kwong Road, Ho Man Tin Hill eastern slopes and "
                "Shun Yung Street"
            ),
        },
        "context_bbox_wgs84": {
            "west": CONTEXT_WEST,
            "south": CONTEXT_SOUTH,
            "east": CONTEXT_EAST,
            "north": CONTEXT_NORTH,
        },
        "network": {
            "provider": "OpenStreetMap via OSMnx/Overpass",
            "network_type": "drive",
            "network_filter": MOTOR_ROAD_FILTER,
            "excluded_modes": [
                "rail",
                "metro",
                "bicycle-only",
                "pedestrian-only",
            ],
            "analysis_graph": graph_summary(context_graph),
        },
        "travel_time": {
            "included": False,
            "reason": (
                "No invented or generic speed was attached. For observed "
                "travel time, use TD processed road-segment speeds together "
                "with TD Road Network (2nd Generation) geometries; these "
                "official endpoints were not reachable during this run."
            ),
            "td_speed_url": (
                "https://resource.data.one.gov.hk/td/traffic-detectors/"
                "irnAvgSpeed-all.xml"
            ),
        },
        "outputs": {
            "analysis_graphml": str(CONTEXT_GRAPH_PATH),
            "analysis_gpkg": str(CONTEXT_GPKG_PATH),
            "figure": str(FIG_CONTEXT_PATH),
        },
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    configure()
    boundary, boundary_source = load_or_fetch_boundary(args.refresh)
    area_km2 = boundary.to_crs(HK80).geometry.area.sum() / 1e6
    print(f"Boundary source: {boundary_source}; area: {area_km2:.3f} km2")

    context_graph = load_or_download_context_graph(args.refresh)
    validate_topology(context_graph)
    save_network(context_graph)
    export_link_node_files(context_graph)
    make_context_figure(
        boundary,
        refresh=args.refresh,
        basemap_enabled=not args.no_basemap,
    )
    write_metadata(context_graph, boundary, boundary_source)

    print(json.dumps({
        "analysis_graph": graph_summary(context_graph),
    }, indent=2))
    print(f"Saved figure: {FIG_CONTEXT_PATH}")
    print(f"Saved topology: {TOPO_DIR}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
