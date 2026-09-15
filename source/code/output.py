# -*- coding: utf-8 -*-
# @Time   : 2026/9/6
# @File   : output.py
"""Generate concise, report-ready figures and tables for the Ho Man Tin / central
Kowloon OD-aware road-network resilience analysis.

The script is matched to the distance-based workflow:
    od_prepare.py -> graph_metric.py -> removal.py -> accessibility.py -> output.py

Display terminology
-------------------
BC   : OD-demand-weighted betweenness centrality.
GNE  : demand-weighted global network efficiency.
LSCC : largest strongly connected component.
TTD  : demand-weighted total travel distance.

Node and link criticality maps show normalized within-metric criticality on a
0-1 scale; raw values remain in the CSV tables. Hospital accessibility uses
directed shortest-path road distance.

Main figures
------------
Fig_00_Study_Area.jpg
Fig_01_Node_Criticality.jpg
Fig_02_Link_Criticality.jpg
Fig_03_Network_Robustness.jpg
Fig_04_Multi_Metric_Criticality.jpg
Fig_05_Network_Accessibility.jpg
Fig_06_Hospital_Accessibility.jpg
Fig_07_Hospital_Accessibility_Impact.jpg
Fig_08_Robustness_Summary.jpg
"""

from __future__ import annotations

OUTPUT_VERSION = "2026-09-06-od-distance-v5-contrast-cache-fix"

import argparse
import io
import json
import math
import re
import sys
import textwrap
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize, PowerNorm, TwoSlopeNorm, SymLogNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from PIL import Image, ImageEnhance
from pyproj import Transformer
import requests
from shapely import wkt
from shapely.geometry import Point, Polygon


# =============================================================================
# Paths and fixed project settings
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RES_DIR = PROJECT_ROOT / "res"
OUT_DIR = RES_DIR / "fig_tab"

OD_DIR = RES_DIR / "od"
GRAPH_DIR = RES_DIR / "graph_metric"
REMOVAL_DIR = RES_DIR / "removal"
ACCESS_DIR = RES_DIR / "accessibility"
SHP_ACCESS_DIR = RES_DIR / "shp" / "accessibility"

ROOT_NODE_CSV = PROJECT_ROOT / "node.csv"
ROOT_LINK_CSV = PROJECT_ROOT / "link.csv"

OD_META_PATH = OD_DIR / "od_meta.json"
ZONE_PATH = OD_DIR / "zone.csv"
ZONE_CONNECTOR_PATH = OD_DIR / "zone_connector.csv"
OD_ZONE_PATH = OD_DIR / "od_zone.csv"
OD_CONNECTOR_PATH = OD_DIR / "od_connector.csv"
STUDY_TPUSB_PATH = OD_DIR / "study_tpusb.geojson"

NODE_METRIC_PATH = GRAPH_DIR / "node_metric.csv"
LINK_METRIC_PATH = GRAPH_DIR / "link_metric.csv"
GRAPH_META_PATH = GRAPH_DIR / "metric_meta.json"

REMOVAL_CURVE_PATH = REMOVAL_DIR / "removal_curve.csv"
REMOVAL_SUMMARY_PATH = REMOVAL_DIR / "removal_summary.csv"
REMOVAL_META_PATH = REMOVAL_DIR / "removal_meta.json"

NODE_ACCESS_PATH = ACCESS_DIR / "node_access.csv"
HOSPITAL_PATH = ACCESS_DIR / "hospital.csv"
HOSPITAL_ACCESS_PATH = ACCESS_DIR / "hospital_access.csv"
HOSPITAL_SUMMARY_PATH = ACCESS_DIR / "hospital_summary.csv"
HOSPITAL_CATCHMENT_PATH = ACCESS_DIR / "hospital_catchment.csv"
ACCESS_IMPACT_PATH = ACCESS_DIR / "access_impact.csv"
ACCESS_REMOVAL_CURVE_PATH = ACCESS_DIR / "access_removal_curve.csv"
ACCESS_REMOVAL_SUMMARY_PATH = ACCESS_DIR / "access_removal_summary.csv"
ACCESS_META_PATH = ACCESS_DIR / "accessibility_meta.json"

GRID_SHP = SHP_ACCESS_DIR / "grid_100m.shp"
HOSPITAL_SHP = SHP_ACCESS_DIR / "hospital.shp"

OD_INPUT_DIR = PROJECT_ROOT / "HK&HMT OD flow from HKTCS022-myy-260905"
HMT_INTERNAL_OD_PATH = OD_INPUT_DIR / "ho_man_tin_internal_tpusb_od_flow.csv"

# Ho Man Tin boundary from the earlier topology workflow. If the cached
# GeoJSON is absent, use the same documented TPB-description fallback polygon
# as topo.py so Fig. 00 remains reproducible and does not depend on TPUSB
# endpoint coverage.
HMT_BOUNDARY_PATH = RES_DIR / "topology" / "homantin_boundary.geojson"
HMT_FALLBACK_BOUNDARY_COORDS = [
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

TILE_CACHE_DIR = PROJECT_ROOT / ".cache" / "output" / "osm_standard_tiles_v2"
OFFICIAL_ROAD_CACHE = PROJECT_ROOT / ".cache" / "output" / "landsd_road_centreline.geojson"

WGS84 = "EPSG:4326"
WEB_MERCATOR = "EPSG:3857"
HK80 = "EPSG:2326"

OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_HEADERS = {
    "User-Agent": "HoManTin-Network-Resilience/2.0 (academic report cartography)"
}

DEFAULT_DPI = 300
DEFAULT_TOP_N = 10
DEFAULT_LABEL_TOP = 0
DEFAULT_ZOOM = 15
DEFAULT_MAP_PAD = 0.025

# Mild nonlinear display normalization. These affect cartographic contrast only;
# all reported numerical values and rankings remain unchanged.
CRITICALITY_DISPLAY_GAMMA = 1.35
ACCESSIBILITY_DISPLAY_GAMMA = 1.25

# Colorbar ticks are intentionally denser at the low end and progressively
# wider at the high end. Combined with gamma > 1, this gives more visual
# separation among high-criticality/high-accessibility values without using a
# logarithmic scale or changing any reported metric.
NONLINEAR_TICK_FRACTIONS = np.asarray([0.0, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0])


def truncated_cmap(name: str, low: float = 0.22, high: float = 1.0, n: int = 256):
    """Return a backwards-compatible truncated Matplotlib colormap."""
    base = plt.get_cmap(name)
    colors = base(np.linspace(float(low), float(high), int(n)))
    return LinearSegmentedColormap.from_list(
        f"{name}_truncated_{low:.2f}_{high:.2f}", colors, N=int(n)
    )


def impact_diverging_cmap():
    """Diverging map with a narrow neutral centre so small nonzero changes remain visible."""
    return LinearSegmentedColormap.from_list(
        "hospital_p90_change",
        [
            (0.00, "#2166AC"),
            (0.34, "#67A9CF"),
            (0.485, "#D4E6F1"),
            (0.500, "#F2F2F2"),
            (0.515, "#F7D7CF"),
            (0.66, "#EF8A62"),
            (1.00, "#B2182B"),
        ],
        N=256,
    )


def set_fractional_colorbar_ticks(cbar, vmin: float, vmax: float, decimals: int = 2) -> None:
    """Use low-end-dense raw-value ticks on a nonlinear colorbar."""
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
        return
    ticks = vmin + NONLINEAR_TICK_FRACTIONS * (vmax - vmin)
    cbar.set_ticks(ticks)
    # Keep labels compact while preserving the actual metric scale.
    labels = []
    for value in ticks:
        if abs(value) >= 100:
            labels.append(f"{value:.0f}")
        elif abs(value) >= 10:
            labels.append(f"{value:.1f}")
        else:
            labels.append(f"{value:.{decimals}f}".rstrip("0").rstrip("."))
    cbar.set_ticklabels(labels)

# Fallback codes are derived from the supplied Ho Man Tin internal-OD file.
# When that file is present, these values are ignored and the codes are read
# directly, keeping the locator figure data-driven.
HMT_TPUSB_FALLBACK = {
    "231002", "232001", "233003", "234013", "234014", "234021", "234022",
    "234025", "234026", "235001", "235002", "235004", "235005", "235006",
    "235007", "235010", "236008", "236009", "236010", "236011", "236022",
    "236023", "236024", "237001", "237002", "237003", "237004",
}

# Common metrics for both nodes and directed links.
METRICS: Dict[str, Dict[str, Any]] = {
    "odbc": {
        "label": "Betweenness centrality",
        "short": "BC",
        "unit": "",
        "percent": False,
    },
    "od_eff_loss": {
        "label": "Global network efficiency",
        "short": "GNE",
        "unit": "%",
        "percent": True,
    },
    "lscc_loss": {
        "label": "Largest strongly connected component",
        "short": "LSCC",
        "unit": "%",
        "percent": True,
    },
    "td_loss": {
        "label": "Total travel distance",
        "short": "TTD",
        "unit": "%",
        "percent": True,
    },
}

STRATEGY_STYLE: Dict[str, Dict[str, Any]] = {
    "random": {"label": "Random", "color": "#4A4A4A", "linestyle": "-", "lw": 2.0},
    "odbc": {"label": "BC-targeted", "color": "#0072B2", "linestyle": "--", "lw": 1.9},
    "od_eff": {"label": "GNE-targeted", "color": "#009E73", "linestyle": "-.", "lw": 1.9},
    "lscc": {"label": "LSCC-targeted", "color": "#7A5195", "linestyle": ":", "lw": 2.1},
    "td": {"label": "TTD-targeted", "color": "#D55E00", "linestyle": (0, (5, 1.5)), "lw": 1.9},
}
STRATEGY_ORDER = ["random", "odbc", "od_eff", "lscc", "td"]

NODE_COLOR = "#2166AC"
LINK_COLOR = "#B2182B"
NODE_CMAP = "Blues"
LINK_CMAP = "Reds"
ACCESS_CMAP = "YlGnBu"
DISTANCE_CMAP = "viridis"
IMPACT_CMAP = "magma_r"


# =============================================================================
# Generic I/O and validation
# =============================================================================

def read_csv_auto(path: Path) -> pd.DataFrame:
    errors = []
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as exc:
            errors.append((enc, str(exc)))
    raise UnicodeError(f"Could not decode {path}: {errors}")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required upstream outputs are missing. Complete od_prepare.py, "
            "graph_metric.py, removal.py and accessibility.py first:\n  "
            + "\n  ".join(missing)
        )


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def clean_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    return "" if s.lower() in {"nan", "none"} else s


def english_road_name(value: object, fallback: str = "") -> str:
    """Extract the existing English/ASCII component from an OSM bilingual name."""
    s = clean_name(value)
    if not s:
        return fallback
    out = s.encode("ascii", errors="ignore").decode("ascii")
    out = re.sub(r"\s*;\s*", "; ", out)
    out = re.sub(r"\s+", " ", out).strip(" ;,-")
    return out if re.search(r"[A-Za-z]", out) else fallback


def choose_font() -> str:
    for candidate in ("Arial", "Liberation Sans", "DejaVu Sans"):
        try:
            path = font_manager.findfont(candidate, fallback_to_default=False)
            if path and Path(path).exists():
                return candidate
        except Exception:
            continue
    return "DejaVu Sans"


def configure_matplotlib() -> None:
    font = choose_font()
    mpl.rcParams.update(
        {
            "font.family": font,
            "font.size": 9.5,
            "axes.titlesize": 10.7,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.4,
            "figure.titlesize": 13.0,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#4A4A4A",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    return path


def format_metric_value(metric: str, value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    if METRICS[metric]["percent"]:
        return 100.0 * float(value)
    return float(value)


# =============================================================================
# Spatial input
# =============================================================================

def load_base_nodes() -> gpd.GeoDataFrame:
    df = read_csv_auto(ROOT_NODE_CSV)
    required = {"node_id", "lon", "lat"}
    if not required.issubset(df.columns):
        raise ValueError(f"node.csv missing {sorted(required - set(df.columns))}")
    df["node_id"] = pd.to_numeric(df["node_id"], errors="raise").astype(np.int64)
    geom = gpd.points_from_xy(numeric(df, "lon"), numeric(df, "lat"))
    return gpd.GeoDataFrame(df.drop(columns=["geometry"], errors="ignore"), geometry=geom, crs=WGS84)


def load_base_links() -> gpd.GeoDataFrame:
    df = read_csv_auto(ROOT_LINK_CSV)
    required = {"link_id", "from_node", "to_node", "length_m"}
    if not required.issubset(df.columns):
        raise ValueError(f"link.csv missing {sorted(required - set(df.columns))}")
    geom_col = "geometry" if "geometry" in df.columns else "wkt" if "wkt" in df.columns else None
    if geom_col is None:
        raise ValueError("link.csv needs a WKT geometry column named geometry or wkt.")
    geometry = df[geom_col].apply(lambda x: wkt.loads(x) if isinstance(x, str) and x.strip() else None)
    keep = df.drop(columns=[geom_col])
    for c in ("link_id", "from_node", "to_node"):
        keep[c] = pd.to_numeric(keep[c], errors="raise").astype(np.int64)
    return gpd.GeoDataFrame(keep, geometry=geometry, crs=WGS84)


def load_grid_geometry() -> gpd.GeoDataFrame:
    if not GRID_SHP.exists():
        raise FileNotFoundError(f"Missing {GRID_SHP}")
    g = gpd.read_file(GRID_SHP)
    if g.crs is None:
        g = g.set_crs(HK80)
    return g


def load_hospital_geometry(hospital: pd.DataFrame) -> gpd.GeoDataFrame:
    if HOSPITAL_SHP.exists():
        try:
            g = gpd.read_file(HOSPITAL_SHP)
            # Always attach authoritative full-name CSV fields.
            key = "hosp_id" if "hosp_id" in g.columns else "hospital_id"
            g[key] = pd.to_numeric(g[key], errors="raise").astype(np.int64)
            h = hospital.copy()
            h["hospital_id"] = pd.to_numeric(h["hospital_id"], errors="raise").astype(np.int64)
            g = g[[key, "geometry"]].rename(columns={key: "hospital_id"}).merge(h, on="hospital_id", how="right")
            return gpd.GeoDataFrame(g, geometry="geometry", crs=gpd.read_file(HOSPITAL_SHP).crs or WGS84)
        except Exception as exc:
            warnings.warn(f"Could not use hospital.shp ({exc}); constructing points from CSV.")
    return gpd.GeoDataFrame(
        hospital.copy(),
        geometry=gpd.points_from_xy(numeric(hospital, "longitude"), numeric(hospital, "latitude")),
        crs=WGS84,
    )


def load_study_tpusb() -> gpd.GeoDataFrame:
    g = gpd.read_file(STUDY_TPUSB_PATH)
    if "tpusb" not in g.columns:
        # Very defensive fallback for a future schema.
        candidates = [c for c in g.columns if "tp" in c.lower() and c != g.geometry.name]
        if not candidates:
            raise ValueError("study_tpusb.geojson contains no TPUSB code field.")
        g = g.rename(columns={candidates[0]: "tpusb"})
    g["tpusb"] = g["tpusb"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    if g.crs is None:
        g = g.set_crs(WGS84)
    else:
        g = g.to_crs(WGS84)
    return g


def load_hmt_boundary() -> gpd.GeoDataFrame:
    """Load the prior topology-workflow Ho Man Tin boundary, with offline fallback."""
    if HMT_BOUNDARY_PATH.exists():
        boundary = gpd.read_file(HMT_BOUNDARY_PATH)
        if boundary.empty:
            raise ValueError(f"Ho Man Tin boundary file is empty: {HMT_BOUNDARY_PATH}")
        if boundary.crs is None:
            boundary = boundary.set_crs(WGS84)
        else:
            boundary = boundary.to_crs(WGS84)
        geom = boundary.geometry.union_all() if hasattr(boundary.geometry, "union_all") else boundary.unary_union
        return gpd.GeoDataFrame({"name": ["Ho Man Tin area"]}, geometry=[geom], crs=WGS84)

    polygon = Polygon(HMT_FALLBACK_BOUNDARY_COORDS)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return gpd.GeoDataFrame({"name": ["Ho Man Tin area"]}, geometry=[polygon], crs=WGS84)


def infer_hmt_codes() -> set[str]:
    if HMT_INTERNAL_OD_PATH.exists():
        try:
            df = read_csv_auto(HMT_INTERNAL_OD_PATH)
            codes: set[str] = set()
            for c in ("origin_tpusb", "destination_tpusb"):
                if c in df.columns:
                    codes.update(
                        df[c].dropna().astype(str).str.replace(r"\.0$", "", regex=True).str.strip().tolist()
                    )
            if codes:
                return codes
        except Exception as exc:
            warnings.warn(f"Could not infer Ho Man Tin TPUSB codes from internal OD file: {exc}")
    return set(HMT_TPUSB_FALLBACK)


def merge_metric_geometry(
    geometry: gpd.GeoDataFrame,
    metric: pd.DataFrame,
    key: str,
) -> gpd.GeoDataFrame:
    left_cols = [key, "geometry"]
    for c in ("from_node", "to_node", "name", "highway", "length_m"):
        if c in geometry.columns and c not in left_cols:
            left_cols.append(c)
    left = geometry[left_cols].copy()
    left[key] = pd.to_numeric(left[key], errors="raise").astype(np.int64)
    right = metric.copy()
    right[key] = pd.to_numeric(right[key], errors="raise").astype(np.int64)
    # Numerical result CSVs may retain the original WKT text in a column named
    # geometry. Never merge that text column into a GeoDataFrame: otherwise
    # pandas creates geometry_x/geometry_y and GeoPandas loses the active
    # geometry column.
    right = right.drop(columns=["geometry", "wkt"], errors="ignore")
    # Avoid duplicate descriptive columns coming from both sides.
    duplicate = [c for c in left.columns if c in right.columns and c not in {key, "geometry"}]
    right = right.drop(columns=duplicate, errors="ignore")
    out = left.merge(right, on=key, how="left")
    return gpd.GeoDataFrame(out, geometry="geometry", crs=geometry.crs)


# =============================================================================
# Optional official road names
# =============================================================================

def detect_english_name_field(roads: gpd.GeoDataFrame) -> Optional[str]:
    fields = [c for c in roads.columns if c != roads.geometry.name]
    scored: List[Tuple[float, str]] = []
    for c in fields:
        k = re.sub(r"[^a-z0-9]", "", str(c).lower())
        score = 0.0
        if "name" in k or "ename" in k:
            score += 1.0
        if "street" in k or "road" in k:
            score += 1.0
        if "english" in k or "ename" in k or k.endswith("en"):
            score += 2.0
        if score > 0:
            scored.append((score, c))
    if not scored:
        return None
    return sorted(scored, reverse=True)[0][1]


def optionally_apply_official_road_names(
    links: gpd.GeoDataFrame,
    link_metric: pd.DataFrame,
    user_path: Optional[str],
) -> pd.DataFrame:
    """Attach official English road names when a local/cached file is available.

    This function never downloads automatically and never prevents plotting.
    The analysis is independent of the label source.
    """
    path: Optional[Path] = None
    if user_path:
        p = Path(user_path).expanduser().resolve()
        if p.exists():
            path = p
        else:
            warnings.warn(f"--official-road-file not found: {p}; using OSM English names.")
    elif OFFICIAL_ROAD_CACHE.exists():
        path = OFFICIAL_ROAD_CACHE

    result = link_metric.copy()
    result["display_name"] = result.get("name", pd.Series(index=result.index, dtype=object)).apply(
        lambda x: english_road_name(x, "")
    )
    result["name_source"] = np.where(result["display_name"].ne(""), "OSM English component", "link ID")

    if path is None:
        result.loc[result["display_name"].eq(""), "display_name"] = result.loc[
            result["display_name"].eq(""), "link_id"
        ].map(lambda x: f"Link {int(x)}")
        return result

    try:
        roads = gpd.read_file(path)
        if roads.crs is None:
            roads = roads.set_crs(WGS84)
        roads = roads.to_crs(HK80)
        field = detect_english_name_field(roads)
        if field is None:
            raise ValueError("No plausible English road-name field detected.")
        roads = roads[[field, "geometry"]].copy()
        roads[field] = roads[field].apply(clean_name)
        roads = roads[roads[field].ne("") & roads.geometry.notna() & (~roads.geometry.is_empty)].reset_index(drop=True)
        if roads.empty:
            raise ValueError("Official road file has no usable named features.")

        link_geo = links[["link_id", "geometry"]].to_crs(HK80).copy()
        # GeoPandas nearest join is concise and deterministic. A generous 60 m
        # tolerance is used only for cartographic labels, not analysis.
        near = gpd.sjoin_nearest(
            link_geo,
            roads,
            how="left",
            max_distance=60.0,
            distance_col="official_match_m",
        )
        near = near.sort_values(["link_id", "official_match_m"]).drop_duplicates("link_id")
        names = near.set_index("link_id")[field].to_dict()
        match_dist = near.set_index("link_id")["official_match_m"].to_dict()
        mask = result["link_id"].map(names).fillna("").astype(str).str.strip().ne("")
        result.loc[mask, "display_name"] = result.loc[mask, "link_id"].map(names)
        result.loc[mask, "name_source"] = "LandsD Road Centreline"
        result["official_match_m"] = result["link_id"].map(match_dist)
    except Exception as exc:
        warnings.warn(f"Official road-name matching failed ({exc}); using OSM English names.")

    result.loc[result["display_name"].fillna("").eq(""), "display_name"] = result.loc[
        result["display_name"].fillna("").eq(""), "link_id"
    ].map(lambda x: f"Link {int(x)}")
    return result


# =============================================================================
# Basemap and map decoration
# =============================================================================

def lonlat_to_tile(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    lat = min(85.05112878, max(-85.05112878, float(lat)))
    n = 2 ** int(zoom)
    x = int(math.floor((float(lon) + 180.0) / 360.0 * n))
    lat_rad = math.radians(lat)
    y = int(math.floor((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n))
    return x, y


def tile_bounds_3857(x: int, y: int, zoom: int) -> Tuple[float, float, float, float]:
    lim = 20037508.342789244
    n = 2 ** int(zoom)
    left = -lim + 2 * lim * x / n
    right = -lim + 2 * lim * (x + 1) / n
    top = lim - 2 * lim * y / n
    bottom = lim - 2 * lim * (y + 1) / n
    return left, bottom, right, top


def tile_path(zoom: int, x: int, y: int) -> Path:
    return TILE_CACHE_DIR / str(zoom) / str(x) / f"{y}.png"


def fetch_tile(
    zoom: int,
    x: int,
    y: int,
    refresh: bool,
    timeout: float = 4.0,
    retries: int = 1,
) -> Optional[Image.Image]:
    path = tile_path(zoom, x, y)
    if path.exists() and not refresh:
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass

    url = OSM_TILE_URL.format(z=zoom, x=x, y=y)
    last_exc: Optional[Exception] = None
    # First try without inheriting a potentially problematic system proxy;
    # then allow the environment on the second attempt. This mirrors the CSDI
    # robustness needed elsewhere in this project.
    for trust_env in (False, True):
        session = requests.Session()
        session.trust_env = trust_env
        for attempt in range(retries):
            try:
                r = session.get(url, headers={**OSM_HEADERS, "Connection": "close"}, timeout=timeout)
                r.raise_for_status()
                content_type = str(r.headers.get("content-type", "")).lower()
                if "image" not in content_type:
                    raise RuntimeError(f"OSM returned non-image content-type: {content_type}")
                im = Image.open(io.BytesIO(r.content)).convert("RGB")
                if im.size != (256, 256):
                    raise RuntimeError(f"Unexpected OSM tile size: {im.size}")
                path.parent.mkdir(parents=True, exist_ok=True)
                im.save(path)
                return im
            except Exception as exc:
                last_exc = exc
                time.sleep(0.35 * (attempt + 1))
    if path.exists():
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    warnings.warn(f"OSM tile unavailable z={zoom} x={x} y={y}: {last_exc}")
    return None


def pale_grayscale(image: Image.Image) -> Image.Image:
    gray = image.convert("L").convert("RGB")
    gray = ImageEnhance.Contrast(gray).enhance(0.50)
    return ImageEnhance.Brightness(gray).enhance(1.18)


def build_basemap(
    bbox_wgs: Dict[str, float],
    zoom: int,
    refresh: bool,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float, float]]]:
    west, south, east, north = (
        float(bbox_wgs["west"]), float(bbox_wgs["south"]),
        float(bbox_wgs["east"]), float(bbox_wgs["north"]),
    )
    x0, ys = lonlat_to_tile(west, south, zoom)
    x1, yn = lonlat_to_tile(east, north, zoom)
    xmin, xmax = min(x0, x1), max(x0, x1)
    ymin, ymax = min(ys, yn), max(ys, yn)

    tile_size = 256
    mosaic = Image.new("RGB", ((xmax - xmin + 1) * tile_size, (ymax - ymin + 1) * tile_size), "#F4F4F1")

    # Probe one tile first. If OSM is unreachable (common on restricted/VPN or
    # proxy connections), abort raster fetching immediately instead of waiting
    # for every tile to time out. Vector-only cartography is then used.
    first = fetch_tile(zoom, xmin, ymin, refresh)
    if first is None:
        warnings.warn("OSM basemap probe failed; using vector-only background for all figures.")
        return None, None
    mosaic.paste(first, (0, 0))
    any_real = True

    for yy in range(ymin, ymax + 1):
        for xx in range(xmin, xmax + 1):
            if xx == xmin and yy == ymin:
                continue
            im = fetch_tile(zoom, xx, yy, refresh)
            if im is not None:
                mosaic.paste(im, ((xx - xmin) * tile_size, (yy - ymin) * tile_size))
    mosaic = pale_grayscale(mosaic)
    left, _, _, top = tile_bounds_3857(xmin, ymin, zoom)
    _, bottom, right, _ = tile_bounds_3857(xmax, ymax, zoom)
    return np.asarray(mosaic), (left, right, bottom, top)


def bbox_from_links(links: gpd.GeoDataFrame, pad_ratio: float = DEFAULT_MAP_PAD) -> Dict[str, float]:
    g = links.to_crs(WGS84)
    minx, miny, maxx, maxy = g.total_bounds
    dx = max(maxx - minx, 1e-5)
    dy = max(maxy - miny, 1e-5)
    return {
        "west": float(minx - pad_ratio * dx),
        "south": float(miny - pad_ratio * dy),
        "east": float(maxx + pad_ratio * dx),
        "north": float(maxy + pad_ratio * dy),
    }


def bbox_to_3857(bbox: Dict[str, float]) -> Tuple[float, float, float, float]:
    t = Transformer.from_crs(WGS84, WEB_MERCATOR, always_xy=True)
    west, south = t.transform(bbox["west"], bbox["south"])
    east, north = t.transform(bbox["east"], bbox["north"])
    return float(west), float(east), float(south), float(north)


def draw_map_base(
    ax: plt.Axes,
    bbox_3857: Tuple[float, float, float, float],
    basemap: Optional[np.ndarray],
    basemap_extent: Optional[Tuple[float, float, float, float]],
) -> None:
    west, east, south, north = bbox_3857
    # Always set a neutral background first. This also covers any small area
    # outside the raster mosaic when a figure (notably hospital accessibility)
    # needs a slightly wider extent than the main road-network map.
    ax.set_facecolor("#F6F6F3")
    if basemap is not None and basemap_extent is not None:
        ax.imshow(basemap, extent=basemap_extent, origin="upper", interpolation="bilinear", zorder=0)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def add_north_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "N", xy=(0.955, 0.945), xytext=(0.955, 0.805),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="center", fontsize=11.5, fontweight="bold",
        color="#111111",
        arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.7, mutation_scale=17),
        zorder=40,
    )


def nice_scale_length(width_m: float) -> float:
    target = width_m * 0.17
    candidates = np.asarray([100, 200, 250, 500, 750, 1000, 1500, 2000, 2500, 5000], dtype=float)
    return float(candidates[int(np.argmin(np.abs(candidates - target)))])


def add_scale_bar(ax: plt.Axes, bbox_3857: Tuple[float, float, float, float]) -> None:
    west, east, south, north = bbox_3857
    width, height = east - west, north - south
    length = nice_scale_length(width)
    x1 = east - 0.045 * width
    x0 = x1 - length
    y = south + 0.050 * height
    cap = 0.009 * height
    ax.plot([x0, x1], [y, y], color="#111111", lw=2.0, zorder=40)
    ax.plot([x0, x0], [y - cap, y + cap], color="#111111", lw=1.1, zorder=40)
    ax.plot([x1, x1], [y - cap, y + cap], color="#111111", lw=1.1, zorder=40)
    label = f"{length/1000:g} km" if length >= 1000 else f"{length:g} m"
    ax.text((x0 + x1) / 2, y + 0.014 * height, label, ha="center", va="bottom", fontsize=7.7,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8), zorder=40)


def decorate_map(ax: plt.Axes, bbox_3857: Tuple[float, float, float, float], basemap_used: bool) -> None:
    add_north_arrow(ax)
    add_scale_bar(ax, bbox_3857)
    if basemap_used:
        ax.text(0.995, 0.006, "© OpenStreetMap contributors", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=5.6, color="#666666", zorder=45)


def panel_label(ax: plt.Axes, label: str, x: float = 0.035, y: float = 1.015) -> None:
    """Plain panel label placed just above the plotting area."""
    ax.text(
        x, y, label, transform=ax.transAxes, ha="left", va="bottom",
        fontsize=10.0, fontweight="bold", color="#222222", zorder=60,
        clip_on=False,
    )


def panel_title(ax: plt.Axes, label: str, title: str, pad: float = 5.0) -> None:
    """Consistent report panel title, e.g. ``(a) BC``."""
    ax.set_title(f"({label}) {title}", loc="left", fontweight="bold", pad=pad)

def _safe_offset_geometry(geom, distance_m: float):
    if geom is None or geom.is_empty or abs(float(distance_m)) < 1e-12:
        return geom
    try:
        side = "left" if distance_m > 0 else "right"
        shifted = geom.parallel_offset(abs(float(distance_m)), side=side, join_style=2, mitre_limit=5.0)
        if shifted is None or shifted.is_empty:
            return geom
        if shifted.geom_type in {"LineString", "LinearRing"}:
            return shifted
        if hasattr(shifted, "geoms") and len(shifted.geoms):
            return max(list(shifted.geoms), key=lambda x: x.length)
    except Exception:
        pass
    return geom


def apply_directional_offset(gdf: gpd.GeoDataFrame, offset_m: float = 2.8) -> gpd.GeoDataFrame:
    if "from_node" not in gdf.columns or "to_node" not in gdf.columns:
        return gdf
    out = gdf.copy().reset_index(drop=True)
    u = pd.to_numeric(out["from_node"], errors="coerce")
    v = pd.to_numeric(out["to_node"], errors="coerce")
    out["_pair"] = list(zip(np.minimum(u, v), np.maximum(u, v)))
    shifts = np.zeros(len(out), dtype=float)
    for _, idx in out.groupby("_pair", sort=False).groups.items():
        seq = list(idx)
        if len(seq) <= 1:
            continue
        pos = [i for i in seq if u.iloc[i] <= v.iloc[i]]
        neg = [i for i in seq if u.iloc[i] > v.iloc[i]]
        if pos and neg:
            for j, i in enumerate(sorted(pos)):
                shifts[i] = +offset_m * (1.0 + 0.28 * j)
            for j, i in enumerate(sorted(neg)):
                shifts[i] = -offset_m * (1.0 + 0.28 * j)
        else:
            center = 0.5 * (len(seq) - 1)
            for j, i in enumerate(sorted(seq)):
                shifts[i] = (j - center) * offset_m
    out["geometry"] = [_safe_offset_geometry(g, d) for g, d in zip(out.geometry, shifts)]
    return out.drop(columns=["_pair"], errors="ignore")


def positive_rank(series: pd.Series, zero_tolerance: float = 1e-15) -> pd.Series:
    """0-1 display rank where exact/near-zero values remain exactly 0.

    This avoids the misleading behaviour of percentile ranking when many LSCC
    losses are zero (especially for directed links).
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    neutral = s.notna() & (s <= zero_tolerance)
    out.loc[neutral] = 0.0
    active = s.notna() & (s > zero_tolerance)
    if active.any():
        out.loc[active] = s.loc[active].rank(method="average", pct=True, ascending=True)
    return out


def add_visual_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for metric in METRICS:
        out[f"{metric}_visual"] = positive_rank(out[metric])
    return out


def plot_network_underlay(ax: plt.Axes, links_3857: gpd.GeoDataFrame, lw: float = 0.55, alpha: float = 0.55) -> None:
    links_3857.plot(ax=ax, color="#777B7C", linewidth=lw, alpha=alpha, zorder=2)


# =============================================================================
# Tables
# =============================================================================

def table_network_overview(
    nodes: pd.DataFrame,
    links: pd.DataFrame,
    od_meta: Dict[str, Any],
    graph_meta: Dict[str, Any],
    access_meta: Dict[str, Any],
) -> pd.DataFrame:
    total_km = float(numeric(links, "length_m").sum() / 1000.0)
    median_m = float(numeric(links, "length_m").median())
    b = graph_meta["baseline"]
    rows = [
        ("Road-network nodes", len(nodes), "nodes"),
        ("Directed road links", len(links), "links"),
        ("Unique ordered node pairs", graph_meta["graph"].get("unique_ordered_node_pairs", np.nan), "pairs"),
        ("Total directed road length", total_km, "km"),
        ("Median link length", median_m, "m"),
        ("TPUSB zones intersecting computational domain", od_meta["domain"].get("study_zone_count", np.nan), "zones"),
        ("Road-proxy TPUSB OD cells", od_meta["od"].get("domain_zone_od_count", np.nan), "zone OD cells"),
        ("Road-proxy TPUSB OD flow", od_meta["od"].get("domain_road_proxy_flow", np.nan), "weighted person trips/reference weekday"),
        ("Active connector OD pairs", b.get("active_od_pairs", np.nan), "connector OD pairs"),
        ("Active connector demand", b.get("active_connector_demand", np.nan), "weighted person trips/reference weekday"),
        ("Baseline OD reachability", 100.0 * b.get("baseline_reachable_demand_share", np.nan), "% of connector demand"),
        ("Baseline GNE", b.get("od_efficiency", np.nan) * 1000.0, "km^-1"),
        ("Baseline LSCC share", 100.0 * b.get("lscc_share", np.nan), "% of nodes"),
        ("Baseline TTD", b.get("total_distance_m_trip", np.nan) / 1e9, "million km-trip"),
        ("Hospital-accessibility grid", access_meta["study_domain"].get("grid_cell_count", np.nan), "100 m cells"),
        ("Hospital candidates", access_meta["hospitals"].get("count", np.nan), "hospitals"),
    ]
    out = pd.DataFrame(rows, columns=["Indicator", "Value", "Unit"])
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce").round(4)
    return out


def table_od_domain(od_meta: Dict[str, Any], od_zone: pd.DataFrame, connectors: pd.DataFrame) -> pd.DataFrame:
    m = od_meta
    road_modes = "|".join(m["method"].get("road_modes", []))
    rows = [
        ("Computational domain", "Full current road-network envelope + OD boundary buffer", ""),
        ("Shortest-path impedance", "Road distance", "length_m"),
        ("TPUSB zones", m["domain"].get("study_zone_count"), "zones"),
        ("Target road connectors per TPUSB", m["domain"].get("connectors_per_zone_target"), "nodes/zone"),
        ("Generated TPUSB connectors", m["domain"].get("connector_count"), "connectors"),
        ("Zones using nearest-node fallback", m["domain"].get("zones_using_nearest_fallback"), "zones"),
        ("Road-mode codes used for proxy", road_modes, "HKTCS main_mode codes"),
        ("All-mode OD flow in domain", m["od"].get("domain_allmode_flow"), "weighted person trips/reference weekday"),
        ("Road-proxy OD flow in domain", m["od"].get("domain_road_proxy_flow"), "weighted person trips/reference weekday"),
        ("Road-proxy OD cells", len(od_zone), "OD cells"),
        ("Connector OD records", len(connectors), "connector OD records"),
        ("Connector demand total", m["connector_od"].get("connector_demand_total"), "weighted person trips/reference weekday"),
        ("Dropped same-node intrazonal flow", m["connector_od"].get("dropped_single_connector_intrazonal_flow"), "weighted person trips/reference weekday"),
    ]
    return pd.DataFrame(rows, columns=["Item", "Value", "Unit_or_note"])


def table_road_class(link_metric: pd.DataFrame) -> pd.DataFrame:
    work = link_metric.copy()
    work["Road Class"] = work.get("highway", "unknown").fillna("unknown").astype(str)
    work["Length (km)"] = numeric(work, "length_m") / 1000.0
    out = work.groupby("Road Class", as_index=False).agg(
        **{"Directed Links": ("link_id", "count"), "Directed Length (km)": ("Length (km)", "sum")}
    )
    total = out["Directed Length (km)"].sum()
    out["Length Share (%)"] = 100.0 * out["Directed Length (km)"] / total if total > 0 else np.nan
    out = out.sort_values("Directed Length (km)", ascending=False).reset_index(drop=True)
    out["Directed Length (km)"] = out["Directed Length (km)"].round(2)
    out["Length Share (%)"] = out["Length Share (%)"].round(1)
    return out


def top_by_metric(df: pd.DataFrame, entity: str, top_n: int) -> pd.DataFrame:
    id_col = "node_id" if entity == "node" else "link_id"
    rows: List[Dict[str, Any]] = []
    for metric, spec in METRICS.items():
        work = df.copy()
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        work = work.sort_values([metric, id_col], ascending=[False, True]).head(top_n)
        for rank, (_, r) in enumerate(work.iterrows(), 1):
            row: Dict[str, Any] = {
                "Metric": spec["short"],
                "Rank": rank,
                "Component ID": int(r[id_col]),
                "BC": float(r["odbc"]),
                "GNE Loss (%)": 100.0 * float(r["od_eff_loss"]),
                "LSCC Loss (%)": 100.0 * float(r["lscc_loss"]),
                "TTD Loss (%)": 100.0 * float(r["td_loss"]),
                "Reachable OD Demand after Removal (%)": 100.0 * float(r["reachable_demand_share"]),
            }
            if entity == "node":
                row["OSM Node ID"] = r.get("osm_id", np.nan)
                row["Longitude"] = r.get("lon", np.nan)
                row["Latitude"] = r.get("lat", np.nan)
            else:
                row["Road Name"] = r.get("display_name", r.get("name", ""))
                row["Road Class"] = r.get("highway", "")
                row["From Node"] = r.get("from_node", np.nan)
                row["To Node"] = r.get("to_node", np.nan)
                row["Length (m)"] = r.get("length_m", np.nan)
            rows.append(row)
    return pd.DataFrame(rows)

def table_robustness_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize retained-performance robustness plus TTD-loss robustness.

    Retention metrics: higher AUC is better; threshold columns report removal
    fractions at 90%, 80% and 50% retained performance.
    TTD loss: lower AUC is better; threshold columns report removal fractions
    at 10%, 25% and 50% TTD increase. Upstream removal.py stores these three
    TTD thresholds in f90/f80/f50 for schema compatibility.
    """
    metric_map = {
        "od_eff_norm": "GNE retained",
        "lscc_norm": "LSCC retained",
        "reachable_demand_share": "Reachable OD demand",
        "td_loss": "TTD loss",
    }
    out = summary[summary["metric"].isin(metric_map)].copy()
    rows: List[Dict[str, Any]] = []
    for _, r in out.iterrows():
        metric = str(r["metric"])
        row: Dict[str, Any] = {
            "Component": {"node": "Node", "link": "Link"}.get(str(r["component"]), str(r["component"])),
            "Attack Strategy": STRATEGY_STYLE.get(str(r["strategy"]), {"label": str(r["strategy"])})["label"],
            "Performance Metric": metric_map[metric],
            "AUC": float(r["auc"]) if pd.notna(r["auc"]) else np.nan,
            "AUC Interpretation": "Lower is better" if metric == "td_loss" else "Higher is better",
        }
        if metric == "td_loss":
            row["Threshold 1"] = "TTD loss = 10%"
            row["Removal at Threshold 1 (%)"] = 100.0 * float(r["f90"]) if pd.notna(r["f90"]) else np.nan
            row["Threshold 2"] = "TTD loss = 25%"
            row["Removal at Threshold 2 (%)"] = 100.0 * float(r["f80"]) if pd.notna(r["f80"]) else np.nan
            row["Threshold 3"] = "TTD loss = 50%"
            row["Removal at Threshold 3 (%)"] = 100.0 * float(r["f50"]) if pd.notna(r["f50"]) else np.nan
        else:
            row["Threshold 1"] = "Retained = 90%"
            row["Removal at Threshold 1 (%)"] = 100.0 * float(r["f90"]) if pd.notna(r["f90"]) else np.nan
            row["Threshold 2"] = "Retained = 80%"
            row["Removal at Threshold 2 (%)"] = 100.0 * float(r["f80"]) if pd.notna(r["f80"]) else np.nan
            row["Threshold 3"] = "Retained = 50%"
            row["Removal at Threshold 3 (%)"] = 100.0 * float(r["f50"]) if pd.notna(r["f50"]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).round(4)

def table_hospital_accessibility(summary: pd.DataFrame) -> pd.DataFrame:
    r = summary.iloc[0]
    return pd.DataFrame(
        [
            ("Grid origins", r["origin_count"], "100 m cells"),
            ("Reachable grid origins", r["reachable_count"], "cells"),
            ("Reachable share", 100.0 * r["reachable_share"], "%"),
            ("Unreachable share", 100.0 * r["unreachable_share"], "%"),
            ("Mean nearest-hospital network distance", r["mean_km"], "km"),
            ("Median nearest-hospital network distance", r["median_km"], "km"),
            ("P90 nearest-hospital network distance", r["p90_km"], "km"),
            ("P95 nearest-hospital network distance", r["p95_km"], "km"),
        ],
        columns=["Indicator", "Value", "Unit"],
    ).round(4)


def table_hospital_catchments(catchment: pd.DataFrame) -> pd.DataFrame:
    out = catchment.copy()
    out["Grid Share (%)"] = 100.0 * numeric(out, "grid_share")
    out["Mean Distance (km)"] = numeric(out, "mean_m") / 1000.0
    out["Median Distance (km)"] = numeric(out, "median_m") / 1000.0
    out["P90 Distance (km)"] = numeric(out, "p90_m") / 1000.0
    return out.rename(
        columns={
            "hospital_id": "Hospital ID",
            "institution_eng": "Hospital",
            "with_ae": "A&E",
            "grid_count": "Assigned Grid Cells",
        }
    )[["Hospital ID", "Hospital", "A&E", "Assigned Grid Cells", "Grid Share (%)", "Mean Distance (km)", "Median Distance (km)", "P90 Distance (km)"]].round(4)


def hospital_critical_table(
    impact: pd.DataFrame,
    entity: str,
    metric_df: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    sub = impact[impact["component"] == entity].copy()
    sub = sub.sort_values(["reach_loss", "p90_increase_m", "mean_increase_m"], ascending=False).head(top_n)
    id_col = "node_id" if entity == "node" else "link_id"
    info = metric_df.copy().rename(columns={id_col: "component_id"})
    keep = [c for c in ["component_id", "display_name", "name", "highway", "from_node", "to_node", "osm_id", "odbc", "od_eff_loss", "lscc_loss", "td_loss"] if c in info.columns]
    out = sub.merge(info[keep], on="component_id", how="left", suffixes=("", "_metric"))
    out.insert(0, "Rank", np.arange(1, len(out) + 1))
    out["Hospital Accessibility Loss (%)"] = 100.0 * numeric(out, "reach_loss")
    out["Mean Hospital Distance Change (m)"] = numeric(out, "mean_increase_m")
    out["P90 Hospital Distance Change (m)"] = numeric(out, "p90_increase_m")
    out["GNE Loss (%)"] = 100.0 * numeric(out, "od_eff_loss")
    out["LSCC Loss (%)"] = 100.0 * numeric(out, "lscc_loss")
    out["TTD Loss (%)"] = 100.0 * numeric(out, "td_loss")
    out = out.rename(columns={"component_id": "Component ID", "odbc": "BC"})
    cols = ["Rank", "Component ID"]
    if entity == "link":
        if "display_name" in out.columns:
            out = out.rename(columns={"display_name": "Road Name"})
            cols += ["Road Name"]
        cols += [c for c in ["highway", "from_node", "to_node"] if c in out.columns]
    else:
        if "osm_id" in out.columns:
            out = out.rename(columns={"osm_id": "OSM Node ID"})
            cols += ["OSM Node ID"]
    cols += [
        "Hospital Accessibility Loss (%)", "Mean Hospital Distance Change (m)",
        "P90 Hospital Distance Change (m)", "BC", "GNE Loss (%)", "LSCC Loss (%)", "TTD Loss (%)",
    ]
    return out[cols].round(4)

def write_tables(
    node_metric: pd.DataFrame,
    link_metric: pd.DataFrame,
    od_meta: Dict[str, Any],
    graph_meta: Dict[str, Any],
    removal_summary: pd.DataFrame,
    access_meta: Dict[str, Any],
    hospital_summary: pd.DataFrame,
    hospital_catchment: pd.DataFrame,
    access_impact: pd.DataFrame,
    access_removal_summary: Optional[pd.DataFrame],
    od_zone: pd.DataFrame,
    od_connector: pd.DataFrame,
    top_n: int,
) -> List[Path]:
    tables = {
        "Tab_01_Network_Overview.csv": table_network_overview(node_metric, link_metric, od_meta, graph_meta, access_meta),
        "Tab_02_OD_and_Study_Domain.csv": table_od_domain(od_meta, od_zone, od_connector),
        "Tab_03_Road_Class_Composition.csv": table_road_class(link_metric),
        "Tab_04_Top_Nodes_by_Metric.csv": top_by_metric(node_metric, "node", top_n),
        "Tab_05_Top_Links_by_Metric.csv": top_by_metric(link_metric, "link", top_n),
        "Tab_06_Robustness_Summary.csv": table_robustness_summary(removal_summary),
        "Tab_07_Hospital_Accessibility.csv": table_hospital_accessibility(hospital_summary),
        "Tab_08_Hospital_Catchments.csv": table_hospital_catchments(hospital_catchment),
        "Tab_09_Hospital_Critical_Nodes.csv": hospital_critical_table(access_impact, "node", node_metric, top_n),
        "Tab_10_Hospital_Critical_Links.csv": hospital_critical_table(access_impact, "link", link_metric, top_n),
    }
    paths: List[Path] = []
    for name, table in tables.items():
        path = OUT_DIR / name
        table.to_csv(path, index=False, encoding="utf-8-sig")
        paths.append(path)
    return paths

def style_cartesian_ax(ax: plt.Axes) -> None:
    ax.grid(True, color="#D9D9D9", lw=0.65, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_strategy_curves(
    ax: plt.Axes,
    group: pd.DataFrame,
    y_col: str,
    y_transform=lambda x: x,
    interval: bool = True,
) -> None:
    for strategy in STRATEGY_ORDER:
        sub = group[group["strategy"] == strategy].sort_values("fraction")
        if sub.empty:
            continue
        style = STRATEGY_STYLE[strategy]
        x = 100.0 * numeric(sub, "fraction").to_numpy(dtype=float)
        y = np.asarray(y_transform(numeric(sub, y_col).to_numpy(dtype=float)), dtype=float)
        ax.plot(x, y, label=style["label"], color=style["color"], linestyle=style["linestyle"], lw=style["lw"], zorder=5)
        lo_col, hi_col = f"{y_col}_p025", f"{y_col}_p975"
        if interval and strategy == "random" and lo_col in sub.columns and hi_col in sub.columns:
            lo = np.asarray(y_transform(numeric(sub, lo_col).to_numpy(dtype=float)), dtype=float)
            hi = np.asarray(y_transform(numeric(sub, hi_col).to_numpy(dtype=float)), dtype=float)
            ax.fill_between(x, lo, hi, color=style["color"], alpha=0.13, linewidth=0, zorder=3)


def strategy_legend_handles() -> List[Line2D]:
    return [
        Line2D([0], [0], color=STRATEGY_STYLE[s]["color"], linestyle=STRATEGY_STYLE[s]["linestyle"], lw=STRATEGY_STYLE[s]["lw"], label=STRATEGY_STYLE[s]["label"])
        for s in STRATEGY_ORDER
    ]


# =============================================================================
# Figure 00: study domain + Ho Man Tin focal area
# =============================================================================

def figure_study_domain(
    links: gpd.GeoDataFrame,
    hmt_boundary: gpd.GeoDataFrame,
    bbox_3857: Tuple[float, float, float, float],
    basemap: Optional[np.ndarray],
    basemap_extent: Optional[Tuple[float, float, float, float]],
    dpi: int,
) -> Path:
    """Road network plus a single Ho Man Tin boundary, with no TPUSB overlay."""
    l3857 = links.to_crs(WEB_MERCATOR)
    hmt = hmt_boundary.to_crs(WEB_MERCATOR)

    fig, ax = plt.subplots(figsize=(9.2, 7.8))
    draw_map_base(ax, bbox_3857, basemap, basemap_extent)
    l3857.plot(ax=ax, color="#4F5557", linewidth=0.60, alpha=0.78, zorder=3)
    hmt.plot(
        ax=ax, facecolor="#F2C14E", edgecolor="#9A5B00",
        linewidth=1.45, alpha=0.34, zorder=5,
    )

    ax.set_title("Study area", fontweight="bold", pad=7)
    handles = [
        Line2D([0], [0], color="#4F5557", lw=1.5, label="Road network"),
        Patch(facecolor="#F2C14E", edgecolor="#9A5B00", alpha=0.45, label="Ho Man Tin area"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.92)
    decorate_map(ax, bbox_3857, basemap is not None)
    return save_figure(fig, OUT_DIR / "Fig_00_Study_Area.jpg", dpi)


def annotate_top_nodes(ax: plt.Axes, g: gpd.GeoDataFrame, metric: str, n: int) -> None:
    if n <= 0:
        return
    top = g.sort_values(metric, ascending=False).head(n)
    for _, r in top.iterrows():
        if r.geometry is None or r.geometry.is_empty:
            continue
        ax.annotate(f"N{int(r.node_id)}", xy=(r.geometry.x, r.geometry.y), xytext=(4, 4), textcoords="offset points",
                    fontsize=6.7, color="#17365D", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.78), zorder=30)


def annotate_top_links(ax: plt.Axes, g: gpd.GeoDataFrame, metric: str, n: int) -> None:
    if n <= 0:
        return
    top = g.sort_values(metric, ascending=False).head(max(n * 3, n))
    seen: set[str] = set()
    shown = 0
    for _, r in top.iterrows():
        if shown >= n:
            break
        geom = r.geometry
        if geom is None or geom.is_empty:
            continue
        name = clean_name(r.get("display_name", "")) or f"Link {int(r.link_id)}"
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            p = geom.interpolate(0.5, normalized=True)
        except Exception:
            p = geom.centroid
        ax.annotate(name, xy=(p.x, p.y), xytext=(5, 4), textcoords="offset points",
                    fontsize=6.2, color="#7A0C17",
                    bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.78), zorder=35)
        shown += 1


def figure_node_criticality(
    node_metric: pd.DataFrame,
    nodes: gpd.GeoDataFrame,
    links: gpd.GeoDataFrame,
    bbox_3857: Tuple[float, float, float, float],
    basemap: Optional[np.ndarray],
    basemap_extent: Optional[Tuple[float, float, float, float]],
    label_top: int,
    dpi: int,
) -> Path:
    g = merge_metric_geometry(nodes, node_metric, "node_id")
    g = add_visual_ranks(g).to_crs(WEB_MERCATOR)
    l = links.to_crs(WEB_MERCATOR)
    display_norm = PowerNorm(gamma=CRITICALITY_DISPLAY_GAMMA, vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.4))
    axes = axes.ravel()
    for i, (metric, spec) in enumerate(METRICS.items()):
        ax = axes[i]
        draw_map_base(ax, bbox_3857, basemap, basemap_extent)
        plot_network_underlay(ax, l, lw=0.45, alpha=0.40)
        rank = g[f"{metric}_visual"].fillna(0.0).clip(0, 1)
        sizes = 4.5 + 34.0 * np.square(rank)
        g.plot(
            ax=ax, column=f"{metric}_visual", cmap=NODE_CMAP, norm=display_norm,
            markersize=sizes, alpha=0.90, linewidth=0.10, edgecolor="#123B61", zorder=8,
        )
        annotate_top_nodes(ax, g, metric, label_top)
        panel_title(ax, chr(ord("a") + i), spec["short"])
        decorate_map(ax, bbox_3857, basemap is not None)

    sm = mpl.cm.ScalarMappable(norm=display_norm, cmap=NODE_CMAP)
    sm.set_array([])
    cax = fig.add_axes([0.915, 0.17, 0.018, 0.66])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("Normalized node criticality")
    set_fractional_colorbar_ticks(cbar, 0.0, 1.0, decimals=2)
    fig.suptitle("Node criticality", y=0.992, fontweight="bold")
    fig.subplots_adjust(left=0.035, right=0.895, top=0.94, bottom=0.035, hspace=0.09, wspace=0.05)
    return save_figure(fig, OUT_DIR / "Fig_01_Node_Criticality.jpg", dpi)

def figure_link_criticality(
    link_metric: pd.DataFrame,
    links: gpd.GeoDataFrame,
    bbox_3857: Tuple[float, float, float, float],
    basemap: Optional[np.ndarray],
    basemap_extent: Optional[Tuple[float, float, float, float]],
    label_top: int,
    dpi: int,
) -> Path:
    g = merge_metric_geometry(links, link_metric, "link_id")
    g = add_visual_ranks(g).to_crs(WEB_MERCATOR)
    g = apply_directional_offset(g, offset_m=2.8)
    under = links.to_crs(WEB_MERCATOR)
    display_norm = PowerNorm(gamma=CRITICALITY_DISPLAY_GAMMA, vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.4))
    axes = axes.ravel()
    for i, (metric, spec) in enumerate(METRICS.items()):
        ax = axes[i]
        draw_map_base(ax, bbox_3857, basemap, basemap_extent)
        plot_network_underlay(ax, under, lw=0.45, alpha=0.36)
        rank = g[f"{metric}_visual"].fillna(0.0).clip(0, 1)
        bins = np.linspace(0, 1, 10)
        for j in range(len(bins) - 1):
            lo, hi = bins[j], bins[j + 1]
            mask = (rank >= lo) & ((rank < hi) if j < len(bins) - 2 else (rank <= hi))
            sub = g.loc[mask]
            if sub.empty:
                continue
            mid = 0.5 * (lo + hi)
            visual_mid = float(display_norm(mid))
            sub.plot(
                ax=ax, color=mpl.colormaps[LINK_CMAP](visual_mid),
                linewidth=0.40 + 3.05 * (mid ** 1.65), alpha=0.90, zorder=7 + j,
            )
        annotate_top_links(ax, g, metric, label_top)
        panel_title(ax, chr(ord("a") + i), spec["short"])
        decorate_map(ax, bbox_3857, basemap is not None)

    sm = mpl.cm.ScalarMappable(norm=display_norm, cmap=LINK_CMAP)
    sm.set_array([])
    cax = fig.add_axes([0.915, 0.17, 0.018, 0.66])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("Normalized link criticality")
    set_fractional_colorbar_ticks(cbar, 0.0, 1.0, decimals=2)
    fig.suptitle("Link criticality", y=0.992, fontweight="bold")
    fig.subplots_adjust(left=0.035, right=0.895, top=0.94, bottom=0.035, hspace=0.09, wspace=0.05)
    return save_figure(fig, OUT_DIR / "Fig_02_Link_Criticality.jpg", dpi)

def figure_network_robustness(curve: pd.DataFrame, dpi: int) -> Path:
    specs = [
        ("od_eff_norm", "GNE retained (%)"),
        ("lscc_norm", "LSCC retained (%)"),
        ("reachable_demand_share", "Reachable OD demand (%)"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11.6, 9.2), sharex=True, sharey="row")
    letters = "abcdef"
    for row, (metric, ylabel) in enumerate(specs):
        for col, component in enumerate(("node", "link")):
            ax = axes[row, col]
            group = curve[curve["component"] == component]
            plot_strategy_curves(ax, group, metric, lambda x: 100.0 * x, interval=True)
            style_cartesian_ax(ax)
            ax.set_xlim(0, max(40.0, 100.0 * numeric(group, "fraction").max()))
            ax.set_ylim(0, 102)
            ax.set_yticks(np.arange(0, 101, 20))
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title("Node removal" if component == "node" else "Link removal", fontweight="bold", pad=9)
            if row == len(specs) - 1:
                ax.set_xlabel("Removed components (%)")
            panel_label(ax, f"({letters[row * 2 + col]})", x=0.035, y=1.015)

    fig.legend(
        handles=strategy_legend_handles(), loc="upper center", ncol=5,
        frameon=False, bbox_to_anchor=(0.5, 0.987), columnspacing=1.5, handlelength=2.8,
    )
    fig.suptitle("Network robustness under random and targeted removal", y=1.018, fontweight="bold")
    fig.subplots_adjust(top=0.925, bottom=0.075, left=0.085, right=0.985, hspace=0.25, wspace=0.12)
    return save_figure(fig, OUT_DIR / "Fig_03_Network_Robustness.jpg", dpi)

def figure_criticality_bubble(node_metric: pd.DataFrame, link_metric: pd.DataFrame, dpi: int) -> Path:
    """Two complementary BC-GNE bubble views for nodes and links.

    Row 1: bubble size = normalized LSCC criticality; color = normalized TTD.
    Row 2: bubble size = normalized TTD criticality; color = normalized LSCC.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 10.1), sharex=True, sharey=True)
    entities = [
        ("Node", node_metric, NODE_CMAP),
        ("Link", link_metric, LINK_CMAP),
    ]
    size_min, size_span = 9.0, 125.0

    for row in range(2):
        for col, (entity, df, cmap) in enumerate(entities):
            ax = axes[row, col]
            work = add_visual_ranks(df)
            x = work["odbc_visual"].fillna(0).clip(0, 1).to_numpy(dtype=float)
            y = work["od_eff_loss_visual"].fillna(0).clip(0, 1).to_numpy(dtype=float)
            lscc = work["lscc_loss_visual"].fillna(0).clip(0, 1).to_numpy(dtype=float)
            ttd = work["td_loss_visual"].fillna(0).clip(0, 1).to_numpy(dtype=float)

            if row == 0:
                size_metric, color_metric = lscc, ttd
                size_name, color_name = "LSCC", "TTD"
            else:
                size_metric, color_metric = ttd, lscc
                size_name, color_name = "TTD", "LSCC"

            size = size_min + size_span * np.power(size_metric, 1.45)
            sc = ax.scatter(
                x, y, s=size, c=color_metric, cmap=cmap, vmin=0, vmax=1,
                alpha=0.45, edgecolors="none", rasterized=True,
            )
            style_cartesian_ax(ax)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks(np.arange(0, 1.01, 0.2))
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            if row == 1:
                ax.set_xlabel("Normalized BC criticality")
            if col == 0:
                ax.set_ylabel("Normalized GNE criticality")
            letter = chr(ord("a") + row * 2 + col)
            panel_title(ax, letter, f"{entity}: {size_name}-sized bubbles")
            cb = fig.colorbar(sc, ax=ax, orientation="vertical", fraction=0.034, pad=0.022, shrink=0.90)
            cb.set_label(f"Normalized {color_name} criticality")
            cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    # Separate size legends for the two rows. Keeping only four reference sizes
    # avoids obscuring the scatter while making the fourth metric explicit.
    levels = (0.25, 0.50, 0.75, 1.00)
    for row, metric_name in ((0, "LSCC"), (1, "TTD")):
        handles = [
            plt.scatter([], [], s=size_min + size_span * (v ** 1.45),
                        facecolor="#777777", alpha=0.45, edgecolor="none", label=f"{v:.2f}")
            for v in levels
        ]
        axes[row, 1].legend(
            handles=handles,
            title=f"Normalized {metric_name} criticality\n(bubble size)",
            loc="lower right", frameon=True, facecolor="white", framealpha=0.88,
            edgecolor="#CCCCCC", ncol=1, fontsize=7.5, title_fontsize=7.8,
            borderpad=0.45, labelspacing=0.45, handletextpad=0.55,
        )

    fig.suptitle("Multi-metric criticality", y=0.992, fontweight="bold")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.935, bottom=0.075, hspace=0.20, wspace=0.14)
    return save_figure(fig, OUT_DIR / "Fig_04_Multi_Metric_Criticality.jpg", dpi)

def figure_structural_accessibility(
    node_access: pd.DataFrame,
    nodes: gpd.GeoDataFrame,
    links: gpd.GeoDataFrame,
    bbox_3857: Tuple[float, float, float, float],
    basemap: Optional[np.ndarray],
    basemap_extent: Optional[Tuple[float, float, float, float]],
    dpi: int,
) -> Path:
    g = merge_metric_geometry(nodes, node_access, "node_id").to_crs(WEB_MERCATOR)
    l = links.to_crs(WEB_MERCATOR)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))

    ax = axes[0]
    draw_map_base(ax, bbox_3857, basemap, basemap_extent)
    plot_network_underlay(ax, l, lw=0.48, alpha=0.40)
    rank = numeric(g, "harmonic_access").rank(method="average", pct=True).fillna(0.0)
    sizes = 5.0 + 38.0 * np.square(rank)
    values = numeric(g, "harmonic_access")
    vmin = float(values.min())
    vmax = float(values.max())
    access_norm = PowerNorm(gamma=ACCESSIBILITY_DISPLAY_GAMMA, vmin=vmin, vmax=vmax)
    g.plot(
        ax=ax, column="harmonic_access", cmap=ACCESS_CMAP, norm=access_norm,
        markersize=sizes, alpha=0.90, linewidth=0.08, edgecolor="#24434A", zorder=7,
    )
    panel_title(ax, "a", "Network accessibility")
    decorate_map(ax, bbox_3857, basemap is not None)
    sm = mpl.cm.ScalarMappable(norm=access_norm, cmap=ACCESS_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.034, pad=0.020, shrink=0.86)
    cb.set_label("Harmonic accessibility (km$^{-1}$)")
    set_fractional_colorbar_ticks(cb, vmin, vmax, decimals=2)

    ax = axes[1]
    vals = numeric(node_access, "harmonic_access").dropna()
    ax.hist(vals, bins=35, color="#4C78A8", alpha=0.86, edgecolor="white", linewidth=0.35)
    if len(vals):
        ax.axvline(vals.median(), color="#222222", linestyle="--", lw=1.25, label=f"Median = {vals.median():.3f}")
    style_cartesian_ax(ax)
    ax.set_xlabel("Harmonic accessibility (km$^{-1}$)")
    ax.set_ylabel("Nodes")
    panel_title(ax, "b", "Distribution of network accessibility")
    ax.legend(frameon=False, loc="upper right")

    fig.suptitle("Network accessibility", y=0.995, fontweight="bold")
    fig.subplots_adjust(top=0.89, bottom=0.12, left=0.045, right=0.98, wspace=0.20)
    return save_figure(fig, OUT_DIR / "Fig_05_Network_Accessibility.jpg", dpi)

def short_hospital_name(name: str) -> str:
    """Stable English abbreviations used consistently in hospital maps/legends."""
    s = clean_name(name)
    replacements = {
        "Hong Kong Children's Hospital": "Children's Hosp.",
        "Hong Kong Buddhist Hospital": "Buddhist Hosp.",
        "Hong Kong Eye Hospital": "Eye Hosp.",
        "Queen Elizabeth Hospital": "Queen Elizabeth Hosp.",
        "Kowloon Hospital": "Kowloon Hosp.",
        "Kwong Wah Hospital": "Kwong Wah Hosp.",
        "Our Lady of Maryknoll Hospital": "Maryknoll Hosp.",
        "United Christian Hospital": "United Christian Hosp.",
        "Tsan Yuk Hospital": "Tsan Yuk Hosp.",
        "Tung Wah Eastern Hospital": "Tung Wah Eastern Hosp.",
        "Tung Wah Hospital": "Tung Wah Hosp.",
        "Ruttonjee Hospital": "Ruttonjee Hosp.",
        "Tang Shiu Kin Hospital": "Tang Shiu Kin Hosp.",
        "Kai Tak Hospital": "Kai Tak Hosp.",
    }
    return replacements.get(s, re.sub(r"\bHospital\b", "Hosp.", s))


def active_hospital_ids_from_access(hospital_access: pd.DataFrame) -> List[int]:
    """Hospitals that actually receive at least one accessible 100 m grid cell."""
    if "nearest_hospital_id" not in hospital_access.columns:
        return []
    ids = pd.to_numeric(
        hospital_access.loc[bool_series(hospital_access["reachable"]), "nearest_hospital_id"],
        errors="coerce",
    ).dropna().astype(int)
    return sorted(int(x) for x in ids[ids >= 0].unique())


def expand_map_bbox_to_active_hospitals(
    base_bbox: Tuple[float, float, float, float],
    hospitals_3857: gpd.GeoDataFrame,
    active_ids: Sequence[int],
    pad_ratio: float = 0.035,
) -> Tuple[float, float, float, float]:
    """Expand a map bbox only enough to include hospitals with nonzero catchment."""
    west, east, south, north = [float(x) for x in base_bbox]
    if not active_ids or hospitals_3857.empty or "hospital_id" not in hospitals_3857.columns:
        return west, east, south, north

    hid = pd.to_numeric(hospitals_3857["hospital_id"], errors="coerce")
    active = hospitals_3857[hid.isin([int(x) for x in active_ids])].copy()
    active = active[active.geometry.notna() & (~active.geometry.is_empty)]
    if active.empty:
        return west, east, south, north

    minx, miny, maxx, maxy = active.total_bounds
    west = min(west, float(minx))
    east = max(east, float(maxx))
    south = min(south, float(miny))
    north = max(north, float(maxy))

    width = max(east - west, 1.0)
    height = max(north - south, 1.0)
    return (
        west - pad_ratio * width,
        east + pad_ratio * width,
        south - pad_ratio * height,
        north + pad_ratio * height,
    )


def bbox_3857_to_wgs84(bbox: Tuple[float, float, float, float]) -> Dict[str, float]:
    west, east, south, north = [float(x) for x in bbox]
    t = Transformer.from_crs(WEB_MERCATOR, WGS84, always_xy=True)
    lon_w, lat_s = t.transform(west, south)
    lon_e, lat_n = t.transform(east, north)
    return {
        "west": float(lon_w),
        "south": float(lat_s),
        "east": float(lon_e),
        "north": float(lat_n),
    }


def annotate_hospitals(
    ax: plt.Axes,
    h: gpd.GeoDataFrame,
    bbox: Tuple[float, float, float, float],
    hospital_ids: Optional[Sequence[int]] = None,
) -> None:
    """Annotate selected hospitals using the same abbreviations as catchment legends."""
    west, east, south, north = bbox
    width, height = east - west, north - south
    inside = h.cx[west:east, south:north].copy()
    if hospital_ids is not None and "hospital_id" in inside.columns:
        allowed = {int(x) for x in hospital_ids}
        ids = pd.to_numeric(inside["hospital_id"], errors="coerce")
        inside = inside[ids.isin(allowed)]
    for _, r in inside.iterrows():
        if r.geometry is None or r.geometry.is_empty:
            continue
        x = float(r.geometry.x)
        y = float(r.geometry.y)
        xn = (x - west) / max(width, 1e-12)
        yn = (y - south) / max(height, 1e-12)

        # Put labels inward from the map frame. In particular, bottom-right
        # facilities are labelled above/left of their marker so they do not
        # collide with the lower-right scale bar.
        if xn > 0.72:
            tx = x - 0.009 * width
            ha = "right"
        else:
            tx = x + 0.009 * width
            ha = "left"

        if yn < 0.18:
            ty = y + 0.012 * height
            va = "bottom"
        elif yn > 0.88:
            ty = y - 0.012 * height
            va = "top"
        else:
            ty = y + 0.006 * height
            va = "bottom"

        ax.text(
            tx, ty,
            short_hospital_name(r.get("institution_eng", "Hospital")),
            fontsize=5.9,
            ha=ha,
            va=va,
            color="#5A0015",
            bbox=dict(boxstyle="round,pad=0.09", fc="white", ec="none", alpha=0.78),
            zorder=25,
        )


def figure_hospital_accessibility(
    hospital_access: pd.DataFrame,
    grid_geom: gpd.GeoDataFrame,
    hospital_gdf: gpd.GeoDataFrame,
    links: gpd.GeoDataFrame,
    bbox_3857: Tuple[float, float, float, float],
    basemap: Optional[np.ndarray],
    basemap_extent: Optional[Tuple[float, float, float, float]],
    dpi: int,
    zoom: int = DEFAULT_ZOOM,
    refresh_basemap: bool = False,
) -> Path:
    grid = grid_geom[["grid_id", "geometry"]].copy()
    grid["grid_id"] = pd.to_numeric(grid["grid_id"], errors="raise").astype(np.int64)
    acc = hospital_access.copy()
    acc["grid_id"] = pd.to_numeric(acc["grid_id"], errors="raise").astype(np.int64)
    grid = grid.merge(
        acc[["grid_id", "reachable", "distance_km", "nearest_hospital_id"]],
        on="grid_id", how="left",
    )
    grid = gpd.GeoDataFrame(grid, geometry="geometry", crs=grid_geom.crs).to_crs(WEB_MERCATOR)
    hosp = hospital_gdf.to_crs(WEB_MERCATOR).copy()
    l = links.to_crs(WEB_MERCATOR)

    accessible = grid[bool_series(grid["reachable"])].copy()
    inaccessible = grid[~bool_series(grid["reachable"])].copy()
    active_ids = active_hospital_ids_from_access(acc)
    active_set = set(active_ids)

    # The main study maps use the road-network extent, but a hospital that
    # actually owns a nonzero nearest-hospital catchment may sit just outside
    # that envelope. Expand Fig. 06 only enough to show every such facility.
    hospital_bbox = expand_map_bbox_to_active_hospitals(bbox_3857, hosp, active_ids)

    # If a raster basemap is already available, request a hospital-specific
    # mosaic for the expanded extent. If that request fails, retain the main
    # mosaic; draw_map_base() leaves a neutral background outside its coverage.
    hosp_basemap = basemap
    hosp_basemap_extent = basemap_extent
    if basemap is not None and hospital_bbox != bbox_3857:
        try:
            hb_wgs = bbox_3857_to_wgs84(hospital_bbox)
            rebuilt, rebuilt_extent = build_basemap(hb_wgs, int(zoom), bool(refresh_basemap))
            if rebuilt is not None and rebuilt_extent is not None:
                hosp_basemap, hosp_basemap_extent = rebuilt, rebuilt_extent
        except Exception as exc:
            warnings.warn(
                "Could not extend the OSM basemap to all active hospital locations "
                f"({exc}); using the existing basemap with a neutral background outside it."
            )

    # Separate hospitals that do and do not own at least one catchment cell.
    if "hospital_id" in hosp.columns:
        hosp_ids = pd.to_numeric(hosp["hospital_id"], errors="coerce")
        hosp_active = hosp[hosp_ids.isin(active_set)].copy()
        hosp_inactive = hosp[~hosp_ids.isin(active_set)].copy()
    else:
        hosp_active = hosp.copy()
        hosp_inactive = hosp.iloc[0:0].copy()

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.7))

    # (a) Absolute nearest-hospital network distance.
    ax = axes[0]
    draw_map_base(ax, hospital_bbox, hosp_basemap, hosp_basemap_extent)
    vmax = 1.0
    if not accessible.empty:
        vmax = max(0.5, float(numeric(accessible, "distance_km").max()))
        accessible.plot(
            ax=ax, column="distance_km", cmap=DISTANCE_CMAP, vmin=0, vmax=vmax,
            edgecolor="none", alpha=0.73, zorder=3,
        )
    if not inaccessible.empty:
        inaccessible.plot(ax=ax, facecolor="#D5D5D5", edgecolor="none", alpha=0.58, zorder=3)
    l.plot(ax=ax, color="#55595B", linewidth=0.42, alpha=0.62, zorder=8)

    # Facilities with no assigned catchment are retained only as weak context;
    # labels are reserved for hospitals that actually serve at least one grid cell.
    if not hosp_inactive.empty:
        hosp_inactive.plot(
            ax=ax, marker="P", color="#BDBDBD", edgecolor="white",
            linewidth=0.35, markersize=31, alpha=0.65, zorder=18,
        )
    if not hosp_active.empty:
        hosp_active.plot(
            ax=ax, marker="P", color="#980043", edgecolor="white",
            linewidth=0.45, markersize=49, zorder=20,
        )
    annotate_hospitals(ax, hosp_active, hospital_bbox, active_ids)
    panel_title(ax, "a", "Nearest-hospital network distance")
    decorate_map(ax, hospital_bbox, hosp_basemap is not None)
    if not accessible.empty:
        sm = mpl.cm.ScalarMappable(norm=Normalize(0, vmax), cmap=DISTANCE_CMAP)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.034, pad=0.020, shrink=0.86)
        cb.set_label("Network distance (km)")

    # (b) Nearest-hospital service catchments. Only hospitals with nonzero
    # catchment receive a category/color, preventing zero-catchment facilities
    # from consuming legend colors or being confused with active hospitals.
    ax = axes[1]
    draw_map_base(ax, hospital_bbox, hosp_basemap, hosp_basemap_extent)
    hospital_ids = active_ids
    cmap = plt.get_cmap("tab20", max(1, len(hospital_ids)))
    color_by_id = {hid: cmap(i) for i, hid in enumerate(hospital_ids)}
    nearest_ids = pd.to_numeric(accessible["nearest_hospital_id"], errors="coerce")
    for hid in hospital_ids:
        sub = accessible[nearest_ids == hid]
        if sub.empty:
            continue
        sub.plot(ax=ax, facecolor=color_by_id[hid], edgecolor="none", alpha=0.66, zorder=3)
    if not inaccessible.empty:
        inaccessible.plot(ax=ax, facecolor="#D5D5D5", edgecolor="none", alpha=0.50, zorder=3)
    l.plot(ax=ax, color="#55595B", linewidth=0.40, alpha=0.64, zorder=8)

    # Match each active hospital marker to its catchment color. This makes
    # edge-of-domain facilities such as Tung Wah Eastern Hospital immediately
    # identifiable even when their catchment is relatively small.
    if not hosp_inactive.empty:
        hosp_inactive.plot(
            ax=ax, marker="P", color="#BDBDBD", edgecolor="white",
            linewidth=0.35, markersize=27, alpha=0.55, zorder=18,
        )
    if "hospital_id" in hosp_active.columns:
        for _, r in hosp_active.iterrows():
            if r.geometry is None or r.geometry.is_empty or pd.isna(r.get("hospital_id")):
                continue
            hid = int(r["hospital_id"])
            ax.scatter(
                [r.geometry.x], [r.geometry.y], marker="P", s=54,
                c=[color_by_id.get(hid, "#333333")], edgecolors="white",
                linewidths=0.55, zorder=21,
            )
    panel_title(ax, "b", "Nearest-hospital catchments")
    decorate_map(ax, hospital_bbox, hosp_basemap is not None)

    hospital_name_by_id: Dict[int, str] = {}
    if "hospital_id" in hosp.columns:
        for _, r in hosp.iterrows():
            if pd.isna(r.get("hospital_id")):
                continue
            hospital_name_by_id[int(r["hospital_id"])] = short_hospital_name(
                r.get("institution_eng", "Hospital")
            )
    handles = [
        Patch(
            facecolor=color_by_id[hid], edgecolor="none", alpha=0.76,
            label=hospital_name_by_id.get(hid, f"Hospital {hid}"),
        )
        for hid in hospital_ids
    ]
    if handles:
        ax.legend(
            handles=handles, title="Nearest hospital", loc="lower left",
            frameon=True, facecolor="white", framealpha=0.88, edgecolor="#CCCCCC",
            fontsize=6.0, title_fontsize=6.3, ncol=1, borderpad=0.45,
            labelspacing=0.32, handlelength=1.3,
        )

    # (c) Distribution of absolute nearest-hospital distance.
    ax = axes[2]
    vals = numeric(acc.loc[bool_series(acc["reachable"]), :], "distance_km").dropna()
    ax.hist(vals, bins=32, color="#3B8BC2", alpha=0.87, edgecolor="white", linewidth=0.35)
    if len(vals):
        for q, ls, lab in ((0.5, "--", "Median"), (0.9, "-.", "P90"), (0.95, ":", "P95")):
            v = vals.quantile(q)
            ax.axvline(v, color="#222222", linestyle=ls, lw=1.15, label=f"{lab} = {v:.2f} km")
    style_cartesian_ax(ax)
    ax.set_xlabel("Nearest-hospital network distance (km)")
    ax.set_ylabel("Accessible grid cells")
    panel_title(ax, "c", "Distribution of nearest-hospital distance")
    ax.legend(frameon=False, loc="upper right")
    accessible_share = 100.0 * bool_series(acc["reachable"]).mean()
    ax.text(
        0.98, 0.82, f"Accessible grid share: {accessible_share:.1f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.3,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#BBBBBB", alpha=0.90),
    )

    fig.suptitle("Hospital accessibility", y=0.995, fontweight="bold")
    fig.subplots_adjust(top=0.89, bottom=0.12, left=0.035, right=0.992, wspace=0.17)
    return save_figure(fig, OUT_DIR / "Fig_06_Hospital_Accessibility.jpg", dpi)


def figure_hospital_access_impact(
    access_impact: pd.DataFrame,
    nodes: gpd.GeoDataFrame,
    links: gpd.GeoDataFrame,
    bbox_3857: Tuple[float, float, float, float],
    basemap: Optional[np.ndarray],
    basemap_extent: Optional[Tuple[float, float, float, float]],
    dpi: int,
) -> Path:
    node_imp = access_impact[access_impact["component"] == "node"].copy().rename(columns={"component_id": "node_id"})
    link_imp = access_impact[access_impact["component"] == "link"].copy().rename(columns={"component_id": "link_id"})
    ng = merge_metric_geometry(nodes, node_imp, "node_id").to_crs(WEB_MERCATOR)
    lg = merge_metric_geometry(links, link_imp, "link_id").to_crs(WEB_MERCATOR)
    lg = apply_directional_offset(lg, offset_m=2.8)
    under = links.to_crs(WEB_MERCATOR)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.5))

    # ------------------------------------------------------------------
    # Accessibility loss. A large share of components has exactly zero
    # effect, especially links. Keep zero-impact components in the neutral
    # network underlay and color only nonzero impacts. This avoids a nearly
    # white map without converting the metric to ranks.
    # ------------------------------------------------------------------
    for col, (g, entity, base_cmap) in enumerate(((ng, "Node", NODE_CMAP), (lg, "Link", LINK_CMAP))):
        ax = axes[0, col]
        draw_map_base(ax, bbox_3857, basemap, basemap_extent)
        plot_network_underlay(ax, under, lw=0.48, alpha=0.48)

        vals = 100.0 * numeric(g, "reach_loss").clip(lower=0)
        positive = vals > 1e-12
        pos_vals = vals[positive]
        gg = g.loc[positive].copy()
        gg["_access_loss"] = pos_vals

        if len(pos_vals):
            # Clip only the visual scale at the upper tail; raw values remain
            # untouched in the CSV/table. The truncated colormap ensures even
            # small but nonzero impacts are visible.
            vmax = max(0.01, float(pos_vals.quantile(0.985)))
            norm = PowerNorm(gamma=0.72, vmin=0.0, vmax=vmax, clip=True)
            cmap = truncated_cmap(base_cmap, 0.28, 1.0)
            ratio = np.clip(pos_vals.to_numpy(dtype=float) / max(vmax, 1e-12), 0, 1)
            if entity == "Node":
                sizes = 8.0 + 34.0 * np.sqrt(ratio)
                gg.plot(
                    ax=ax, column="_access_loss", cmap=cmap, norm=norm,
                    markersize=sizes, edgecolor="#FFFFFF", linewidth=0.15,
                    alpha=0.96, zorder=9,
                )
            else:
                widths = 0.75 + 2.05 * np.sqrt(ratio)
                # GeoPandas cannot pass one linewidth array reliably across
                # all supported versions, so use a few stable width bins.
                rbins = np.linspace(0, 1, 7)
                for bi in range(len(rbins) - 1):
                    lo, hi = rbins[bi], rbins[bi + 1]
                    mask = (ratio >= lo) & ((ratio < hi) if bi < len(rbins) - 2 else (ratio <= hi))
                    sub = gg.iloc[np.where(mask)[0]]
                    if sub.empty:
                        continue
                    mid = 0.5 * (lo + hi)
                    raw_mid = mid * vmax
                    sub.plot(
                        ax=ax,
                        color=cmap(norm(raw_mid)),
                        linewidth=0.75 + 2.05 * math.sqrt(mid),
                        alpha=0.96, zorder=9 + bi,
                    )
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, fraction=0.034, pad=0.020, shrink=0.84)
            cb.set_label("Hospital accessibility loss (%)")
        else:
            ax.text(
                0.03, 0.04, "No nonzero accessibility loss",
                transform=ax.transAxes, fontsize=7.2, color="#555555",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#CCCCCC", alpha=0.88),
            )

        panel_title(ax, "a" if col == 0 else "b", f"{entity} accessibility loss")
        decorate_map(ax, bbox_3857, basemap is not None)

    # ------------------------------------------------------------------
    # P90 hospital-distance change among grid cells that remain accessible.
    # Most isolated removals produce exactly zero change. Plot zero-change
    # components only as the neutral underlay, then use a symmetric log norm
    # and a narrow neutral centre for the nonzero components. The colorbar
    # remains in metres and therefore retains its physical interpretation.
    # ------------------------------------------------------------------
    div_cmap = impact_diverging_cmap()
    for col, (g, entity) in enumerate(((ng, "Node"), (lg, "Link"))):
        ax = axes[1, col]
        draw_map_base(ax, bbox_3857, basemap, basemap_extent)
        plot_network_underlay(ax, under, lw=0.48, alpha=0.48)

        vals = numeric(g, "p90_increase_m")
        nonzero = np.isfinite(vals) & (np.abs(vals) > 1e-9)
        nz_vals = vals[nonzero]
        gg = g.loc[nonzero].copy()
        gg["_p90_change"] = nz_vals

        if len(nz_vals):
            abs_nz = np.abs(nz_vals.to_numpy(dtype=float))
            lim = max(20.0, float(np.quantile(abs_nz, 0.975)))
            linthresh = max(3.0, min(12.0, float(np.quantile(abs_nz, 0.25))))
            norm = SymLogNorm(
                linthresh=linthresh, linscale=0.65,
                vmin=-lim, vmax=lim, base=10, clip=True,
            )
            ratio = np.clip(abs_nz / max(lim, 1e-12), 0, 1)
            if entity == "Node":
                sizes = 9.0 + 31.0 * np.sqrt(ratio)
                gg.plot(
                    ax=ax, column="_p90_change", cmap=div_cmap, norm=norm,
                    markersize=sizes, edgecolor="#FFFFFF", linewidth=0.13,
                    alpha=0.96, zorder=9,
                )
            else:
                rbins = np.linspace(0, 1, 7)
                for bi in range(len(rbins) - 1):
                    lo, hi = rbins[bi], rbins[bi + 1]
                    mask = (ratio >= lo) & ((ratio < hi) if bi < len(rbins) - 2 else (ratio <= hi))
                    sub = gg.iloc[np.where(mask)[0]]
                    if sub.empty:
                        continue
                    representative = sub["_p90_change"].median()
                    sub.plot(
                        ax=ax,
                        color=div_cmap(norm(float(representative))),
                        linewidth=0.72 + 2.10 * math.sqrt(0.5 * (lo + hi)),
                        alpha=0.96, zorder=9 + bi,
                    )
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=div_cmap)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, fraction=0.034, pad=0.020, shrink=0.84)
            cb.set_label("P90 hospital distance change (m)")
            # Keep physical metre labels instead of Matplotlib's default
            # logarithmic-looking 10^x labels for SymLogNorm.
            candidate_ticks = [-lim, -100.0, -10.0, 0.0, 10.0, 100.0, lim]
            ticks = []
            for tick in candidate_ticks:
                if -lim - 1e-9 <= tick <= lim + 1e-9:
                    if not any(abs(tick - old_tick) < 1e-8 for old_tick in ticks):
                        ticks.append(float(tick))
            ticks = sorted(ticks)
            cb.set_ticks(ticks)
            cb.set_ticklabels([f"{t:.0f}" for t in ticks])
        else:
            ax.text(
                0.03, 0.04, "No nonzero P90 distance change",
                transform=ax.transAxes, fontsize=7.2, color="#555555",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#CCCCCC", alpha=0.88),
            )

        panel_title(ax, "c" if col == 0 else "d", f"{entity} P90 distance change")
        decorate_map(ax, bbox_3857, basemap is not None)

    fig.suptitle("Hospital accessibility impact of isolated component removal", y=0.995, fontweight="bold")
    fig.text(
        0.5, 0.018,
        "P90 distance is evaluated among grid cells that remain hospital-accessible after removal; zero-change components are shown by the neutral road-network underlay.",
        ha="center", va="bottom", fontsize=7.35, color="#555555",
    )
    fig.subplots_adjust(top=0.91, bottom=0.075, left=0.035, right=0.985, hspace=0.10, wspace=0.13)
    return save_figure(fig, OUT_DIR / "Fig_07_Hospital_Accessibility_Impact.jpg", dpi)

# =============================================================================
# Figure 09: hospital accessibility under progressive attacks
# =============================================================================

# =============================================================================
# Figure 10: robustness AUC (retention metrics only)
# =============================================================================

def figure_robustness_auc(summary: pd.DataFrame, dpi: int) -> Path:
    """Normalized AUC for the three retention metrics displayed in Fig. 03."""
    metrics = [
        ("od_eff_norm", "GNE retention"),
        ("lscc_norm", "LSCC retention"),
        ("reachable_demand_share", "Reachable OD demand"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.6), sharey=True)
    x = np.arange(len(STRATEGY_ORDER), dtype=float)
    width = 0.34
    displayed = summary[summary["metric"].isin([m[0] for m in metrics])]
    max_auc = float(pd.to_numeric(displayed["auc"], errors="coerce").max()) if not displayed.empty else 1.0
    ymax = min(1.0, max(0.5, math.ceil((max_auc * 1.10) * 10.0) / 10.0))
    for i, (metric, title) in enumerate(metrics):
        ax = axes[i]
        sub = summary[summary["metric"] == metric]
        node = sub[sub["component"] == "node"].set_index("strategy")["auc"]
        link = sub[sub["component"] == "link"].set_index("strategy")["auc"]
        nv = [node.get(s, np.nan) for s in STRATEGY_ORDER]
        lv = [link.get(s, np.nan) for s in STRATEGY_ORDER]
        ax.bar(x - width / 2, nv, width=width, color=NODE_COLOR, alpha=0.88, label="Node")
        ax.bar(x + width / 2, lv, width=width, color=LINK_COLOR, alpha=0.88, label="Link")
        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_STYLE[s]["label"].replace("-targeted", "") for s in STRATEGY_ORDER], rotation=28, ha="right")
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        style_cartesian_ax(ax)
        panel_title(ax, chr(ord("a") + i), title)
        if i == 0:
            ax.set_ylabel("Normalized AUC")
            ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Robustness summary", y=0.995, fontweight="bold")
    fig.text(
        0.5, 0.018,
        "AUC is the mean retained performance over the evaluated removal range; higher values indicate greater robustness.",
        ha="center", va="bottom", fontsize=7.5, color="#555555",
    )
    fig.subplots_adjust(top=0.83, bottom=0.24, left=0.07, right=0.985, wspace=0.12)
    return save_figure(fig, OUT_DIR / "Fig_08_Robustness_Summary.jpg", dpi)

def write_output_index(
    figure_paths: Sequence[Path],
    table_paths: Sequence[Path],
    graph_meta: Dict[str, Any],
    od_meta: Dict[str, Any],
    access_meta: Dict[str, Any],
    basemap_used: bool,
    road_name_mode: str,
) -> Path:
    figure_desc = {
        "Fig_00_Study_Area.jpg": "Road network and dissolved Ho Man Tin focal-area boundary.",
        "Fig_01_Node_Criticality.jpg": "Normalized node criticality by BC, GNE, LSCC and TTD.",
        "Fig_02_Link_Criticality.jpg": "Normalized link criticality by BC, GNE, LSCC and TTD.",
        "Fig_03_Network_Robustness.jpg": "GNE, LSCC and reachable OD-demand retention under random and targeted removal.",
        "Fig_04_Multi_Metric_Criticality.jpg": "Complementary BC-GNE bubble views: LSCC size/TTD color and TTD size/LSCC color for nodes and links.",
        "Fig_05_Network_Accessibility.jpg": "Node harmonic accessibility and its distribution.",
        "Fig_06_Hospital_Accessibility.jpg": "Nearest-hospital network distance, nearest-hospital catchments and distance distribution, with the road network overlaid.",
        "Fig_07_Hospital_Accessibility_Impact.jpg": "Isolated node/link impacts on hospital accessibility and P90 distance.",
        "Fig_08_Robustness_Summary.jpg": "Normalized AUC summary for the three retained-performance metrics in Fig. 03.",
    }
    table_desc = {
        "Tab_01_Network_Overview.csv": "Core network, OD, baseline-performance and hospital-accessibility quantities.",
        "Tab_02_OD_and_Study_Domain.csv": "OD proxy construction and connector summary.",
        "Tab_03_Road_Class_Composition.csv": "Road-link count and length by road class.",
        "Tab_04_Top_Nodes_by_Metric.csv": "Top nodes ranked separately by BC, GNE, LSCC and TTD criticality.",
        "Tab_05_Top_Links_by_Metric.csv": "Top links ranked separately by BC, GNE, LSCC and TTD criticality.",
        "Tab_06_Robustness_Summary.csv": "AUC and thresholds for GNE/LSCC/OD-demand retention plus TTD loss; direction of better performance is stated explicitly.",
        "Tab_07_Hospital_Accessibility.csv": "Baseline hospital-accessibility summary.",
        "Tab_08_Hospital_Catchments.csv": "Nearest-hospital catchment statistics.",
        "Tab_09_Hospital_Critical_Nodes.csv": "Nodes with greatest isolated hospital-accessibility impacts.",
        "Tab_10_Hospital_Critical_Links.csv": "Links with greatest isolated hospital-accessibility impacts.",
    }
    lines = [
        "Ho Man Tin / Central Kowloon Road-Network Resilience – Output Index",
        "=" * 72,
        f"Output script version: {OUTPUT_VERSION}",
        "",
        "TERMINOLOGY",
        "-----------",
        "BC: OD-demand-weighted betweenness centrality.",
        "GNE: demand-weighted global network efficiency.",
        "LSCC: largest strongly connected component.",
        "TTD: demand-weighted total travel distance.",
        "",
        "ROUTING / OD",
        "------------",
        "Shortest-path impedance is road distance (length_m) on the directed network.",
        "Zone OD is distributed across road connector pairs before shortest-path routing.",
        "Each connector OD pair is assigned to a shortest-distance route; equal-length parallel",
        "links share load equally. No congestion equilibrium or capacity model is used.",
        "",
        "CRITICALITY",
        "-----------",
        "Map colors/sizes are normalized within each metric to 0-1 for visualization; mild nonlinear color normalization improves contrast without altering metric values or rankings.",
        "GNE, LSCC and TTD criticality are isolated-removal losses; BC is OD-weighted shortest-path load.",
        "",
        "ROBUSTNESS",
        "----------",
        "Fig. 03 reports GNE retained, LSCC retained and reachable OD demand.",
        "Random removal is averaged over cumulative Monte Carlo sequences; targeted removal",
        "uses static intact-network BC, GNE-loss, LSCC-loss and TTD-loss rankings.",
        "Fig. 08 uses normalized AUC: integral of retained performance over removal fraction,",
        "divided by the evaluated removal-fraction span. Higher AUC means greater robustness.",
        "TTD-loss AUC is not displayed in Fig. 08 because it is a loss metric (lower is better) and is not",
        "bounded on the same 0-1 retention scale; Tab. 06 reports its AUC and 10/25/50% loss thresholds separately.",
        "",
        "HOSPITAL ACCESSIBILITY",
        "----------------------",
        "Hospital accessibility uses shortest road-network distance from each 100 m grid origin; nearest-hospital catchments are defined by the hospital receiving each grid cell under that distance rule.",
        "Hospital accessibility loss is the reduction in the share of grid cells that can reach",
        "at least one hospital. P90 distance is computed among grid cells that remain accessible.",
        "",
        "CARTOGRAPHY",
        "-----------",
        f"Basemap used: {'yes' if basemap_used else 'no (vector-only fallback)' }.",
        f"Road-label source: {road_name_mode}.",
        "Node criticality uses blue; link criticality uses red.",
        "",
        "FIGURES",
        "-------",
    ]
    for p in figure_paths:
        lines.append(f"{p.name}: {figure_desc.get(p.name, '')}")
    lines += ["", "TABLES", "------"]
    for p in table_paths:
        lines.append(f"{p.name}: {table_desc.get(p.name, '')}")
    lines += ["", "All CSV tables are UTF-8 with BOM for reliable Excel display."]
    path = OUT_DIR / "Output_Index.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"figure DPI (default {DEFAULT_DPI})")
    p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help=f"top components in report tables (default {DEFAULT_TOP_N})")
    p.add_argument("--label-top", type=int, default=DEFAULT_LABEL_TOP, help=f"top labels per criticality map panel (default {DEFAULT_LABEL_TOP})")
    p.add_argument("--zoom", type=int, default=DEFAULT_ZOOM, help=f"OSM tile zoom (default {DEFAULT_ZOOM})")
    p.add_argument("--no-basemap", action="store_true", help="do not request/use OSM raster tiles")
    p.add_argument("--refresh-basemap", action="store_true", help="refresh cached OSM tiles")
    p.add_argument("--official-road-file", type=str, default=None, help="optional local LandsD Road Centreline file for English link labels")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    require_files(
        [
            ROOT_NODE_CSV, ROOT_LINK_CSV,
            OD_META_PATH, OD_ZONE_PATH, OD_CONNECTOR_PATH,
            NODE_METRIC_PATH, LINK_METRIC_PATH, GRAPH_META_PATH,
            REMOVAL_CURVE_PATH, REMOVAL_SUMMARY_PATH, REMOVAL_META_PATH,
            NODE_ACCESS_PATH, HOSPITAL_PATH, HOSPITAL_ACCESS_PATH,
            HOSPITAL_SUMMARY_PATH, HOSPITAL_CATCHMENT_PATH,
            ACCESS_IMPACT_PATH, ACCESS_META_PATH, GRID_SHP,
        ]
    )

    print("=" * 72)
    print("Ho Man Tin / central Kowloon report-output generation")
    print("=" * 72)

    # Numerical outputs.
    node_metric = read_csv_auto(NODE_METRIC_PATH)
    link_metric = read_csv_auto(LINK_METRIC_PATH)
    removal_curve = read_csv_auto(REMOVAL_CURVE_PATH)
    removal_summary = read_csv_auto(REMOVAL_SUMMARY_PATH)
    node_access = read_csv_auto(NODE_ACCESS_PATH)
    hospital = read_csv_auto(HOSPITAL_PATH)
    hospital_access = read_csv_auto(HOSPITAL_ACCESS_PATH)
    hospital_summary = read_csv_auto(HOSPITAL_SUMMARY_PATH)
    hospital_catchment = read_csv_auto(HOSPITAL_CATCHMENT_PATH)
    access_impact = read_csv_auto(ACCESS_IMPACT_PATH)
    od_zone = read_csv_auto(OD_ZONE_PATH)
    od_connector = read_csv_auto(OD_CONNECTOR_PATH)

    od_meta = read_json(OD_META_PATH)
    graph_meta = read_json(GRAPH_META_PATH)
    removal_meta = read_json(REMOVAL_META_PATH)
    access_meta = read_json(ACCESS_META_PATH)

    # Validate the revised metric schema explicitly so an old result directory
    # cannot silently produce wrong figures.
    for name, df, id_col in (("node_metric.csv", node_metric, "node_id"), ("link_metric.csv", link_metric, "link_id")):
        required = {id_col, *METRICS.keys(), "reachable_demand_share"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} is not from the revised OD-aware workflow; missing {sorted(missing)}")

    # Spatial data and display road names.
    nodes = load_base_nodes()
    links = load_base_links()
    grid_geom = load_grid_geometry()
    hospital_gdf = load_hospital_geometry(hospital)
    hmt_boundary = load_hmt_boundary()

    link_metric = optionally_apply_official_road_names(links, link_metric, args.official_road_file)
    if (link_metric.get("name_source", pd.Series(dtype=str)) == "LandsD Road Centreline").any():
        road_name_mode = "Lands Department Road Centreline (matched locally), with OSM/link-ID fallback"
    else:
        road_name_mode = "English component of existing OSM road names, with link-ID fallback"

    # Dynamic full-network plotting extent. No old Ho Man Tin-only hard-coded bbox.
    bbox_wgs = bbox_from_links(links)
    bbox_3857 = bbox_to_3857(bbox_wgs)

    basemap = basemap_extent = None
    if not args.no_basemap:
        print("Preparing optional pale OSM basemap...")
        try:
            basemap, basemap_extent = build_basemap(bbox_wgs, int(args.zoom), bool(args.refresh_basemap))
        except Exception as exc:
            warnings.warn(f"Basemap unavailable ({exc}); continuing with vector-only background.")
            basemap = basemap_extent = None

    print("Writing report tables...")
    table_paths = write_tables(
        node_metric, link_metric, od_meta, graph_meta, removal_summary,
        access_meta, hospital_summary, hospital_catchment, access_impact,
        None, od_zone, od_connector, max(1, int(args.top_n)),
    )

    dpi = max(120, int(args.dpi))
    label_top = max(0, int(args.label_top))
    figure_paths: List[Path] = []

    jobs = [
        ("Fig. 00 study area", lambda: figure_study_domain(links, hmt_boundary, bbox_3857, basemap, basemap_extent, dpi)),
        ("Fig. 01 node criticality", lambda: figure_node_criticality(node_metric, nodes, links, bbox_3857, basemap, basemap_extent, label_top, dpi)),
        ("Fig. 02 link criticality", lambda: figure_link_criticality(link_metric, links, bbox_3857, basemap, basemap_extent, label_top, dpi)),
        ("Fig. 03 network robustness", lambda: figure_network_robustness(removal_curve, dpi)),
        ("Fig. 04 multi-metric criticality", lambda: figure_criticality_bubble(node_metric, link_metric, dpi)),
        ("Fig. 05 network accessibility", lambda: figure_structural_accessibility(node_access, nodes, links, bbox_3857, basemap, basemap_extent, dpi)),
        ("Fig. 06 hospital accessibility", lambda: figure_hospital_accessibility(
            hospital_access, grid_geom, hospital_gdf, links, bbox_3857,
            basemap, basemap_extent, dpi, zoom=int(args.zoom),
            refresh_basemap=bool(args.refresh_basemap),
        )),
        ("Fig. 07 hospital accessibility impact", lambda: figure_hospital_access_impact(access_impact, nodes, links, bbox_3857, basemap, basemap_extent, dpi)),
        ("Fig. 08 robustness summary", lambda: figure_robustness_auc(removal_summary, dpi)),
    ]

    for label, fn in jobs:
        print(f"Generating {label}...")
        figure_paths.append(fn())

    index_path = write_output_index(
        figure_paths, table_paths, graph_meta, od_meta, access_meta,
        basemap_used=basemap is not None,
        road_name_mode=road_name_mode,
    )

    elapsed = time.perf_counter() - started
    print()
    print("Completed report outputs.")
    print(f"  Figures : {len(figure_paths)}")
    print(f"  Tables  : {len(table_paths)}")
    print(f"  Output  : {OUT_DIR}")
    print(f"  Index   : {index_path}")
    print(f"  Runtime : {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
