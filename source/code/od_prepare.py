# -*- coding: utf-8 -*-
# @Time   : 2026/9/5
# @Author : Junze Yang / revised with ChatGPT
# @File   : od_prepare.py

"""Prepare road-oriented TCS 2022 TPUSB OD demand for the study road network.

Run order
---------
    python code/net.py                 # only if node.csv/link.csv must be rebuilt
    python code/od_prepare.py
    python code/graph_metric.py
    python code/removal.py
    python code/accessibility.py

The input OD directory must be next to ``code``::

    PROJECT_ROOT/
        node.csv
        link.csv
        HK&HMT OD flow from HKTCS022-myy-260905/
            all_tpusb_od_flow.csv
            NOTE.md
        code/
            od_prepare.py

Method
------
* All routing costs downstream are road-network distance (``length_m``).
* TCS ``main_mode`` is an OD-level union of modes, not a modal grouping key.
  Therefore exact road-only OD flow cannot be recovered from the supplied
  aggregate table. We construct a transparent *road-mode proxy*:

      road_proxy_flow = expanded_hk_flow * road_factor

  Pure road-mode rows receive factor 1, pure non-road rows factor 0. For mixed
  rows, ``road_factor`` is the share of the listed modes' territory-wide
  WT_TRIP totals attributable to road modes. The territory-wide mode totals are
  parsed from NOTE.md (with documented fallback constants).
* Road modes: 5 minibus, 6 franchised bus, 7 private car/motorcycle,
  8 taxi, 9 special-purpose bus.
* The computational OD domain follows the full current road-network envelope,
  not the smaller Ho Man Tin focal area.
* Each included 2021 TPUSB is represented by up to three spatially spread
  actual road nodes. Zone demand is split equally across connector pairs.
  Intrazonal OD is retained by using distinct connector-node pairs where
  possible; zero-length same-node connector pairs are excluded and the
  remaining connector-pair weights are renormalized.

2021 TPUSB boundaries
---------------------
The script first uses ``--tpusb-file`` if supplied, then a local cache. If no
local boundary file exists, it attempts to download the official Planning
Department CSDI 2021 TPU/Subunit layer (dataset
``pland_rcd_1634022783366_65050``). If automatic download is blocked, download
that official layer manually and rerun with ``--tpusb-file PATH``.

Outputs
-------
    res/od/zone.csv
    res/od/zone_connector.csv
    res/od/od_zone.csv
    res/od/od_connector.csv
    res/od/od_meta.json
    res/od/study_tpusb.geojson
    res/od/cache/tpusb_2021.geojson
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

try:
    import geopandas as gpd
    from shapely import wkt
    from shapely.geometry import Point, box
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "od_prepare.py requires geopandas and shapely. Install geopandas first."
    ) from exc

from scipy.spatial import cKDTree


# =============================================================================
# Paths and settings
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NODE_PATH = PROJECT_ROOT / "node.csv"
LINK_PATH = PROJECT_ROOT / "link.csv"

OD_DIR = PROJECT_ROOT / "HK&HMT OD flow from HKTCS022-myy-260905"
ALL_OD_PATH = OD_DIR / "all_tpusb_od_flow.csv"
NOTE_PATH = OD_DIR / "NOTE.md"

RESULT_DIR = PROJECT_ROOT / "res" / "od"
CACHE_DIR = RESULT_DIR / "cache"
TPUSB_CACHE_PATH = CACHE_DIR / "tpusb_2021.geojson"

ZONE_OUT = RESULT_DIR / "zone.csv"
CONNECTOR_OUT = RESULT_DIR / "zone_connector.csv"
OD_ZONE_OUT = RESULT_DIR / "od_zone.csv"
OD_CONNECTOR_OUT = RESULT_DIR / "od_connector.csv"
STUDY_TPUSB_OUT = RESULT_DIR / "study_tpusb.geojson"
META_OUT = RESULT_DIR / "od_meta.json"

WGS84 = "EPSG:4326"
HK80 = "EPSG:2326"

ROAD_MODES = {"5", "6", "7", "8", "9"}
DEFAULT_CONNECTORS = 3
DEFAULT_DOMAIN_BUFFER_M = 150.0
DEFAULT_NEAREST_FALLBACK_K = 12
MASTER_SEED = 20260905
EPS = 1e-12

# Planning Department: Boundaries of Tertiary Planning Units & Subunits
# (for 2021 Population Census), official CSDI dataset.
CSDI_DATASET_ID = "pland_rcd_1634022783366_65050"
TPUSB_DOWNLOAD_URLS = [
    (
        "https://portal.csdi.gov.hk/csdi-webpage/file-api?"
        "dataset_id=%s&format=geojson&layer_name=tpusu_2021" % CSDI_DATASET_ID
    ),
    (
        "https://portal.csdi.gov.hk/csdi-webpage/file-api?"
        "dataset_id=%s&format=geojson&layer_name=TPUSU_2021" % CSDI_DATASET_ID
    ),
    (
        "https://portal.csdi.gov.hk/server/rest/services/common/%s/"
        "FeatureServer/0/query?where=1%%3D1&outFields=*&returnGeometry=true&"
        "outSR=4326&f=geojson" % CSDI_DATASET_ID
    ),
]

# Fallback values copied from the supplied NOTE.md, Section 6.
FALLBACK_MODE_TOTALS = {
    "1": 4369936.8003,
    "2": 291500.0881,
    "3": 134671.5275,
    "4": 57807.5802,
    "5": 1133975.5668,
    "6": 3091760.5132,
    "7": 1905902.8078,
    "8": 801600.3737,
    "9": 575991.2980,
    "10": 71298.7024,
    "99": 3755160.7893,
    "999": 6836.0513,
}


# =============================================================================
# Generic helpers
# =============================================================================

def read_csv_auto(path: Path, **kwargs) -> Tuple[pd.DataFrame, str]:
    errors: List[str] = []
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs), enc
        except UnicodeDecodeError as exc:
            errors.append("%s: %s" % (enc, exc))
    raise UnicodeError("Cannot decode %s. %s" % (path, errors))


def require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError("%s is missing required columns: %s" % (name, missing))


def normalize_code(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return re.sub(r"\D", "", text)


def normalize_mode_list(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [x for x in re.split(r"\s*\|\s*", str(value).strip()) if x]


def parse_mode_totals(note_path: Path) -> Tuple[Dict[str, float], str]:
    if not note_path.exists():
        return dict(FALLBACK_MODE_TOTALS), "fallback constants (NOTE.md missing)"

    text = note_path.read_text(encoding="utf-8-sig", errors="replace")
    totals: Dict[str, float] = {}
    # Markdown rows: | 5 | 小巴 | 7,310 | 1,133,975.5668 |
    pattern = re.compile(
        r"^\|\s*(\d+)\s*\|[^\n]*?\|\s*[\d,]+\s*\|\s*([\d,]+(?:\.\d+)?)\s*\|\s*$",
        flags=re.MULTILINE,
    )
    for mode, total in pattern.findall(text):
        try:
            totals[str(mode)] = float(total.replace(",", ""))
        except ValueError:
            continue

    if len(totals) >= 8:
        merged = dict(FALLBACK_MODE_TOTALS)
        merged.update(totals)
        return merged, str(note_path)
    return dict(FALLBACK_MODE_TOTALS), "fallback constants (NOTE.md table not parsed)"


def road_factor(main_mode: object, mode_totals: Dict[str, float]) -> float:
    modes = normalize_mode_list(main_mode)
    if not modes:
        return 0.0
    denominator = sum(float(mode_totals.get(m, 0.0)) for m in modes)
    if denominator <= 0:
        return 1.0 if set(modes).issubset(ROAD_MODES) else 0.0
    numerator = sum(float(mode_totals.get(m, 0.0)) for m in modes if m in ROAD_MODES)
    return float(min(1.0, max(0.0, numerator / denominator)))


# =============================================================================
# Network geometry
# =============================================================================

def load_network_spatial() -> Tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nodes, _ = read_csv_auto(NODE_PATH)
    links, _ = read_csv_auto(LINK_PATH)
    require_columns(nodes, ["node_id", "lon", "lat"], "node.csv")
    require_columns(links, ["link_id", "from_node", "to_node", "length_m"], "link.csv")

    nodes = nodes.copy()
    links = links.copy()
    nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="raise").astype(np.int64)
    links["link_id"] = pd.to_numeric(links["link_id"], errors="raise").astype(np.int64)

    ng = gpd.GeoDataFrame(
        nodes,
        geometry=gpd.points_from_xy(nodes["lon"], nodes["lat"]),
        crs=WGS84,
    )

    if "geometry" in links.columns:
        geom = links["geometry"].apply(
            lambda x: wkt.loads(x) if isinstance(x, str) and x.strip() else None
        )
        lg = gpd.GeoDataFrame(links, geometry=geom, crs=WGS84)
    else:
        node_xy = nodes.set_index("node_id")[["lon", "lat"]]
        from shapely.geometry import LineString
        geoms = []
        for row in links.itertuples(index=False):
            a = node_xy.loc[int(row.from_node)]
            b = node_xy.loc[int(row.to_node)]
            geoms.append(LineString([(a.lon, a.lat), (b.lon, b.lat)]))
        lg = gpd.GeoDataFrame(links, geometry=geoms, crs=WGS84)

    return nodes, links, ng, lg


# =============================================================================
# Official TPUSB boundary loading and code detection
# =============================================================================

def _valid_geojson_payload(content: bytes) -> bool:
    lead = content.lstrip()[:1]
    if lead not in (b"{", b"["):
        return False
    try:
        payload = json.loads(content.decode("utf-8-sig"))
        return isinstance(payload, dict) and (
            payload.get("type") in {"FeatureCollection", "Feature"}
            or "features" in payload
        )
    except Exception:
        return False


def obtain_tpusb_file(user_file: Optional[str], refresh: bool) -> Tuple[Path, str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if user_file:
        path = Path(user_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError("--tpusb-file does not exist: %s" % path)
        return path, "user supplied"

    if TPUSB_CACHE_PATH.exists() and not refresh:
        return TPUSB_CACHE_PATH, "cache"

    last_errors: List[str] = []
    for url in TPUSB_DOWNLOAD_URLS:
        try:
            response = requests.get(
                url,
                timeout=90,
                headers={
                    "User-Agent": "HoManTin-Network-Resilience/1.0 academic analysis"
                },
            )
            response.raise_for_status()
            if not _valid_geojson_payload(response.content):
                raise RuntimeError(
                    "response is not a GeoJSON FeatureCollection (content-type=%s)"
                    % response.headers.get("content-type")
                )
            TPUSB_CACHE_PATH.write_bytes(response.content)
            return TPUSB_CACHE_PATH, url
        except Exception as exc:
            last_errors.append("%s -> %s" % (url, exc))

    if TPUSB_CACHE_PATH.exists():
        warnings.warn(
            "TPUSB refresh failed; using existing cache. Errors: %s" % last_errors
        )
        return TPUSB_CACHE_PATH, "stale cache"

    raise RuntimeError(
        "Could not obtain the official 2021 TPUSB boundary automatically. "
        "Download Planning Department dataset %s (layer tpusu_2021) and run "
        "with --tpusb-file PATH. Attempts: %s"
        % (CSDI_DATASET_ID, last_errors)
    )


def detect_tpusb_codes(
    polygons: gpd.GeoDataFrame,
    od_codes: Sequence[str],
) -> Tuple[pd.Series, str]:
    """Detect/construct six-digit TPUSB codes by maximizing overlap with OD codes."""
    od_set = set(str(x).zfill(6) for x in od_codes if str(x))
    attrs = [c for c in polygons.columns if c != polygons.geometry.name]

    best_score = -1
    best_series: Optional[pd.Series] = None
    best_desc = ""

    # 1) Direct six-digit column.
    for col in attrs:
        s = polygons[col].map(normalize_code)
        s6 = s.where(s.str.len() == 6, "")
        score = int(s6.isin(od_set).sum())
        if score > best_score:
            best_score = score
            best_series = s6
            best_desc = str(col)

    # 2) Typical 3-digit TPU + 3-digit Subunit combination.
    # Limit to columns that actually contain mostly short numeric-like codes.
    candidate_cols: List[str] = []
    for col in attrs:
        s = polygons[col].map(normalize_code)
        nonempty = s[s.ne("")]
        if len(nonempty) == 0:
            continue
        short_share = float(nonempty.str.len().between(1, 3).mean())
        if short_share >= 0.7:
            candidate_cols.append(col)

    for i, a in enumerate(candidate_cols):
        sa = polygons[a].map(normalize_code)
        for b in candidate_cols:
            if a == b:
                continue
            sb = polygons[b].map(normalize_code)
            combo = sa.map(lambda x: x.zfill(3) if x else "") + sb.map(
                lambda x: x.zfill(3) if x else ""
            )
            combo = combo.where(combo.str.len() == 6, "")
            score = int(combo.isin(od_set).sum())
            if score > best_score:
                best_score = score
                best_series = combo
                best_desc = "%s + %s (3+3)" % (a, b)

    if best_series is None or best_score <= 0:
        raise ValueError(
            "Could not identify a TPUSB code field from the official boundary. "
            "Columns: %s" % attrs
        )

    # Strong warning instead of silent misuse if overlap is suspiciously small.
    unique_matches = len(set(best_series[best_series.isin(od_set)]))
    if unique_matches < 20:
        warnings.warn(
            "Only %d unique TPUSB codes overlap the OD table using '%s'. "
            "Inspect the official boundary schema if this is unexpected."
            % (unique_matches, best_desc)
        )

    return best_series.astype(str), best_desc


# =============================================================================
# Study zones and connector nodes
# =============================================================================

def farthest_spread_nodes(
    candidate_indices: np.ndarray,
    node_xy: np.ndarray,
    reference_xy: np.ndarray,
    k: int,
) -> List[int]:
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    if len(candidate_indices) == 0:
        return []
    k = min(int(k), len(candidate_indices))
    pts = node_xy[candidate_indices]

    # First: road node closest to polygon representative point.
    first_local = int(np.argmin(np.linalg.norm(pts - reference_xy[None, :], axis=1)))
    chosen_local = [first_local]

    while len(chosen_local) < k:
        chosen_pts = pts[np.asarray(chosen_local)]
        # Maximize distance to the nearest already selected node.
        dist = np.linalg.norm(pts[:, None, :] - chosen_pts[None, :, :], axis=2)
        min_dist = dist.min(axis=1)
        min_dist[np.asarray(chosen_local)] = -1.0
        next_local = int(np.argmax(min_dist))
        if next_local in chosen_local:
            break
        chosen_local.append(next_local)

    return [int(candidate_indices[i]) for i in chosen_local]


def build_study_zones_and_connectors(
    polygons: gpd.GeoDataFrame,
    nodes_gdf: gpd.GeoDataFrame,
    links_gdf: gpd.GeoDataFrame,
    connectors_per_zone: int,
    domain_buffer_m: float,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, Dict[str, object]]:
    poly = polygons.to_crs(HK80).copy()
    nodes_hk = nodes_gdf.to_crs(HK80).copy().reset_index(drop=True)
    links_hk = links_gdf.to_crs(HK80).copy()

    # Full current road-network envelope, buffered slightly to avoid clipping
    # TPUSBs that touch the network at its boundary.
    minx, miny, maxx, maxy = links_hk.total_bounds
    domain_geom = box(minx, miny, maxx, maxy).buffer(float(domain_buffer_m))
    study = poly[poly.geometry.intersects(domain_geom)].copy()
    study = study[study["tpusb"].astype(str).str.len().eq(6)].copy()
    # Dissolve rather than drop duplicates so multipart TPUSB features are not
    # silently truncated if the official layer stores one code in several parts.
    study = study[["tpusb", "geometry"]].dissolve(by="tpusb", as_index=False).reset_index(drop=True)
    if study.empty:
        raise RuntimeError("No TPUSB polygon intersects the full road-network domain.")

    node_xy = np.column_stack(
        [nodes_hk.geometry.x.to_numpy(dtype=float), nodes_hk.geometry.y.to_numpy(dtype=float)]
    )
    tree = cKDTree(node_xy)
    sindex = nodes_hk.sindex

    connector_rows: List[Dict[str, object]] = []
    fallback_count = 0

    for zone in study.itertuples(index=False):
        code = str(zone.tpusb)
        geom = zone.geometry
        rep = geom.representative_point()
        ref_xy = np.asarray([rep.x, rep.y], dtype=float)

        try:
            cand = np.asarray(
                list(sindex.query(geom.buffer(1.0), predicate="intersects")),
                dtype=int,
            )
        except TypeError:  # older geopandas spatial-index API
            cand = np.asarray(list(sindex.intersection(geom.bounds)), dtype=int)
            cand = np.asarray(
                [i for i in cand if geom.buffer(1.0).covers(nodes_hk.geometry.iloc[int(i)])],
                dtype=int,
            )

        if len(cand) > 0:
            selected = farthest_spread_nodes(
                cand, node_xy, ref_xy, int(connectors_per_zone)
            )
            source = "inside"
        else:
            # Boundary polygons occasionally contain no retained simplified road
            # node. Use nearest actual network nodes as a controlled fallback.
            fallback_count += 1
            kk = min(
                max(int(connectors_per_zone), DEFAULT_NEAREST_FALLBACK_K),
                len(nodes_hk),
            )
            _, nearest = tree.query(ref_xy, k=kk)
            nearest = np.atleast_1d(nearest).astype(int)
            selected = farthest_spread_nodes(
                nearest, node_xy, ref_xy, int(connectors_per_zone)
            )
            source = "nearest"

        if not selected:
            continue

        k = len(selected)
        for order, idx in enumerate(selected, start=1):
            nrow = nodes_hk.iloc[int(idx)]
            dist = float(np.linalg.norm(node_xy[int(idx)] - ref_xy))
            connector_rows.append(
                {
                    "tpusb": code,
                    "connector_order": int(order),
                    "node_id": int(nrow["node_id"]),
                    "weight": float(1.0 / k),
                    "selection": source,
                    "rep_snap_dist_m": dist,
                    "node_x_hk80": float(node_xy[int(idx), 0]),
                    "node_y_hk80": float(node_xy[int(idx), 1]),
                }
            )

    connectors = pd.DataFrame(connector_rows)
    if connectors.empty:
        raise RuntimeError("No TPUSB-to-road connectors were generated.")

    connected_codes = set(connectors["tpusb"].astype(str))
    study = study[study["tpusb"].astype(str).isin(connected_codes)].copy()

    meta = {
        "network_bbox_hk80": [float(minx), float(miny), float(maxx), float(maxy)],
        "domain_buffer_m": float(domain_buffer_m),
        "study_zone_count": int(len(study)),
        "connector_count": int(len(connectors)),
        "zones_using_nearest_fallback": int(fallback_count),
        "connectors_per_zone_target": int(connectors_per_zone),
    }
    return study, connectors, meta


# =============================================================================
# OD filtering / expansion
# =============================================================================

def build_zone_od(
    all_od: pd.DataFrame,
    study_codes: Sequence[str],
    mode_totals: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    od = all_od.copy()
    require_columns(
        od,
        ["origin_tpusb", "destination_tpusb", "main_mode", "expanded_hk_flow"],
        "all_tpusb_od_flow.csv",
    )
    od["origin_tpusb"] = od["origin_tpusb"].map(normalize_code).str.zfill(6)
    od["destination_tpusb"] = od["destination_tpusb"].map(normalize_code).str.zfill(6)
    od["expanded_hk_flow"] = pd.to_numeric(od["expanded_hk_flow"], errors="coerce")
    od = od[np.isfinite(od["expanded_hk_flow"]) & (od["expanded_hk_flow"] > 0)].copy()

    codes = set(str(x) for x in study_codes)
    inside = od["origin_tpusb"].isin(codes) & od["destination_tpusb"].isin(codes)
    domain_od = od.loc[inside].copy()

    domain_od["road_factor"] = domain_od["main_mode"].apply(
        lambda x: road_factor(x, mode_totals)
    )
    domain_od["road_proxy_flow"] = (
        domain_od["expanded_hk_flow"] * domain_od["road_factor"]
    )
    domain_od = domain_od[domain_od["road_proxy_flow"] > EPS].copy()

    keep_preferred = [
        "origin_tpusb",
        "destination_tpusb",
        "main_mode",
        "sample_flow",
        "expanded_hk_flow",
        "road_factor",
        "road_proxy_flow",
        "kish_neff",
        "origin_26PDD",
        "origin_26PDD_name",
        "destination_26PDD",
        "destination_26PDD_name",
    ]
    keep = [c for c in keep_preferred if c in domain_od.columns]
    domain_od = domain_od[keep].sort_values(
        ["origin_tpusb", "destination_tpusb"]
    ).reset_index(drop=True)
    domain_od.insert(0, "zone_od_id", np.arange(len(domain_od), dtype=np.int64))

    stats = {
        "domain_allmode_flow": float(
            od.loc[inside, "expanded_hk_flow"].to_numpy(dtype=np.float64).sum(dtype=np.float64)
        ),
        "domain_road_proxy_flow": float(domain_od["road_proxy_flow"].to_numpy(dtype=np.float64).sum(dtype=np.float64)),
        "domain_zone_od_count": int(len(domain_od)),
        "domain_intrazonal_count": int(
            (domain_od["origin_tpusb"] == domain_od["destination_tpusb"]).sum()
        ),
    }
    return domain_od, stats


def expand_connector_od(zone_od: pd.DataFrame, connectors: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    by_zone: Dict[str, pd.DataFrame] = {
        str(code): grp.sort_values("connector_order").reset_index(drop=True)
        for code, grp in connectors.groupby("tpusb", sort=False)
    }

    rows: List[Dict[str, object]] = []
    dropped_single_intrazonal_flow = 0.0

    for row in zone_od.itertuples(index=False):
        oz = str(row.origin_tpusb)
        dz = str(row.destination_tpusb)
        flow = float(row.road_proxy_flow)
        oc = by_zone.get(oz)
        dc = by_zone.get(dz)
        if oc is None or dc is None or len(oc) == 0 or len(dc) == 0:
            continue

        pairs: List[Tuple[int, int, float]] = []
        for o in oc.itertuples(index=False):
            for d in dc.itertuples(index=False):
                on = int(o.node_id)
                dn = int(d.node_id)
                # Zero-length same-road-node pair is not a road trip. For
                # intrazonal OD, retain other distinct connector combinations.
                if on == dn:
                    continue
                pairs.append((on, dn, float(o.weight) * float(d.weight)))

        if not pairs:
            dropped_single_intrazonal_flow += flow
            continue

        weight_sum = sum(p[2] for p in pairs)
        if weight_sum <= 0:
            continue

        for on, dn, pair_weight in pairs:
            q = flow * pair_weight / weight_sum
            rows.append(
                {
                    "zone_od_id": int(row.zone_od_id),
                    "origin_tpusb": oz,
                    "destination_tpusb": dz,
                    "origin_node": int(on),
                    "destination_node": int(dn),
                    "pair_weight": float(pair_weight / weight_sum),
                    "demand": float(q),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No connector-level road OD demand was generated.")

    # Multiple zone OD records should not duplicate the same zone pair, but
    # connector pairs can coincide across adjacent small zones. Keep zone IDs
    # for traceability; graph_metric.py may aggregate by road-node pair.
    out.insert(0, "connector_od_id", np.arange(len(out), dtype=np.int64))

    input_flow = float(zone_od["road_proxy_flow"].to_numpy(dtype=np.float64).sum(dtype=np.float64))
    output_flow = float(out["demand"].to_numpy(dtype=np.float64).sum(dtype=np.float64))
    stats = {
        "connector_od_count": int(len(out)),
        "zone_road_proxy_flow": input_flow,
        "connector_demand_total": output_flow,
        "dropped_single_connector_intrazonal_flow": float(dropped_single_intrazonal_flow),
        "flow_conservation_relative_error": (
            abs(output_flow + dropped_single_intrazonal_flow - input_flow) / input_flow
            if input_flow > 0
            else 0.0
        ),
    }
    return out, stats


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tpusb-file",
        type=str,
        default=None,
        help="local official 2021 TPUSB boundary file (GeoJSON/SHP/GPKG); optional",
    )
    parser.add_argument(
        "--refresh-tpusb",
        action="store_true",
        help="redownload official TPUSB boundary even if cache exists",
    )
    parser.add_argument(
        "--connectors",
        type=int,
        default=DEFAULT_CONNECTORS,
        help="representative road nodes per TPUSB (default: 3)",
    )
    parser.add_argument(
        "--domain-buffer",
        type=float,
        default=DEFAULT_DOMAIN_BUFFER_M,
        help="buffer around full road-network bbox for selecting TPUSBs, metres",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    required = [NODE_PATH, LINK_PATH, ALL_OD_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: %s" % missing)

    print("=" * 78)
    print("Road-oriented TPUSB OD preparation")
    print("=" * 78)

    nodes, links, nodes_gdf, links_gdf = load_network_spatial()
    all_od, od_enc = read_csv_auto(
        ALL_OD_PATH,
        dtype={"origin_tpusb": str, "destination_tpusb": str, "main_mode": str},
    )
    print("Loaded network: %d nodes, %d directed links" % (len(nodes), len(links)))
    print("Loaded OD table: %d zone OD records (%s)" % (len(all_od), od_enc))

    mode_totals, mode_source = parse_mode_totals(NOTE_PATH)
    print("Road-mode proxy weights: %s" % mode_source)

    tpusb_path, tpusb_source = obtain_tpusb_file(
        args.tpusb_file, bool(args.refresh_tpusb)
    )
    polygons = gpd.read_file(tpusb_path)
    if polygons.empty:
        raise RuntimeError("TPUSB boundary file is empty: %s" % tpusb_path)
    if polygons.crs is None:
        warnings.warn("TPUSB boundary CRS is missing; assuming EPSG:4326.")
        polygons = polygons.set_crs(WGS84)
    else:
        polygons = polygons.to_crs(WGS84)

    od_codes = pd.concat(
        [all_od["origin_tpusb"], all_od["destination_tpusb"]], ignore_index=True
    ).map(normalize_code)
    code_series, code_field = detect_tpusb_codes(polygons, od_codes)
    polygons = polygons.copy()
    polygons["tpusb"] = code_series
    polygons = polygons[polygons["tpusb"].str.len().eq(6)].copy()
    print("TPUSB boundary source: %s" % tpusb_source)
    print("Detected TPUSB code: %s" % code_field)

    study, connectors, domain_meta = build_study_zones_and_connectors(
        polygons,
        nodes_gdf,
        links_gdf,
        connectors_per_zone=max(1, int(args.connectors)),
        domain_buffer_m=max(0.0, float(args.domain_buffer)),
    )
    print(
        "Study domain: %d TPUSBs, %d road connectors"
        % (len(study), len(connectors))
    )

    zone_od, od_stats = build_zone_od(
        all_od, study["tpusb"].astype(str).tolist(), mode_totals
    )
    connector_od, connector_stats = expand_connector_od(zone_od, connectors)

    # Zone table in WGS84 with representative coordinates for later plots.
    study_wgs = study.to_crs(WGS84).copy()
    reps = study_wgs.geometry.representative_point()
    zone_table = pd.DataFrame(
        {
            "tpusb": study_wgs["tpusb"].astype(str).to_numpy(),
            "rep_lon": reps.x.to_numpy(dtype=float),
            "rep_lat": reps.y.to_numpy(dtype=float),
            "area_m2": study.geometry.area.to_numpy(dtype=float),
        }
    ).sort_values("tpusb").reset_index(drop=True)

    zone_table.to_csv(ZONE_OUT, index=False, encoding="utf-8-sig")
    connectors.sort_values(["tpusb", "connector_order"]).to_csv(
        CONNECTOR_OUT, index=False, encoding="utf-8-sig"
    )
    zone_od.to_csv(OD_ZONE_OUT, index=False, encoding="utf-8-sig")
    connector_od.to_csv(OD_CONNECTOR_OUT, index=False, encoding="utf-8-sig")

    # GeoJSON keeps the complete geometry/schema for the future output.py.
    study_wgs[["tpusb", "geometry"]].to_file(
        STUDY_TPUSB_OUT, driver="GeoJSON"
    )

    meta = {
        "input": {
            "od_directory": str(OD_DIR),
            "all_od_csv": str(ALL_OD_PATH),
            "od_encoding": od_enc,
            "note": str(NOTE_PATH),
            "tpusb_boundary": str(tpusb_path),
            "tpusb_source": tpusb_source,
            "tpusb_code_detection": code_field,
        },
        "method": {
            "routing_cost_downstream": "length_m",
            "road_modes": sorted(ROAD_MODES),
            "road_proxy": (
                "expanded_hk_flow multiplied by road-mode share among the modes "
                "listed in main_mode, using territory-wide WT_TRIP mode totals "
                "from NOTE.md; exact road-only OD cannot be recovered from the "
                "supplied aggregated main_mode field"
            ),
            "connector_method": (
                "up to K spatially spread actual road nodes per TPUSB; equal zone "
                "weights; connector-pair demand renormalized after zero-length "
                "same-node pairs are removed"
            ),
            "seed": MASTER_SEED,
        },
        "network": {
            "nodes": int(len(nodes)),
            "directed_links": int(len(links)),
        },
        "domain": domain_meta,
        "od": od_stats,
        "connector_od": connector_stats,
        "mode_totals": {k: float(v) for k, v in mode_totals.items()},
        "outputs": {
            "zone": str(ZONE_OUT),
            "zone_connector": str(CONNECTOR_OUT),
            "od_zone": str(OD_ZONE_OUT),
            "od_connector": str(OD_CONNECTOR_OUT),
            "study_tpusb_geojson": str(STUDY_TPUSB_OUT),
        },
    }
    META_OUT.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    relerr = connector_stats["flow_conservation_relative_error"]
    if relerr > 1e-8:
        warnings.warn("Connector OD flow conservation relative error: %.3e" % relerr)

    print("Road-proxy zone OD pairs: %d" % len(zone_od))
    print("Connector OD records: %d" % len(connector_od))
    print(f"Road-proxy demand: {connector_stats['connector_demand_total']:,.2f} trips/reference weekday")
    print("Saved: %s" % RESULT_DIR)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
