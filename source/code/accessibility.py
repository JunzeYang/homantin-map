# -*- coding: utf-8 -*-
# @Time   : 2026/9/5
# @Author : Junze Yang / revised with ChatGPT
# @File   : accessibility.py

"""Full-network structural and hospital accessibility using road distance only.

Key changes from the first version
----------------------------------
* No fixed Ho Man Tin / central-Kowloon bounding box is used.
* The 100 m grid is generated from the full current road-network envelope and
  therefore covers the entire network. By default every grid centroid is kept
  and snapped to its nearest road node; there is no 200 m exclusion hole.
* All accessibility uses directed shortest-path distance (``length_m``).
* Nearby Hospital Authority hospitals are selected relative to the *dynamic*
  full-network envelope with a configurable external buffer, reducing boundary
  bias. Hospitals are snapped once and are not re-snapped after disruption.
* Progressive hospital-accessibility disruption reuses the exact random and
  four targeted removal sequences produced by the revised removal.py.

Required upstream files
-----------------------
    node.csv
    link.csv
    res/graph_metric/node_metric.csv
    res/graph_metric/link_metric.csv
    res/removal/removal_sequence.csv
    res/removal/removal_curve.csv

Outputs
-------
    res/accessibility/node_access.csv
    res/accessibility/grid.csv
    res/accessibility/hospital.csv
    res/accessibility/hospital_access.csv
    res/accessibility/hospital_summary.csv
    res/accessibility/hospital_catchment.csv
    res/accessibility/access_impact.csv
    res/accessibility/access_removal_raw.csv
    res/accessibility/access_removal_curve.csv
    res/accessibility/access_removal_summary.csv
    res/accessibility/accessibility_meta.json

Convenience spatial layers are written to ``res/shp/accessibility`` when
GeoPandas can write ESRI Shapefiles.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import math
import multiprocessing as mp
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from scipy.integrate import trapezoid
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely import wkt as shapely_wkt
from shapely.geometry import Point, box

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =============================================================================
# Paths / settings
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NODE_PATH = PROJECT_ROOT / "node.csv"
LINK_PATH = PROJECT_ROOT / "link.csv"

GRAPH_DIR = PROJECT_ROOT / "res" / "graph_metric"
NODE_METRIC_PATH = GRAPH_DIR / "node_metric.csv"
LINK_METRIC_PATH = GRAPH_DIR / "link_metric.csv"

REMOVAL_DIR = PROJECT_ROOT / "res" / "removal"
REMOVAL_SEQUENCE_PATH = REMOVAL_DIR / "removal_sequence.csv"
REMOVAL_CURVE_PATH = REMOVAL_DIR / "removal_curve.csv"

RESULT_DIR = PROJECT_ROOT / "res" / "accessibility"
CACHE_DIR = RESULT_DIR / "cache"
SHP_DIR = PROJECT_ROOT / "res" / "shp" / "accessibility"

NODE_ACCESS_OUT = RESULT_DIR / "node_access.csv"
GRID_OUT = RESULT_DIR / "grid.csv"
HOSPITAL_OUT = RESULT_DIR / "hospital.csv"
HOSPITAL_ACCESS_OUT = RESULT_DIR / "hospital_access.csv"
HOSPITAL_SUMMARY_OUT = RESULT_DIR / "hospital_summary.csv"
HOSPITAL_CATCHMENT_OUT = RESULT_DIR / "hospital_catchment.csv"
ACCESS_IMPACT_OUT = RESULT_DIR / "access_impact.csv"
ACCESS_REMOVAL_RAW_OUT = RESULT_DIR / "access_removal_raw.csv"
ACCESS_REMOVAL_CURVE_OUT = RESULT_DIR / "access_removal_curve.csv"
ACCESS_REMOVAL_SUMMARY_OUT = RESULT_DIR / "access_removal_summary.csv"
META_OUT = RESULT_DIR / "accessibility_meta.json"

HA_CACHE_PATH = CACHE_DIR / "facility-hosp.json"
HA_HOSPITAL_URL = "https://www.ha.org.hk/opendata/facility-hosp.json"

WGS84 = "EPSG:4326"
HK80 = "EPSG:2326"
DEFAULT_GRID_SIZE_M = 100.0
DEFAULT_GRID_PADDING_M = 100.0
DEFAULT_HOSPITAL_BUFFER_M = 1500.0
DEFAULT_HOSPITAL_SNAP_MAX_M = 1500.0
DEFAULT_WORKERS = max(1, min((os.cpu_count() or 2) - 1, 8))
DEFAULT_TEST_COMPONENTS = 30
EPS = 1e-12

_W: Dict[str, Any] = {}


# =============================================================================
# Helpers
# =============================================================================

def read_csv_auto(path: Path) -> Tuple[pd.DataFrame, str]:
    errors = []
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc), enc
        except UnicodeDecodeError as exc:
            errors.append("%s: %s" % (enc, exc))
    raise UnicodeError("Cannot decode %s. %s" % (path, errors))


def require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError("%s is missing required columns: %s" % (name, missing))


def normalized_auc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return np.nan
    order = np.argsort(x)
    x, y = x[order], y[order]
    span = float(x[-1] - x[0])
    return float(trapezoid(y, x) / span) if span > 0 else np.nan


def fraction_counts(total: int, fractions: Sequence[float]) -> List[int]:
    counts = []
    previous = 0
    for f in fractions:
        k = 0 if f <= 0 else max(1, int(round(float(f) * total)))
        k = min(total, max(previous, k))
        counts.append(k)
        previous = k
    return counts


# =============================================================================
# Network
# =============================================================================

def prepare_network(nodes: pd.DataFrame, links: pd.DataFrame) -> Dict[str, Any]:
    node_ids = np.asarray(
        sorted(pd.to_numeric(nodes["node_id"], errors="raise").astype(np.int64)),
        dtype=np.int64,
    )
    n = len(node_ids)
    node_to_idx = {int(x): i for i, x in enumerate(node_ids)}

    work = links[["link_id", "from_node", "to_node", "length_m"]].copy()
    for c in ("link_id", "from_node", "to_node"):
        work[c] = pd.to_numeric(work[c], errors="raise").astype(np.int64)
    work["length_m"] = pd.to_numeric(work["length_m"], errors="raise").astype(float)
    work["u_idx"] = work["from_node"].map(node_to_idx).astype(np.int32)
    work["v_idx"] = work["to_node"].map(node_to_idx).astype(np.int32)
    work["_pos"] = np.arange(len(work), dtype=np.int32)

    link_ids = work["link_id"].to_numpy(dtype=np.int64)
    lengths = work["length_m"].to_numpy(dtype=np.float64)
    rows: List[int] = []
    cols: List[int] = []
    groups: List[np.ndarray] = []
    for (u, v), grp in work.groupby(["u_idx", "v_idx"], sort=True):
        grp = grp.sort_values(["length_m", "link_id"])
        rows.append(int(u)); cols.append(int(v))
        groups.append(grp["_pos"].to_numpy(dtype=np.int32))

    rows_a = np.asarray(rows, dtype=np.int32)
    cols_a = np.asarray(cols, dtype=np.int32)
    base_w = np.asarray([lengths[int(g[0])] for g in groups], dtype=np.float64)
    adj = csr_matrix((base_w, (rows_a, cols_a)), shape=(n, n), dtype=np.float64)

    link_affects_isolated = np.zeros(len(link_ids), dtype=bool)
    for g in groups:
        first = int(g[0])
        if len(g) == 1 or lengths[int(g[1])] > lengths[first] + 1e-9:
            link_affects_isolated[first] = True

    return {
        "n": n,
        "node_ids": node_ids,
        "node_to_idx": node_to_idx,
        "link_ids": link_ids,
        "link_id_to_pos": {int(x): i for i, x in enumerate(link_ids)},
        "link_lengths": lengths,
        "pair_rows": rows_a,
        "pair_cols": cols_a,
        "pair_groups": tuple(groups),
        "base_weights": base_w,
        "base_adj": adj,
        "n_links": len(link_ids),
        "link_affects_isolated": link_affects_isolated,
    }


# =============================================================================
# Structural node accessibility
# =============================================================================

def compute_node_access(nodes: pd.DataFrame, network: Dict[str, Any]) -> pd.DataFrame:
    print("Computing full-network directed structural accessibility...")
    dist = dijkstra(network["base_adj"], directed=True, return_predecessors=False)
    valid = np.isfinite(dist) & (dist > 0)
    reach = valid.sum(axis=1).astype(np.int64)
    reach_share = reach / max(1, network["n"] - 1)
    inv = np.zeros_like(dist, dtype=np.float64)
    inv[valid] = 1000.0 / dist[valid]  # km^-1
    harmonic = inv.sum(axis=1) / max(1, network["n"] - 1)

    lookup = nodes.set_index("node_id")
    out = pd.DataFrame({
        "node_id": network["node_ids"],
        "harmonic_access": harmonic,
        "reachable_count": reach,
        "reachable_share": reach_share,
    })
    out["lon"] = out["node_id"].map(lookup["lon"]).astype(float)
    out["lat"] = out["node_id"].map(lookup["lat"]).astype(float)
    if "osm_id" in lookup.columns:
        out["osm_id"] = out["node_id"].map(lookup["osm_id"])
    out["access_rank"] = out["harmonic_access"].rank(method="min", ascending=False).astype(int)
    out["access_pct"] = out["harmonic_access"].rank(method="average", pct=True)
    return out


# =============================================================================
# Full-network 100 m grid
# =============================================================================

def project_nodes(nodes: pd.DataFrame) -> Tuple[pd.DataFrame, Transformer, Transformer]:
    to_hk = Transformer.from_crs(WGS84, HK80, always_xy=True)
    to_wgs = Transformer.from_crs(HK80, WGS84, always_xy=True)
    out = nodes.copy()
    x, y = to_hk.transform(
        pd.to_numeric(out["lon"], errors="raise").to_numpy(dtype=float),
        pd.to_numeric(out["lat"], errors="raise").to_numpy(dtype=float),
    )
    out["x_hk80"] = np.asarray(x, dtype=float)
    out["y_hk80"] = np.asarray(y, dtype=float)
    return out, to_hk, to_wgs


def make_full_network_grid(
    nodes_hk: pd.DataFrame,
    to_wgs: Transformer,
    grid_size_m: float,
    padding_m: float,
) -> Tuple[pd.DataFrame, List[Any], Dict[str, float]]:
    minx = float(nodes_hk["x_hk80"].min()) - float(padding_m)
    maxx = float(nodes_hk["x_hk80"].max()) + float(padding_m)
    miny = float(nodes_hk["y_hk80"].min()) - float(padding_m)
    maxy = float(nodes_hk["y_hk80"].max()) + float(padding_m)
    g = float(grid_size_m)
    x0, x1 = math.floor(minx / g) * g, math.ceil(maxx / g) * g
    y0, y1 = math.floor(miny / g) * g, math.ceil(maxy / g) * g

    node_xy = nodes_hk[["x_hk80", "y_hk80"]].to_numpy(dtype=float)
    tree = cKDTree(node_xy)

    rows: List[Dict[str, Any]] = []
    geoms: List[Any] = []
    gid = 0
    for y in np.arange(y0, y1, g):
        for x in np.arange(x0, x1, g):
            cx, cy = float(x + g / 2.0), float(y + g / 2.0)
            snap_d, idx = tree.query([cx, cy], k=1)
            lon, lat = to_wgs.transform(cx, cy)
            rows.append({
                "grid_id": gid,
                "x_hk80": cx,
                "y_hk80": cy,
                "lon": float(lon),
                "lat": float(lat),
                "node_id": int(nodes_hk.iloc[int(idx)]["node_id"]),
                "snap_dist_m": float(snap_d),
                "valid_origin": True,
            })
            geoms.append(box(float(x), float(y), float(x + g), float(y + g)))
            gid += 1

    grid = pd.DataFrame(rows)
    if grid.empty:
        raise RuntimeError("Full-network grid generation produced no cells.")
    return grid, geoms, {
        "minx_hk80": minx, "miny_hk80": miny,
        "maxx_hk80": maxx, "maxy_hk80": maxy,
    }


# =============================================================================
# Hospital Authority data
# =============================================================================

def load_hospital_records(local_json: Optional[Path], refresh: bool) -> Tuple[List[Dict[str, Any]], str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if local_json is not None:
        p = local_json.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError("--hospital-json does not exist: %s" % p)
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        if not isinstance(data, list):
            raise ValueError("Hospital JSON must be a list.")
        return data, str(p)
    if HA_CACHE_PATH.exists() and not refresh:
        return json.loads(HA_CACHE_PATH.read_text(encoding="utf-8-sig")), "cache"
    try:
        r = requests.get(
            HA_HOSPITAL_URL,
            timeout=40,
            headers={"User-Agent": "HoManTin-Network-Resilience/1.0 academic analysis"},
        )
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            raise ValueError("Hospital Authority response is not a list.")
        HA_CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data, HA_HOSPITAL_URL
    except Exception as exc:
        if HA_CACHE_PATH.exists():
            warnings.warn("Hospital download failed; using cache: %s" % exc)
            return json.loads(HA_CACHE_PATH.read_text(encoding="utf-8-sig")), "stale cache"
        raise RuntimeError(
            "Could not download Hospital Authority data and no cache exists. "
            "Pass --hospital-json PATH. Error: %s" % exc
        )


def select_and_snap_hospitals(
    records: List[Dict[str, Any]],
    nodes_hk: pd.DataFrame,
    to_hk: Transformer,
    network_bbox: Dict[str, float],
    buffer_m: float,
    max_snap_m: float,
) -> pd.DataFrame:
    df = pd.DataFrame(records)
    require_columns(
        df,
        ["institution_eng", "institution_tc", "latitude", "longitude"],
        "Hospital Authority JSON",
    )
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df[
        df["institution_eng"].astype(str).str.contains(r"\bHospital\b", case=False, regex=True, na=False)
        & np.isfinite(df["latitude"])
        & np.isfinite(df["longitude"])
    ].copy()
    if df.empty:
        raise RuntimeError("No hospital record is available in the HA dataset.")

    hx, hy = to_hk.transform(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    df["x_hk80"] = np.asarray(hx, dtype=float)
    df["y_hk80"] = np.asarray(hy, dtype=float)
    b = float(buffer_m)
    inside = (
        df["x_hk80"].between(network_bbox["minx_hk80"] - b, network_bbox["maxx_hk80"] + b)
        & df["y_hk80"].between(network_bbox["miny_hk80"] - b, network_bbox["maxy_hk80"] + b)
    )
    df = df.loc[inside].copy()

    tree = cKDTree(nodes_hk[["x_hk80", "y_hk80"]].to_numpy(dtype=float))
    dist, idx = tree.query(df[["x_hk80", "y_hk80"]].to_numpy(dtype=float), k=1)
    df["node_id"] = nodes_hk.iloc[np.asarray(idx, dtype=int)]["node_id"].to_numpy(dtype=np.int64)
    df["snap_dist_m"] = np.asarray(dist, dtype=float)
    df = df[df["snap_dist_m"] <= float(max_snap_m)].copy()
    if df.empty:
        raise RuntimeError("No nearby Hospital Authority hospital can be snapped to the current network.")

    df = df.sort_values(["institution_eng", "latitude", "longitude"]).reset_index(drop=True)
    df.insert(0, "hospital_id", np.arange(len(df), dtype=np.int64))
    if "with_AE_service_eng" in df.columns:
        df["with_ae"] = df["with_AE_service_eng"].astype(str).str.strip().str.lower().eq("yes")
    else:
        df["with_ae"] = False
    keep = [c for c in [
        "hospital_id", "institution_eng", "institution_tc", "cluster_eng", "address_eng",
        "with_ae", "latitude", "longitude", "x_hk80", "y_hk80", "node_id", "snap_dist_m"
    ] if c in df.columns]
    return df[keep]


# =============================================================================
# Hospital shortest distance / baseline summaries
# =============================================================================

def hospital_distances(
    adj: csr_matrix,
    hospital_node_idx: np.ndarray,
    hospital_ids: np.ndarray,
    removed_node_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if removed_node_mask is None:
        keep = np.ones(len(hospital_node_idx), dtype=bool)
    else:
        keep = ~removed_node_mask[hospital_node_idx]
    active_nodes = hospital_node_idx[keep]
    active_ids = hospital_ids[keep]
    n = adj.shape[0]
    if len(active_nodes) == 0:
        return np.full(n, np.inf), np.full(n, -1, dtype=np.int64)

    dist_h = dijkstra(adj.transpose().tocsr(), directed=True, indices=active_nodes, return_predecessors=False)
    if dist_h.ndim == 1:
        dist_h = dist_h.reshape(1, -1)
    arg = np.argmin(dist_h, axis=0)
    mind = dist_h[arg, np.arange(n)]
    hid = np.asarray(active_ids[arg], dtype=np.int64)
    hid[~np.isfinite(mind)] = -1
    if removed_node_mask is not None:
        mind = np.asarray(mind, dtype=float)
        mind[removed_node_mask] = np.inf
        hid[removed_node_mask] = -1
    return np.asarray(mind, dtype=float), hid


def summarize_distances(d: np.ndarray) -> Dict[str, float]:
    d = np.asarray(d, dtype=float)
    reach = np.isfinite(d)
    n = len(d)
    nr = int(reach.sum())
    if nr == 0:
        return {
            "origin_count": n, "reachable_count": 0, "reachable_share": 0.0,
            "mean_m": np.nan, "median_m": np.nan, "p90_m": np.nan, "p95_m": np.nan,
        }
    x = d[reach]
    return {
        "origin_count": int(n),
        "reachable_count": nr,
        "reachable_share": float(nr / n) if n else np.nan,
        "mean_m": float(np.mean(x)),
        "median_m": float(np.median(x)),
        "p90_m": float(np.quantile(x, 0.90)),
        "p95_m": float(np.quantile(x, 0.95)),
    }


def baseline_hospital_access(
    network: Dict[str, Any], grid: pd.DataFrame, hospitals: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    grid_idx = np.asarray([network["node_to_idx"][int(x)] for x in grid["node_id"]], dtype=np.int32)
    hosp_idx = np.asarray([network["node_to_idx"][int(x)] for x in hospitals["node_id"]], dtype=np.int32)
    hosp_ids = hospitals["hospital_id"].to_numpy(dtype=np.int64)
    node_d, node_h = hospital_distances(network["base_adj"], hosp_idx, hosp_ids)
    d = node_d[grid_idx]
    h = node_h[grid_idx]

    name = hospitals.set_index("hospital_id")["institution_eng"].to_dict()
    name_tc = hospitals.set_index("hospital_id")["institution_tc"].to_dict()
    access = grid.copy()
    access["reachable"] = np.isfinite(d)
    access["nearest_hospital_id"] = np.where(access["reachable"], h, -1).astype(np.int64)
    access["nearest_hospital"] = access["nearest_hospital_id"].map(name)
    access["nearest_hospital_tc"] = access["nearest_hospital_id"].map(name_tc)
    access["distance_m"] = np.where(access["reachable"], d, np.nan)
    access["distance_km"] = access["distance_m"] / 1000.0

    s = summarize_distances(d)
    summary = pd.DataFrame([{
        **s,
        "unreachable_count": int(len(d) - np.isfinite(d).sum()),
        "unreachable_share": float(1.0 - s["reachable_share"]),
        "mean_km": s["mean_m"] / 1000.0 if np.isfinite(s["mean_m"]) else np.nan,
        "median_km": s["median_m"] / 1000.0 if np.isfinite(s["median_m"]) else np.nan,
        "p90_km": s["p90_m"] / 1000.0 if np.isfinite(s["p90_m"]) else np.nan,
        "p95_km": s["p95_m"] / 1000.0 if np.isfinite(s["p95_m"]) else np.nan,
    }])

    catch_rows = []
    reachable_access = access[access["reachable"]]
    for hosp in hospitals.itertuples(index=False):
        grp = reachable_access[reachable_access["nearest_hospital_id"] == int(hosp.hospital_id)]
        vals = grp["distance_m"].to_numpy(dtype=float)
        catch_rows.append({
            "hospital_id": int(hosp.hospital_id),
            "institution_eng": hosp.institution_eng,
            "institution_tc": hosp.institution_tc,
            "with_ae": bool(hosp.with_ae),
            "grid_count": int(len(grp)),
            "grid_share": float(len(grp) / len(access)) if len(access) else np.nan,
            "mean_m": float(np.mean(vals)) if len(vals) else np.nan,
            "median_m": float(np.median(vals)) if len(vals) else np.nan,
            "p90_m": float(np.quantile(vals, 0.90)) if len(vals) else np.nan,
        })
    return access, summary, pd.DataFrame(catch_rows), s, grid_idx, hosp_idx, hosp_ids


# =============================================================================
# Disruption workers
# =============================================================================

def init_worker(
    n: int, pair_rows: np.ndarray, pair_cols: np.ndarray,
    pair_groups: Tuple[np.ndarray, ...], base_weights: np.ndarray,
    link_ids: np.ndarray, link_lengths: np.ndarray, node_ids: np.ndarray,
    link_affects_isolated: np.ndarray, grid_idx: np.ndarray,
    hosp_idx: np.ndarray, hosp_ids: np.ndarray, baseline_metrics: Dict[str, float],
) -> None:
    global _W
    _W = {
        "n": int(n), "pair_rows": np.asarray(pair_rows, dtype=np.int32),
        "pair_cols": np.asarray(pair_cols, dtype=np.int32),
        "pair_groups": tuple(np.asarray(x, dtype=np.int32) for x in pair_groups),
        "base_weights": np.asarray(base_weights, dtype=np.float64),
        "link_ids": np.asarray(link_ids, dtype=np.int64),
        "link_lengths": np.asarray(link_lengths, dtype=np.float64),
        "link_id_to_pos": {int(x): i for i, x in enumerate(link_ids)},
        "node_ids": np.asarray(node_ids, dtype=np.int64),
        "node_id_to_idx": {int(x): i for i, x in enumerate(node_ids)},
        "link_affects_isolated": np.asarray(link_affects_isolated, dtype=bool),
        "grid_idx": np.asarray(grid_idx, dtype=np.int32),
        "hosp_idx": np.asarray(hosp_idx, dtype=np.int32),
        "hosp_ids": np.asarray(hosp_ids, dtype=np.int64),
        "baseline_metrics": dict(baseline_metrics),
        "n_links": int(len(link_ids)),
    }


def adj_after_links(removed: np.ndarray) -> csr_matrix:
    weights = np.empty(len(_W["pair_groups"]), dtype=np.float64)
    keep = np.zeros(len(_W["pair_groups"]), dtype=bool)
    for p, positions in enumerate(_W["pair_groups"]):
        chosen = -1
        for pos in positions:
            if not removed[int(pos)]:
                chosen = int(pos); break
        if chosen >= 0:
            weights[p] = _W["link_lengths"][chosen]; keep[p] = True
    return csr_matrix(
        (weights[keep], (_W["pair_rows"][keep], _W["pair_cols"][keep])),
        shape=(_W["n"], _W["n"]), dtype=np.float64,
    )


def adj_after_nodes(removed: np.ndarray) -> csr_matrix:
    keep = (~removed[_W["pair_rows"]]) & (~removed[_W["pair_cols"]])
    return csr_matrix(
        (_W["base_weights"][keep], (_W["pair_rows"][keep], _W["pair_cols"][keep])),
        shape=(_W["n"], _W["n"]), dtype=np.float64,
    )


def state_hospital_metrics(adj: csr_matrix, removed_nodes: Optional[np.ndarray] = None) -> Dict[str, float]:
    node_d, _ = hospital_distances(adj, _W["hosp_idx"], _W["hosp_ids"], removed_nodes)
    grid_d = node_d[_W["grid_idx"]].copy()
    if removed_nodes is not None:
        grid_d[removed_nodes[_W["grid_idx"]]] = np.inf
    return summarize_distances(grid_d)


def impact_row(component: str, cid: int, post: Dict[str, float]) -> Dict[str, Any]:
    base = _W["baseline_metrics"]
    return {
        "component": component,
        "component_id": int(cid),
        "reachable_share": float(post["reachable_share"]),
        "reach_loss": float(base["reachable_share"] - post["reachable_share"]),
        "mean_m": float(post["mean_m"]),
        "median_m": float(post["median_m"]),
        "p90_m": float(post["p90_m"]),
        "p95_m": float(post["p95_m"]),
        "mean_increase_m": float(post["mean_m"] - base["mean_m"]) if np.isfinite(post["mean_m"]) else np.nan,
        "median_increase_m": float(post["median_m"] - base["median_m"]) if np.isfinite(post["median_m"]) else np.nan,
        "p90_increase_m": float(post["p90_m"] - base["p90_m"]) if np.isfinite(post["p90_m"]) else np.nan,
        "p95_increase_m": float(post["p95_m"] - base["p95_m"]) if np.isfinite(post["p95_m"]) else np.nan,
    }


def eval_link_chunk(ids: Sequence[int]) -> List[Dict[str, Any]]:
    rows = []
    for lid in ids:
        pos = _W["link_id_to_pos"][int(lid)]
        if not _W["link_affects_isolated"][pos]:
            post = _W["baseline_metrics"]
        else:
            removed = np.zeros(_W["n_links"], dtype=bool); removed[pos] = True
            post = state_hospital_metrics(adj_after_links(removed))
        rows.append(impact_row("link", int(lid), post))
    return rows


def eval_node_chunk(ids: Sequence[int]) -> List[Dict[str, Any]]:
    rows = []
    for nid in ids:
        idx = _W["node_id_to_idx"][int(nid)]
        removed = np.zeros(_W["n"], dtype=bool); removed[idx] = True
        post = state_hospital_metrics(adj_after_nodes(removed), removed)
        rows.append(impact_row("node", int(nid), post))
    return rows


def chunk_ids(ids: Sequence[int], workers: int) -> List[np.ndarray]:
    arr = np.asarray(ids, dtype=np.int64)
    if len(arr) == 0:
        return []
    n = max(1, min(len(arr), workers * 8))
    return [x for x in np.array_split(arr, n) if len(x)]


def run_isolated(network: Dict[str, Any], workers: int, initargs: Tuple[Any, ...], test_n: Optional[int]) -> pd.DataFrame:
    link_ids = network["link_ids"] if test_n is None else network["link_ids"][:test_n]
    node_ids = network["node_ids"] if test_n is None else network["node_ids"][:test_n]
    tasks = [("link", x) for x in chunk_ids(link_ids, workers)] + [("node", x) for x in chunk_ids(node_ids, workers)]
    rows: List[Dict[str, Any]] = []
    print("Evaluating isolated hospital-access impacts: %d links, %d nodes..." % (len(link_ids), len(node_ids)))
    if workers <= 1:
        init_worker(*initargs)
        iterator = tasks
        if tqdm is not None:
            iterator = tqdm(iterator, desc="hospital isolated impacts", unit="chunk")
        for comp, ids in iterator:
            rows.extend(eval_link_chunk(ids) if comp == "link" else eval_node_chunk(ids))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=initargs) as ex:
            futures = [ex.submit(eval_link_chunk if c == "link" else eval_node_chunk, ids) for c, ids in tasks]
            iterator = as_completed(futures)
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(futures), desc="hospital isolated impacts", unit="chunk")
            for fut in iterator:
                rows.extend(fut.result())
    return pd.DataFrame(rows).sort_values(["component", "component_id"]).reset_index(drop=True)


def eval_progressive_task(task: Tuple[str, str, int, Sequence[int], Sequence[float]]) -> List[Dict[str, Any]]:
    component, strategy, run, sequence, fractions = task
    total = _W["n_links"] if component == "link" else _W["n"]
    counts = fraction_counts(total, fractions)
    targets: Dict[int, List[float]] = {}
    for f, k in zip(fractions, counts):
        targets.setdefault(k, []).append(float(f))

    if component == "link":
        removed = np.zeros(_W["n_links"], dtype=bool)
        def remove(cid: int): removed[_W["link_id_to_pos"][int(cid)]] = True
        def state(): return state_hospital_metrics(adj_after_links(removed))
    else:
        removed = np.zeros(_W["n"], dtype=bool)
        def remove(cid: int): removed[_W["node_id_to_idx"][int(cid)]] = True
        def state(): return state_hospital_metrics(adj_after_nodes(removed), removed)

    rows = []
    kdone = 0
    for k in sorted(targets):
        while kdone < k:
            remove(int(sequence[kdone])); kdone += 1
        post = state()
        for f in targets[k]:
            rows.append({
                "component": component, "strategy": strategy, "run": int(run),
                "target_fraction": f, "fraction": float(k / total), "n_removed": int(k),
                "reachable_share": float(post["reachable_share"]),
                "reach_loss": float(_W["baseline_metrics"]["reachable_share"] - post["reachable_share"]),
                "mean_m": float(post["mean_m"]), "median_m": float(post["median_m"]),
                "p90_m": float(post["p90_m"]), "p95_m": float(post["p95_m"]),
            })
    return rows


# =============================================================================
# Progressive aggregation / graph-metric merge
# =============================================================================

def build_progressive_tasks(seq: pd.DataFrame, curve: pd.DataFrame) -> List[Tuple[str, str, int, np.ndarray, Tuple[float, ...]]]:
    tasks = []
    for (component, strategy, run), grp in seq.groupby(["component", "strategy", "run"], sort=True):
        order = grp.sort_values("order")["component_id"].to_numpy(dtype=np.int64)
        f = curve[(curve["component"] == component) & (curve["strategy"] == strategy)]["target_fraction"]
        fractions = tuple(sorted(set(pd.to_numeric(f, errors="coerce").dropna().astype(float))))
        if fractions:
            tasks.append((str(component), str(strategy), int(run), order, fractions))
    return tasks


def aggregate_access_curves(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["component", "strategy", "target_fraction", "fraction", "n_removed"]
    metrics = ["reachable_share", "reach_loss", "mean_m", "median_m", "p90_m", "p95_m"]
    rows = []
    for key, grp in raw.groupby(keys, sort=True):
        row = {
            "component": key[0], "strategy": key[1], "target_fraction": float(key[2]),
            "fraction": float(key[3]), "n_removed": int(key[4]),
            "runs": int(grp["run"].nunique()),
        }
        for m in metrics:
            x = pd.to_numeric(grp[m], errors="coerce").dropna().to_numpy(dtype=float)
            row[m] = float(np.mean(x)) if len(x) else np.nan
            row[m + "_std"] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0 if len(x) else np.nan
            row[m + "_p025"] = float(np.quantile(x, 0.025)) if len(x) else np.nan
            row[m + "_p975"] = float(np.quantile(x, 0.975)) if len(x) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["component", "strategy", "fraction"]).reset_index(drop=True)


def access_curve_summary(curve: pd.DataFrame, baseline: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for (component, strategy), grp in curve.groupby(["component", "strategy"], sort=True):
        grp = grp.sort_values("fraction")
        x = grp["fraction"].to_numpy(dtype=float)
        rows.append({
            "component": component, "strategy": strategy,
            "metric": "reachable_share", "auc": normalized_auc(x, grp["reachable_share"].to_numpy(dtype=float))
        })
        for metric in ("mean_m", "median_m", "p90_m", "p95_m"):
            y = grp[metric].to_numpy(dtype=float)
            base = float(baseline[metric])
            yn = y / base if base > 0 else np.full_like(y, np.nan)
            rows.append({
                "component": component, "strategy": strategy,
                "metric": metric + "_ratio", "auc": normalized_auc(x, yn)
            })
    return pd.DataFrame(rows)


def merge_graph_metrics(impact: pd.DataFrame) -> pd.DataFrame:
    parts = []
    lm, _ = read_csv_auto(LINK_METRIC_PATH)
    nm, _ = read_csv_auto(NODE_METRIC_PATH)
    metric_cols = ["odbc", "od_eff_loss", "lscc_loss", "td_loss"]

    link = impact[impact["component"] == "link"].copy()
    if not link.empty:
        keep = ["link_id"] + [c for c in metric_cols if c in lm.columns]
        link = link.merge(lm[keep].rename(columns={"link_id": "component_id"}), on="component_id", how="left")
    parts.append(link)

    node = impact[impact["component"] == "node"].copy()
    if not node.empty:
        keep = ["node_id"] + [c for c in metric_cols if c in nm.columns]
        node = node.merge(nm[keep].rename(columns={"node_id": "component_id"}), on="component_id", how="left")
    parts.append(node)
    return pd.concat(parts, ignore_index=True, sort=False)


# =============================================================================
# Spatial outputs
# =============================================================================

# ESRI Shapefile/DBF field names are limited to 10 characters.  Do not let
# GeoPandas/Fiona truncate names implicitly because different long names can
# collapse to the same 10-character prefix and make downstream plotting
# ambiguous.  CSV outputs remain authoritative and keep their full names.
SHP_FIELD_MAP = {
    # grid
    "snap_dist_m": "snap_dist",
    "valid_origin": "valid_org",
    # hospitals
    "hospital_id": "hosp_id",
    "institution_eng": "inst_eng",
    "institution_tc": "inst_tc",
    "cluster_eng": "clust_eng",
    "address_eng": "addr_eng",
    # structural accessibility
    "harmonic_access": "harm_acc",
    "reachable_share": "reach_sh",
    # isolated hospital-access impact
    "mean_increase_m": "mean_inc",
    "median_increase_m": "med_inc",
    "p90_increase_m": "p90_inc",
    "p95_increase_m": "p95_inc",
    # graph metrics
    "od_eff_loss": "od_effls",
}


def _shp_safe(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """Return a Shapefile-safe copy with deterministic <=10-char fields.

    Known analysis fields use the explicit SHP_FIELD_MAP above.  Any future
    long field that is not yet in the map receives a deterministic unique
    short name rather than being silently truncated by the driver.
    """
    out = gdf.copy()
    geom_name = out.geometry.name

    rename = {
        col: short
        for col, short in SHP_FIELD_MAP.items()
        if col in out.columns and col != geom_name
    }
    out = out.rename(columns=rename)

    used = {str(c).lower() for c in out.columns if c == geom_name or len(str(c)) <= 10}
    extra = {}
    for col in out.columns:
        if col == geom_name or len(str(col)) <= 10:
            continue
        base = re.sub(r"[^A-Za-z0-9_]", "_", str(col)).strip("_") or "field"
        base = base[:8]
        candidate = base[:10]
        k = 1
        while candidate.lower() in used:
            suffix = str(k)
            candidate = (base[: 10 - len(suffix)] + suffix)[:10]
            k += 1
        extra[col] = candidate
        used.add(candidate.lower())

    if extra:
        out = out.rename(columns=extra)

    non_geom = [c for c in out.columns if c != out.geometry.name]
    if any(len(str(c)) > 10 for c in non_geom):
        raise RuntimeError("Internal error: Shapefile field name exceeds 10 characters.")
    lower = [str(c).lower() for c in non_geom]
    if len(lower) != len(set(lower)):
        raise RuntimeError("Internal error: duplicate Shapefile field names after shortening.")
    return out


def write_spatial_outputs(
    nodes: pd.DataFrame, links: pd.DataFrame, node_access: pd.DataFrame,
    grid: pd.DataFrame, grid_geoms: List[Any], hospitals: pd.DataFrame, impact: pd.DataFrame,
) -> None:
    if gpd is None:
        return
    SHP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        nodes_g = gpd.GeoDataFrame(
            nodes.drop(columns=["geometry"], errors="ignore").copy(),
            geometry=gpd.points_from_xy(nodes["lon"], nodes["lat"]),
            crs=WGS84,
        )

        na = nodes_g.merge(
            node_access[["node_id", "harmonic_access", "reachable_share"]],
            on="node_id", how="left",
        )
        _shp_safe(na).to_file(
            SHP_DIR / "node_access.shp", driver="ESRI Shapefile", encoding="UTF-8"
        )

        gg = gpd.GeoDataFrame(grid.copy(), geometry=grid_geoms, crs=HK80)
        _shp_safe(gg).to_file(
            SHP_DIR / "grid_100m.shp", driver="ESRI Shapefile", encoding="UTF-8"
        )

        hg = gpd.GeoDataFrame(
            hospitals.copy(),
            geometry=gpd.points_from_xy(hospitals["longitude"], hospitals["latitude"]),
            crs=WGS84,
        )
        _shp_safe(hg).to_file(
            SHP_DIR / "hospital.shp", driver="ESRI Shapefile", encoding="UTF-8"
        )

        node_imp = (
            impact[impact["component"] == "node"]
            .drop(columns=["component"], errors="ignore")
            .copy()
            .rename(columns={"component_id": "node_id"})
        )
        node_imp = nodes_g.merge(node_imp, on="node_id", how="left")
        _shp_safe(node_imp).to_file(
            SHP_DIR / "node_impact.shp", driver="ESRI Shapefile", encoding="UTF-8"
        )

        if "geometry" in links.columns:
            geom = links["geometry"].apply(
                lambda x: shapely_wkt.loads(x) if isinstance(x, str) and x.strip() else None
            )
            links_base = links.drop(columns=["geometry"], errors="ignore").copy()
            links_g = gpd.GeoDataFrame(links_base, geometry=geom, crs=WGS84)
            link_imp = (
                impact[impact["component"] == "link"]
                .drop(columns=["component"], errors="ignore")
                .copy()
                .rename(columns={"component_id": "link_id"})
            )
            link_imp = links_g.merge(link_imp, on="link_id", how="left")
            _shp_safe(link_imp).to_file(
                SHP_DIR / "link_impact.shp", driver="ESRI Shapefile", encoding="UTF-8"
            )
    except Exception as exc:
        warnings.warn("Could not write one or more accessibility shapefiles: %s" % exc)


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--grid-size", type=float, default=DEFAULT_GRID_SIZE_M)
    parser.add_argument("--grid-padding", type=float, default=DEFAULT_GRID_PADDING_M)
    parser.add_argument("--hospital-buffer", type=float, default=DEFAULT_HOSPITAL_BUFFER_M)
    parser.add_argument("--hospital-snap-max", type=float, default=DEFAULT_HOSPITAL_SNAP_MAX_M)
    parser.add_argument("--hospital-json", type=Path, default=None)
    parser.add_argument("--refresh-hospitals", action="store_true")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    workers = max(1, int(args.workers))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    required = [
        NODE_PATH, LINK_PATH, NODE_METRIC_PATH, LINK_METRIC_PATH,
        REMOVAL_SEQUENCE_PATH, REMOVAL_CURVE_PATH,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing upstream files: %s" % missing)

    nodes, node_enc = read_csv_auto(NODE_PATH)
    links, link_enc = read_csv_auto(LINK_PATH)
    seq, _ = read_csv_auto(REMOVAL_SEQUENCE_PATH)
    removal_curve, _ = read_csv_auto(REMOVAL_CURVE_PATH)
    require_columns(nodes, ["node_id", "lon", "lat"], "node.csv")
    require_columns(links, ["link_id", "from_node", "to_node", "length_m"], "link.csv")

    nodes = nodes.copy(); links = links.copy()
    nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="raise").astype(np.int64)
    links["link_id"] = pd.to_numeric(links["link_id"], errors="raise").astype(np.int64)
    network = prepare_network(nodes, links)

    node_access = compute_node_access(nodes, network)
    node_access.to_csv(NODE_ACCESS_OUT, index=False, encoding="utf-8-sig")

    nodes_hk, to_hk, to_wgs = project_nodes(nodes)
    grid, grid_geoms, bbox = make_full_network_grid(
        nodes_hk, to_wgs, float(args.grid_size), float(args.grid_padding)
    )
    grid.to_csv(GRID_OUT, index=False, encoding="utf-8-sig")

    records, hospital_source = load_hospital_records(args.hospital_json, bool(args.refresh_hospitals))
    hospitals = select_and_snap_hospitals(
        records, nodes_hk, to_hk, bbox,
        float(args.hospital_buffer), float(args.hospital_snap_max),
    )
    hospitals.to_csv(HOSPITAL_OUT, index=False, encoding="utf-8-sig")

    access, summary, catchment, baseline_metrics, grid_idx, hosp_idx, hosp_ids = baseline_hospital_access(
        network, grid, hospitals
    )
    access.to_csv(HOSPITAL_ACCESS_OUT, index=False, encoding="utf-8-sig")
    summary.to_csv(HOSPITAL_SUMMARY_OUT, index=False, encoding="utf-8-sig")
    catchment.to_csv(HOSPITAL_CATCHMENT_OUT, index=False, encoding="utf-8-sig")

    initargs = (
        network["n"], network["pair_rows"], network["pair_cols"], network["pair_groups"],
        network["base_weights"], network["link_ids"], network["link_lengths"], network["node_ids"],
        network["link_affects_isolated"], grid_idx, hosp_idx, hosp_ids, baseline_metrics,
    )
    test_n = DEFAULT_TEST_COMPONENTS if args.test else None
    impact = run_isolated(network, workers, initargs, test_n)
    impact = merge_graph_metrics(impact)
    impact.to_csv(ACCESS_IMPACT_OUT, index=False, encoding="utf-8-sig")

    tasks = build_progressive_tasks(seq, removal_curve)
    if args.test:
        tasks = [t for t in tasks if t[2] in (0, 1)][:10]
    print("Evaluating %d hospital-access progressive sequences..." % len(tasks))
    prog_rows: List[Dict[str, Any]] = []
    if workers <= 1:
        init_worker(*initargs)
        iterator = tasks
        if tqdm is not None:
            iterator = tqdm(iterator, desc="hospital progressive", unit="sequence")
        for task in iterator:
            prog_rows.extend(eval_progressive_task(task))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=initargs) as ex:
            futures = [ex.submit(eval_progressive_task, t) for t in tasks]
            iterator = as_completed(futures)
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(futures), desc="hospital progressive", unit="sequence")
            for fut in iterator:
                prog_rows.extend(fut.result())

    prog_raw = pd.DataFrame(prog_rows)
    if prog_raw.empty:
        raise RuntimeError("No progressive hospital-access results were produced.")
    prog_raw = prog_raw.sort_values(["component", "strategy", "run", "fraction"]).reset_index(drop=True)
    prog_curve = aggregate_access_curves(prog_raw)
    prog_summary = access_curve_summary(prog_curve, baseline_metrics)
    prog_raw.to_csv(ACCESS_REMOVAL_RAW_OUT, index=False, encoding="utf-8-sig")
    prog_curve.to_csv(ACCESS_REMOVAL_CURVE_OUT, index=False, encoding="utf-8-sig")
    prog_summary.to_csv(ACCESS_REMOVAL_SUMMARY_OUT, index=False, encoding="utf-8-sig")

    write_spatial_outputs(nodes, links, node_access, grid, grid_geoms, hospitals, impact)

    runtime = time.perf_counter() - started
    meta = {
        "input": {"node_encoding": node_enc, "link_encoding": link_enc},
        "study_domain": {
            "definition": "full current road-network envelope",
            "bbox_hk80": bbox,
            "grid_size_m": float(args.grid_size),
            "grid_padding_m": float(args.grid_padding),
            "grid_cell_count": int(len(grid)),
            "all_grid_cells_retained": True,
            "max_grid_snap_distance_m": float(grid["snap_dist_m"].max()),
        },
        "hospitals": {
            "source": hospital_source,
            "count": int(len(hospitals)),
            "external_buffer_m": float(args.hospital_buffer),
            "snap_max_m": float(args.hospital_snap_max),
        },
        "distance_model": "directed shortest-path road distance using length_m only",
        "baseline_hospital_access": baseline_metrics,
        "progressive_strategies": sorted(prog_raw["strategy"].unique().tolist()),
        "runtime_seconds": float(runtime),
        "outputs": {
            "node_access": str(NODE_ACCESS_OUT), "grid": str(GRID_OUT),
            "hospital": str(HOSPITAL_OUT), "hospital_access": str(HOSPITAL_ACCESS_OUT),
            "hospital_summary": str(HOSPITAL_SUMMARY_OUT), "hospital_catchment": str(HOSPITAL_CATCHMENT_OUT),
            "access_impact": str(ACCESS_IMPACT_OUT), "access_removal_raw": str(ACCESS_REMOVAL_RAW_OUT),
            "access_removal_curve": str(ACCESS_REMOVAL_CURVE_OUT),
            "access_removal_summary": str(ACCESS_REMOVAL_SUMMARY_OUT),
        },
        "test_mode": bool(args.test),
    }
    META_OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nCompleted full-network accessibility analysis.")
    print("  100 m grid cells: %d (all retained)" % len(grid))
    print("  Hospitals: %d" % len(hospitals))
    print("  Baseline median hospital distance: %.3f km" % (baseline_metrics["median_m"] / 1000.0))
    print("  Runtime: %.1f s" % runtime)
    print("  Saved: %s" % RESULT_DIR)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
