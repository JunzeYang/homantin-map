# -*- coding: utf-8 -*-
# @Time   : 2026/9/5
# @Author : Junze Yang / revised with ChatGPT
# @File   : removal.py

"""Progressive random and targeted disruption analysis using OD-aware metrics.

Required upstream results
-------------------------
    res/graph_metric/node_metric.csv
    res/graph_metric/link_metric.csv
    res/graph_metric/metric_meta.json
    res/graph_metric/cache/od_active.csv

Attack strategies (nodes and directed links separately)
-------------------------------------------------------
* Random removal: Monte Carlo cumulative permutations.
* ODBC-targeted: descending ``odbc``.
* OD-efficiency-targeted: descending ``od_eff_loss``.
* LSCC-targeted: descending ``lscc_loss``.
* Total-distance-targeted: descending ``td_loss``.

State performance
-----------------
All shortest paths use ``length_m`` only.

* ``od_eff_norm``: demand-weighted efficiency / baseline efficiency.
* ``lscc_norm``: largest SCC share / baseline share.
* ``td_loss``: demand-weighted penalized total-distance increase relative to
  baseline, using exactly the same data-derived disconnection penalty as
  graph_metric.py.
* ``reachable_demand_share``: share of baseline-active connector OD demand
  still reachable. This diagnostic is retained because total-distance loss
  uses a finite disconnection penalty.

ODBC is used as a targeted-attack ranking, not forced into a network-level
performance curve: the sum of component betweenness/load is not a stable
system-performance measure under rerouting.

Outputs
-------
    res/removal/removal_raw.csv
    res/removal/removal_curve.csv
    res/removal/removal_summary.csv
    res/removal/removal_sequence.csv
    res/removal/removal_meta.json
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
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

try:
    from scipy.integrate import trapezoid
except Exception:
    trapezoid = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =============================================================================
# Paths / defaults
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NODE_PATH = PROJECT_ROOT / "node.csv"
LINK_PATH = PROJECT_ROOT / "link.csv"

METRIC_DIR = PROJECT_ROOT / "res" / "graph_metric"
NODE_METRIC_PATH = METRIC_DIR / "node_metric.csv"
LINK_METRIC_PATH = METRIC_DIR / "link_metric.csv"
METRIC_META_PATH = METRIC_DIR / "metric_meta.json"
OD_ACTIVE_PATH = METRIC_DIR / "cache" / "od_active.csv"

RESULT_DIR = PROJECT_ROOT / "res" / "removal"
RAW_OUT = RESULT_DIR / "removal_raw.csv"
CURVE_OUT = RESULT_DIR / "removal_curve.csv"
SUMMARY_OUT = RESULT_DIR / "removal_summary.csv"
SEQUENCE_OUT = RESULT_DIR / "removal_sequence.csv"
META_OUT = RESULT_DIR / "removal_meta.json"

MASTER_SEED = 20260905
DEFAULT_RANDOM_RUNS = 30
DEFAULT_WORKERS = max(1, min((os.cpu_count() or 2) - 1, 6))
DEFAULT_FRACTIONS = [0.02 * i for i in range(21)]
EPS = 1e-12

TARGET_METRICS = {
    "odbc": "odbc",
    "od_eff": "od_eff_loss",
    "lscc": "lscc_loss",
    "td": "td_loss",
}

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


def parse_fractions(text: str) -> Tuple[float, ...]:
    try:
        vals = sorted(set(float(x.strip()) for x in text.split(",") if x.strip()))
    except Exception as exc:
        raise argparse.ArgumentTypeError("Invalid --fractions: %s" % exc)
    if not vals or vals[0] < 0 or vals[-1] > 1:
        raise argparse.ArgumentTypeError("fractions must be nonempty and in [0,1]")
    if 0.0 not in vals:
        vals = [0.0] + vals
    return tuple(vals)


def fraction_counts(total: int, fractions: Sequence[float]) -> List[int]:
    counts = []
    previous = 0
    for f in fractions:
        if f <= 0:
            k = 0
        else:
            k = max(1, int(round(float(f) * total)))
        k = min(total, max(previous, k))
        counts.append(k)
        previous = k
    return counts


def stable_descending_order(ids: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    safe = np.where(np.isnan(scores), -np.inf, scores)
    order = np.lexsort((ids, -safe))
    return ids[order], scores[order]


def auc_normalized(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return np.nan
    order = np.argsort(x)
    x, y = x[order], y[order]
    span = float(x[-1] - x[0])
    if span <= 0:
        return np.nan
    area = float(trapezoid(y, x) if trapezoid is not None else np.trapz(y, x))
    return area / span


def threshold_fraction(x: np.ndarray, y: np.ndarray, threshold: float, direction: str) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) == 0:
        return np.nan
    order = np.argsort(x)
    x, y = x[order], y[order]

    def crossed(v: float) -> bool:
        return v <= threshold if direction == "down" else v >= threshold

    if crossed(float(y[0])):
        return float(x[0])
    for i in range(1, len(x)):
        if crossed(float(y[i])):
            x0, x1 = float(x[i - 1]), float(x[i])
            y0, y1 = float(y[i - 1]), float(y[i])
            if abs(y1 - y0) <= EPS:
                return x1
            alpha = (threshold - y0) / (y1 - y0)
            return float(x0 + min(1.0, max(0.0, alpha)) * (x1 - x0))
    return np.nan


# =============================================================================
# Network representation
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
    link_lengths = work["length_m"].to_numpy(dtype=np.float64)

    rows: List[int] = []
    cols: List[int] = []
    groups: List[np.ndarray] = []
    for (u, v), grp in work.groupby(["u_idx", "v_idx"], sort=True):
        grp = grp.sort_values(["length_m", "link_id"])
        rows.append(int(u))
        cols.append(int(v))
        groups.append(grp["_pos"].to_numpy(dtype=np.int32))

    rows_a = np.asarray(rows, dtype=np.int32)
    cols_a = np.asarray(cols, dtype=np.int32)
    base_w = np.asarray([link_lengths[int(g[0])] for g in groups], dtype=np.float64)
    adj = csr_matrix((base_w, (rows_a, cols_a)), shape=(n, n), dtype=np.float64)

    return {
        "n": int(n),
        "node_ids": node_ids,
        "node_to_idx": node_to_idx,
        "link_ids": link_ids,
        "link_id_to_pos": {int(x): i for i, x in enumerate(link_ids)},
        "link_lengths": link_lengths,
        "pair_rows": rows_a,
        "pair_cols": cols_a,
        "pair_groups": tuple(groups),
        "base_weights": base_w,
        "base_adj": adj,
        "n_links": int(len(link_ids)),
    }


def lscc_share(adj: csr_matrix, n_total: int) -> float:
    n_comp, labels = connected_components(
        adj, directed=True, connection="strong", return_labels=True
    )
    if labels.size == 0:
        return 0.0
    return float(np.bincount(labels, minlength=n_comp).max() / n_total)


# =============================================================================
# Worker engine
# =============================================================================

def init_worker(
    n: int,
    pair_rows: np.ndarray,
    pair_cols: np.ndarray,
    pair_groups: Tuple[np.ndarray, ...],
    base_weights: np.ndarray,
    link_ids: np.ndarray,
    link_lengths: np.ndarray,
    node_ids: np.ndarray,
    sources: np.ndarray,
    od_origin_idx: np.ndarray,
    od_destination_idx: np.ndarray,
    od_demand: np.ndarray,
    od_baseline_distance: np.ndarray,
    base_eff: float,
    base_lscc: float,
    base_total_distance: float,
    disconnect_penalty_m: float,
) -> None:
    global _W
    _W = {
        "n": int(n),
        "pair_rows": np.asarray(pair_rows, dtype=np.int32),
        "pair_cols": np.asarray(pair_cols, dtype=np.int32),
        "pair_groups": tuple(np.asarray(x, dtype=np.int32) for x in pair_groups),
        "base_weights": np.asarray(base_weights, dtype=np.float64),
        "link_ids": np.asarray(link_ids, dtype=np.int64),
        "link_lengths": np.asarray(link_lengths, dtype=np.float64),
        "link_id_to_pos": {int(x): i for i, x in enumerate(link_ids)},
        "node_ids": np.asarray(node_ids, dtype=np.int64),
        "node_id_to_idx": {int(x): i for i, x in enumerate(node_ids)},
        "sources": np.asarray(sources, dtype=np.int32),
        "source_row": {int(x): i for i, x in enumerate(sources)},
        "od_origin_idx": np.asarray(od_origin_idx, dtype=np.int32),
        "od_destination_idx": np.asarray(od_destination_idx, dtype=np.int32),
        "od_demand": np.asarray(od_demand, dtype=np.float64),
        "od_baseline_distance": np.asarray(od_baseline_distance, dtype=np.float64),
        "base_eff": float(base_eff),
        "base_lscc": float(base_lscc),
        "base_total_distance": float(base_total_distance),
        "disconnect_penalty_m": float(disconnect_penalty_m),
        "n_links": int(len(link_ids)),
    }


def adj_after_links(removed: np.ndarray) -> csr_matrix:
    weights = np.empty(len(_W["pair_groups"]), dtype=np.float64)
    keep = np.zeros(len(_W["pair_groups"]), dtype=bool)
    for p, positions in enumerate(_W["pair_groups"]):
        chosen = -1
        for pos in positions:
            if not removed[int(pos)]:
                chosen = int(pos)
                break
        if chosen >= 0:
            weights[p] = _W["link_lengths"][chosen]
            keep[p] = True
    return csr_matrix(
        (weights[keep], (_W["pair_rows"][keep], _W["pair_cols"][keep])),
        shape=(_W["n"], _W["n"]),
        dtype=np.float64,
    )


def adj_after_nodes(removed: np.ndarray) -> csr_matrix:
    keep = (~removed[_W["pair_rows"]]) & (~removed[_W["pair_cols"]])
    return csr_matrix(
        (_W["base_weights"][keep], (_W["pair_rows"][keep], _W["pair_cols"][keep])),
        shape=(_W["n"], _W["n"]),
        dtype=np.float64,
    )


def state_performance(adj: csr_matrix) -> Dict[str, float]:
    dist = dijkstra(adj, directed=True, indices=_W["sources"], return_predecessors=False)
    if dist.ndim == 1:
        dist = dist.reshape(1, -1)
    src_rows = np.asarray(
        [_W["source_row"][int(x)] for x in _W["od_origin_idx"]], dtype=np.int32
    )
    ds = dist[src_rows, _W["od_destination_idx"]]
    q = _W["od_demand"]
    d0 = _W["od_baseline_distance"]
    reachable = np.isfinite(ds) & (ds > 0)
    q_total = float(q.sum(dtype=np.float64))
    q_reach = float(q[reachable].sum(dtype=np.float64))

    eff = (
        float(np.sum(q[reachable] / ds[reachable], dtype=np.float64) / q_total)
        if q_total > 0 else 0.0
    )
    eff_norm = eff / _W["base_eff"] if _W["base_eff"] > 0 else np.nan
    if abs(eff_norm - 1.0) < 1e-12:
        eff_norm = 1.0

    scc = lscc_share(adj, _W["n"])
    scc_norm = scc / _W["base_lscc"] if _W["base_lscc"] > 0 else np.nan
    if abs(scc_norm - 1.0) < 1e-12:
        scc_norm = 1.0

    penalized = np.where(reachable, ds, d0 + _W["disconnect_penalty_m"])
    total_distance = float(np.sum(q * penalized, dtype=np.float64))
    td_loss = (
        (total_distance - _W["base_total_distance"]) / _W["base_total_distance"]
        if _W["base_total_distance"] > 0 else np.nan
    )
    if abs(td_loss) < 1e-12:
        td_loss = 0.0

    return {
        "od_eff": float(eff),
        "od_eff_norm": float(eff_norm),
        "od_eff_loss": float(1.0 - eff_norm),
        "lscc": float(scc),
        "lscc_norm": float(scc_norm),
        "penalized_total_distance": total_distance,
        "td_loss": float(td_loss),
        "reachable_demand_share": float(q_reach / q_total) if q_total > 0 else np.nan,
    }


def evaluate_sequence(task: Tuple[str, str, int, int, Sequence[int], Sequence[float]]) -> List[Dict[str, Any]]:
    component, strategy, run, seed, sequence, fractions = task
    sequence = [int(x) for x in sequence]
    total = _W["n_links"] if component == "link" else _W["n"]
    counts = fraction_counts(total, fractions)
    count_targets: Dict[int, List[float]] = {}
    for f, k in zip(fractions, counts):
        count_targets.setdefault(int(k), []).append(float(f))

    if max(counts) > len(sequence):
        raise RuntimeError("Removal sequence shorter than requested removal count.")

    if component == "link":
        removed = np.zeros(_W["n_links"], dtype=bool)

        def remove(cid: int) -> None:
            removed[_W["link_id_to_pos"][int(cid)]] = True

        def make_adj() -> csr_matrix:
            return adj_after_links(removed)

        def fractions_lost() -> Tuple[float, float]:
            return float(removed.mean()), 0.0

    elif component == "node":
        removed = np.zeros(_W["n"], dtype=bool)

        def remove(cid: int) -> None:
            removed[_W["node_id_to_idx"][int(cid)]] = True

        def make_adj() -> csr_matrix:
            return adj_after_nodes(removed)

        def fractions_lost() -> Tuple[float, float]:
            link_lost = np.zeros(_W["n_links"], dtype=bool)
            for p, positions in enumerate(_W["pair_groups"]):
                if removed[_W["pair_rows"][p]] or removed[_W["pair_cols"][p]]:
                    link_lost[positions] = True
            return float(link_lost.mean()), float(removed.mean())

    else:
        raise ValueError(component)

    rows: List[Dict[str, Any]] = []
    removed_count = 0
    for k in sorted(count_targets):
        while removed_count < k:
            remove(sequence[removed_count])
            removed_count += 1
        metrics = state_performance(make_adj())
        edge_frac, node_frac = fractions_lost()
        for target_f in count_targets[k]:
            row = {
                "component": component,
                "strategy": strategy,
                "run": int(run),
                "seed": int(seed),
                "target_fraction": float(target_f),
                "fraction": float(k / total),
                "n_removed": int(k),
                "edge_fraction_lost": edge_frac,
                "node_fraction_lost": node_frac,
            }
            row.update(metrics)
            rows.append(row)
    return rows


# =============================================================================
# Sequences / aggregation
# =============================================================================

def build_sequences(
    node_metric: pd.DataFrame,
    link_metric: pd.DataFrame,
    random_runs: int,
) -> Tuple[List[Tuple[str, str, int, int, np.ndarray]], pd.DataFrame]:
    require_columns(node_metric, ["node_id"] + list(TARGET_METRICS.values()), "node_metric.csv")
    require_columns(link_metric, ["link_id"] + list(TARGET_METRICS.values()), "link_metric.csv")

    sequences: List[Tuple[str, str, int, int, np.ndarray]] = []
    seq_rows: List[Dict[str, Any]] = []

    for component, df, id_col in (
        ("node", node_metric, "node_id"),
        ("link", link_metric, "link_id"),
    ):
        ids = pd.to_numeric(df[id_col], errors="raise").astype(np.int64).to_numpy()
        for strategy, score_col in TARGET_METRICS.items():
            scores = pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=float)
            ordered, ordered_scores = stable_descending_order(ids, scores)
            sequences.append((component, strategy, 0, MASTER_SEED, ordered))
            for order, (cid, score) in enumerate(zip(ordered, ordered_scores), start=1):
                seq_rows.append({
                    "component": component,
                    "strategy": strategy,
                    "run": 0,
                    "seed": MASTER_SEED,
                    "order": int(order),
                    "component_id": int(cid),
                    "score": float(score) if np.isfinite(score) else score,
                    "score_metric": score_col,
                })

        for run in range(1, int(random_runs) + 1):
            offset = 100000 if component == "node" else 200000
            seed = int(MASTER_SEED + offset + run)
            rng = np.random.default_rng(seed)
            ordered = rng.permutation(ids).astype(np.int64)
            sequences.append((component, "random", run, seed, ordered))
            for order, cid in enumerate(ordered, start=1):
                seq_rows.append({
                    "component": component,
                    "strategy": "random",
                    "run": int(run),
                    "seed": seed,
                    "order": int(order),
                    "component_id": int(cid),
                    "score": np.nan,
                    "score_metric": "random",
                })

    return sequences, pd.DataFrame(seq_rows)


def aggregate_curves(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["component", "strategy", "target_fraction", "fraction", "n_removed"]
    metrics = ["od_eff_norm", "od_eff_loss", "lscc_norm", "td_loss", "reachable_demand_share"]
    rows: List[Dict[str, Any]] = []
    for key, grp in raw.groupby(keys, sort=True):
        component, strategy, target_f, fraction, n_removed = key
        row: Dict[str, Any] = {
            "component": component,
            "strategy": strategy,
            "target_fraction": float(target_f),
            "fraction": float(fraction),
            "n_removed": int(n_removed),
            "runs": int(grp["run"].nunique()),
            "edge_fraction_lost": float(grp["edge_fraction_lost"].mean()),
            "node_fraction_lost": float(grp["node_fraction_lost"].mean()),
        }
        for metric in metrics:
            x = pd.to_numeric(grp[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(x) == 0:
                row[metric] = row[metric + "_std"] = np.nan
                row[metric + "_p025"] = row[metric + "_p975"] = np.nan
            else:
                row[metric] = float(np.mean(x))
                row[metric + "_std"] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
                row[metric + "_p025"] = float(np.quantile(x, 0.025))
                row[metric + "_p975"] = float(np.quantile(x, 0.975))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["component", "strategy", "fraction"]).reset_index(drop=True)


def build_summary(raw: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (component, strategy), grp in curve.groupby(["component", "strategy"], sort=True):
        grp = grp.sort_values("fraction")
        x = grp["fraction"].to_numpy(dtype=float)

        for metric in ("od_eff_norm", "lscc_norm", "reachable_demand_share"):
            y = grp[metric].to_numpy(dtype=float)
            rows.append({
                "component": component,
                "strategy": strategy,
                "metric": metric,
                "auc": auc_normalized(x, y),
                "f90": threshold_fraction(x, y, 0.90, "down"),
                "f80": threshold_fraction(x, y, 0.80, "down"),
                "f50": threshold_fraction(x, y, 0.50, "down"),
            })

        y = grp["td_loss"].to_numpy(dtype=float)
        rows.append({
            "component": component,
            "strategy": strategy,
            "metric": "td_loss",
            "auc": auc_normalized(x, y),
            "f90": threshold_fraction(x, y, 0.10, "up"),
            "f80": threshold_fraction(x, y, 0.25, "up"),
            "f50": threshold_fraction(x, y, 0.50, "up"),
        })
    return pd.DataFrame(rows)


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--random-runs", type=int, default=DEFAULT_RANDOM_RUNS)
    parser.add_argument("--fractions", type=parse_fractions, default=DEFAULT_FRACTIONS)
    parser.add_argument("--component", choices=("both", "node", "link"), default="both")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    workers = max(1, int(args.workers))
    random_runs = max(1, int(args.random_runs))
    fractions = tuple(float(x) for x in args.fractions)
    if args.test:
        random_runs = 2
        fractions = (0.0, 0.01, 0.02)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        NODE_PATH, LINK_PATH, NODE_METRIC_PATH, LINK_METRIC_PATH,
        METRIC_META_PATH, OD_ACTIVE_PATH,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing upstream files: %s" % missing)

    nodes, _ = read_csv_auto(NODE_PATH)
    links, _ = read_csv_auto(LINK_PATH)
    node_metric, _ = read_csv_auto(NODE_METRIC_PATH)
    link_metric, _ = read_csv_auto(LINK_METRIC_PATH)
    od, _ = read_csv_auto(OD_ACTIVE_PATH)
    meta = json.loads(METRIC_META_PATH.read_text(encoding="utf-8"))

    require_columns(nodes, ["node_id"], "node.csv")
    require_columns(links, ["link_id", "from_node", "to_node", "length_m"], "link.csv")
    require_columns(
        od,
        ["origin_idx", "destination_idx", "demand", "baseline_distance_m"],
        "od_active.csv",
    )

    network = prepare_network(nodes, links)
    baseline = meta["baseline"]
    base_eff = float(baseline["od_efficiency"])
    base_lscc = float(baseline["lscc_share"])
    base_total_distance = float(baseline["total_distance_m_trip"])
    disconnect_penalty_m = float(baseline["disconnect_penalty_m"])

    sources = np.sort(pd.to_numeric(od["origin_idx"], errors="raise").astype(np.int32).unique())
    initargs = (
        network["n"], network["pair_rows"], network["pair_cols"],
        network["pair_groups"], network["base_weights"], network["link_ids"],
        network["link_lengths"], network["node_ids"], sources,
        pd.to_numeric(od["origin_idx"], errors="raise").to_numpy(dtype=np.int32),
        pd.to_numeric(od["destination_idx"], errors="raise").to_numpy(dtype=np.int32),
        pd.to_numeric(od["demand"], errors="raise").to_numpy(dtype=np.float64),
        pd.to_numeric(od["baseline_distance_m"], errors="raise").to_numpy(dtype=np.float64),
        base_eff, base_lscc, base_total_distance, disconnect_penalty_m,
    )

    sequences, sequence_df = build_sequences(node_metric, link_metric, random_runs)
    if args.component != "both":
        sequences = [x for x in sequences if x[0] == args.component]
        sequence_df = sequence_df[sequence_df["component"] == args.component].copy()
    tasks = [(c, s, r, seed, seq, fractions) for c, s, r, seed, seq in sequences]

    print(
        "Evaluating %d progressive attack sequences (%d targeted metrics + random), workers=%d..."
        % (len(tasks), len(TARGET_METRICS), workers)
    )
    rows: List[Dict[str, Any]] = []
    if workers <= 1:
        init_worker(*initargs)
        iterator = tasks
        if tqdm is not None:
            iterator = tqdm(iterator, desc="removal sequences", unit="sequence")
        for task in iterator:
            rows.extend(evaluate_sequence(task))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=initargs) as ex:
            futures = [ex.submit(evaluate_sequence, task) for task in tasks]
            iterator = as_completed(futures)
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(futures), desc="removal sequences", unit="sequence")
            for fut in iterator:
                rows.extend(fut.result())

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("No removal results were produced.")
    raw = raw.sort_values(["component", "strategy", "run", "fraction"]).reset_index(drop=True)

    baseline_rows = raw[raw["n_removed"] == 0]
    if not np.allclose(baseline_rows["od_eff_norm"], 1.0, atol=1e-8, equal_nan=False):
        raise RuntimeError("Removal baseline OD efficiency is inconsistent with graph_metric.py.")
    if not np.allclose(baseline_rows["lscc_norm"], 1.0, atol=1e-8, equal_nan=False):
        raise RuntimeError("Removal baseline LSCC is inconsistent with graph_metric.py.")
    if not np.allclose(baseline_rows["td_loss"], 0.0, atol=1e-8, equal_nan=False):
        raise RuntimeError("Removal baseline total-distance loss is not zero.")

    curve = aggregate_curves(raw)
    summary = build_summary(raw, curve)
    raw.to_csv(RAW_OUT, index=False, encoding="utf-8-sig")
    curve.to_csv(CURVE_OUT, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")
    sequence_df.to_csv(SEQUENCE_OUT, index=False, encoding="utf-8-sig")

    runtime = time.perf_counter() - started
    out_meta = {
        "method": {
            "routing_cost": "length_m",
            "targeted_rankings": TARGET_METRICS,
            "random_runs": int(random_runs),
            "fractions": list(fractions),
            "static_targeted_ranking": True,
            "odbc_as_performance_curve": False,
            "odbc_curve_reason": (
                "ODBC is a component ranking/load centrality; its network-wide sum "
                "changes with path hop count and rerouting and is not treated as a "
                "system-performance measure."
            ),
        },
        "baseline": {
            "od_efficiency": base_eff,
            "lscc_share": base_lscc,
            "total_distance_m_trip": base_total_distance,
            "disconnect_penalty_m": disconnect_penalty_m,
            "active_connector_demand": float(baseline["active_connector_demand"]),
        },
        "performance_fields": [
            "od_eff_norm", "lscc_norm", "td_loss", "reachable_demand_share"
        ],
        "parallel": {"workers": int(workers)},
        "runtime_seconds": float(runtime),
        "outputs": {
            "raw": str(RAW_OUT), "curve": str(CURVE_OUT),
            "summary": str(SUMMARY_OUT), "sequence": str(SEQUENCE_OUT),
        },
        "seed": MASTER_SEED,
    }
    META_OUT.write_text(json.dumps(out_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Completed removal analysis in %.1f s" % runtime)
    print("Saved: %s" % RESULT_DIR)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)