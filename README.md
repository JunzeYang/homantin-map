# Ho Man Tin road resilience atlas

[Open the interactive atlas](https://junzeyang.github.io/homantin-map/)

This repository publishes a static map of the supplied road-network resilience and hospital-accessibility results. The site reads saved outputs; it does not execute the network-analysis scripts in visitors' browsers.

The first screen provides a map with switchable layers and result details. Explore directed link and node criticality, 100 m hospital accessibility, and saved 0–40% removal scenarios. Network and hospital performance curves, original research figures, and downloadable result tables are included.

## Repository layout

| Path | Content |
| --- | --- |
| [`docs/`](docs/) | Complete GitHub Pages website and bundled result data |
| [`source/`](source/) | Supplied network tables, analysis code, spatial files, saved CSV/JSON outputs and research figures |
| [`tools/`](tools/) | Export and validate the static website from saved `source/res` results |
| [`mapbox/`](mapbox/) | WGS84 GeoJSON layers for Mapbox Studio |
| [`mapbox_result_layers.zip`](mapbox_result_layers.zip) | Ready-to-upload Mapbox layer package |

GitHub Pages is configured to deploy from `main` → `/docs`. The empty [`docs/.nojekyll`](docs/.nojekyll) file prevents Jekyll from modifying this plain static site. To update the website after changing **consistent** saved outputs, run `tools/export_results.py` with Shapely and pyproj, then `tools/validate_results.py`, commit `docs/`, and push `main`. These tools package and check outputs; they do not rerun the scientific analysis. On the original machine, `D:/Anaconda3/envs/py39_osm/python.exe` has the required Python packages.

## Interpretation

This snapshot covers the full supplied road-network envelope, not the administrative Ho Man Tin boundary. Road-proxy OD flow, connector demand, and baseline-reachable active demand have different denominators. Hospital distance is directed shortest-path **road distance**, excludes the origin's snap distance, and does not measure travel or ambulance time. Hospital reachability is an equal-weight share of 3,599 grid cells rather than a population share. The nearest-hospital set includes facilities without A&E.

Targeted disruption maps replay saved static rankings. A random map displays saved run **1** while random curves and summary values represent 30 runs. A falling mean hospital distance can arise when more origins become unreachable. Opposite directions and parallel road links can overlap visually; separate link IDs remain in the results.

The supplied large binary caches and Python bytecode are omitted from the public repository. The full original package remains in the local `maplayer_codex` folder. The packaged CSV and geometry values have been checked against the supplied snapshot; scientific model validity has not been re-evaluated. See [`tools/validate_results.py`](tools/validate_results.py) and the saved [source fingerprints](docs/downloads/source_manifest.json).
