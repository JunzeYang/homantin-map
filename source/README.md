# Supplied source snapshot

This directory holds the user's supplied `maplayer_codex` code, network node/link tables, shapefiles, and saved analysis outputs. They are included as provenance and for offline re-export. The GitHub Pages website uses only `docs/` and does not execute these scripts.

Generated binary caches under `res/**/cache/` and Python `__pycache__` are omitted because they are large, machine-specific intermediate files and are not needed to display or re-export the saved website results. The supplied result CSVs, JSON metadata, shapefiles, research figures and tables remain available here. The original full package remains in the local project.

Run `tools/export_results.py` with Shapely and pyproj after changing the source result tables, then `tools/validate_results.py` before updating the website. The exporter reads this snapshot and writes the static `docs/` bundle; it does not import or run the original analysis scripts.
