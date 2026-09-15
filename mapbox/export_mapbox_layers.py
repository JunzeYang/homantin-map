"""Split the saved atlas data bundle into Mapbox Studio GeoJSON uploads."""
import json
import zipfile
from pathlib import Path

here = Path(__file__).resolve().parent
bundle = (here.parent / 'docs' / 'data.js').read_text(encoding='utf-8')
assert bundle.startswith('window.ATLAS_DATA=') and bundle.endswith(';\n')
data = json.loads(bundle[len('window.ATLAS_DATA='):-2])
names = {
    'links': 'critical_links.geojson',
    'nodes': 'critical_nodes.geojson',
    'grid': 'hospital_grid.geojson',
    'hospitals': 'hospitals.geojson',
    'zones': 'study_tpusb.geojson',
}
counts = {'links': 2849, 'nodes': 1733, 'grid': 3599, 'hospitals': 14, 'zones': 1200}
for key, filename in names.items():
    layer = data[key]
    assert layer['type'] == 'FeatureCollection' and len(layer['features']) == counts[key]
    if key == 'links': assert len({f['properties']['link_id'] for f in layer['features']}) == counts[key]
    if key == 'nodes': assert len({f['properties']['node_id'] for f in layer['features']}) == counts[key]
    (here / filename).write_text(json.dumps(layer, ensure_ascii=False, allow_nan=False, separators=(',', ':')), encoding='utf-8')
with zipfile.ZipFile(here.parent / 'mapbox_result_layers.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    z.write(here / 'README.md', 'README.md')
    for filename in names.values(): z.write(here / filename, filename)
print(json.dumps({'layers': counts, 'zip_bytes': (here.parent / 'mapbox_result_layers.zip').stat().st_size}))
