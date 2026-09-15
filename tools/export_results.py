"""Package saved outputs for the atlas. Does not import or execute analysis code."""
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from shapely import wkt
from shapely.geometry import mapping, shape
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs'
RES = ROOT / 'source' / 'res'
sources = {}

def record(path):
    sources[str(path.relative_to(ROOT)).replace('\\', '/')] = hashlib.sha256(path.read_bytes()).hexdigest()

def value(v):
    if v is None or v == '' or v.lower() in ('nan', 'inf', '-inf'): return None
    if v in ('True', 'False'): return v == 'True'
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except ValueError: return v

def rows(relative):
    path = RES / relative
    record(path)
    with path.open(encoding='utf-8-sig', newline='') as f:
        return [{k: value(v) for k, v in r.items()} for r in csv.DictReader(f)]

def meta(relative):
    path = RES / relative
    record(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))

def rounded(obj):
    if isinstance(obj, float): return round(obj, 7) if math.isfinite(obj) else None
    if isinstance(obj, (list, tuple)): return [rounded(x) for x in obj]
    if isinstance(obj, dict): return {k: rounded(v) for k, v in obj.items()}
    return obj

def feature(geom, props):
    return {'type': 'Feature', 'geometry': rounded(geom), 'properties': props}

def collection(features): return {'type': 'FeatureCollection', 'features': features}

OUT.mkdir(parents=True, exist_ok=True)
links = rows('graph_metric/link_metric.csv')
nodes = rows('graph_metric/node_metric.csv')
impact = {(r['component'], r['component_id']): r for r in rows('accessibility/access_impact.csv')}
access = {r['node_id']: r for r in rows('accessibility/node_access.csv')}
lf, nf = [], []
for kind, data in [('link', links), ('node', nodes)]:
    for r in data:
        cid = r[kind + '_id']
        r.update({k: v for k, v in impact[kind, cid].items() if k not in ('component', 'component_id', 'odbc', 'od_eff_loss', 'lscc_loss', 'td_loss')})
        if kind == 'link':
            geom = mapping(wkt.loads(r.pop('geometry')))
            lf.append(feature(geom, r))
        else:
            a = access[cid]
            r.update({k: a[k] for k in ('harmonic_access', 'access_rank', 'access_pct')})
            r['network_reachable_share'] = a['reachable_share']
            nf.append(feature({'type': 'Point', 'coordinates': [r['lon'], r['lat']]}, r))

to_wgs = Transformer.from_crs('EPSG:2326', 'EPSG:4326', always_xy=True)
gf = []
for r in rows('accessibility/hospital_access.csv'):
    x, y = r['x_hk80'], r['y_hk80']
    ring = [to_wgs.transform(x+dx, y+dy) for dx, dy in [(-50,-50),(50,-50),(50,50),(-50,50),(-50,-50)]]
    gf.append(feature({'type': 'Polygon', 'coordinates': [ring]}, r))
catchment = {r['hospital_id']: r for r in rows('accessibility/hospital_catchment.csv')}
hf = []
for r in rows('accessibility/hospital.csv'):
    r.update(catchment.get(r['hospital_id'], {}))
    hf.append(feature({'type': 'Point', 'coordinates': [r['longitude'], r['latitude']]}, r))

zpath = RES / 'od/study_tpusb.geojson'
record(zpath)
zones = json.loads(zpath.read_text(encoding='utf-8-sig'))
for f in zones['features']:
    f['geometry'] = rounded(mapping(shape(f['geometry']).simplify(0.000025, preserve_topology=True)))

sequence = {}
for r in rows('removal/removal_sequence.csv'):
    if r['run'] != (1 if r['strategy'] == 'random' else 0): continue
    key = r['component'] + ':' + r['strategy']
    sequence.setdefault(key, []).append((r['order'], r['component_id']))
sequence = {k: [v for _, v in sorted(a)] for k, a in sequence.items()}
gm = meta('graph_metric/metric_meta.json')
am = meta('accessibility/accessibility_meta.json')
om = meta('od/od_meta.json')
rm = meta('removal/removal_meta.json')
data = {
    'links': collection(lf), 'nodes': collection(nf), 'grid': collection(gf),
    'hospitals': collection(hf), 'zones': zones,
    'curves': rows('removal/removal_curve.csv'),
    'accessCurves': rows('accessibility/access_removal_curve.csv'),
    'curveSummary': rows('removal/removal_summary.csv'), 'sequences': sequence,
    'meta': {'graph': gm['graph'], 'baseline': gm['baseline'], 'metricDefinitions': gm['metrics'],
             'access': am['baseline_hospital_access'], 'study': am['study_domain'],
             'distanceModel': am['distance_model'], 'od': om['od'], 'connector': om['connector_od'],
             'method': rm['method'], 'snapshot': 'Supplied maplayer_codex results'},
}
assert len(lf) == gm['graph']['directed_links']
assert len(nf) == gm['graph']['nodes']
assert len(gf) == am['baseline_hospital_access']['origin_count']
assert sum(f['properties']['reachable'] for f in gf) == am['baseline_hospital_access']['reachable_count']
assert len({f['properties']['link_id'] for f in lf}) == len(lf)
assert len({f['properties']['node_id'] for f in nf}) == len(nf)
for kind, feats in [('link', lf), ('node', nf)]:
    valid = {f['properties'][kind+'_id'] for f in feats}
    for key, seq in sequence.items():
        if key.startswith(kind+':'): assert set(seq) <= valid

downloads = OUT / 'downloads'
downloads.mkdir(exist_ok=True)
for folder in ['graph_metric', 'accessibility', 'removal', 'fig_tab']:
    for p in (RES / folder).glob('*.csv'):
        if p.name.endswith('_raw.csv') or p.name == 'removal_sequence.csv': continue
        shutil.copy2(p, downloads / p.name)
figdir = OUT / 'figures'
figdir.mkdir(exist_ok=True)
figures = []
for p in sorted((RES / 'fig_tab').glob('Fig_*.jpg')):
    if p.stem.endswith('1'): continue
    shutil.copy2(p, figdir / p.name)
    figures.append({'file': p.name, 'title': p.stem.replace('_', ' ')})
data['figures'] = figures
(OUT / 'data.js').write_text('window.ATLAS_DATA=' + json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + ';\n', encoding='utf-8')
(OUT / 'downloads/source_manifest.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')
print(json.dumps({'nodes': len(nf), 'links': len(lf), 'grid': len(gf), 'hospitals': len(hf), 'zones': len(zones['features']), 'data_bytes': (OUT/'data.js').stat().st_size, 'source_files': len(sources)}))
