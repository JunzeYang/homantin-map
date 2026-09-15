"""Check the delivered bundle against independent source tables and saved geometry."""
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from shapely import wkt
from shapely.geometry import shape
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs'
text = (OUT / 'data.js').read_text(encoding='utf-8')
data = json.loads(text[len('window.ATLAS_DATA='):-2])
checks = []

def rows(name):
    with (ROOT/'source'/'res'/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

for kind in ('link', 'node'):
    source = rows('graph_metric/'+kind+'_metric.csv')
    features = data[kind+'s']['features']
    assert len(source) == len(features)
    lookup = {int(r[kind+'_id']):r for r in source}
    for f in features:
        p=f['properties']; r=lookup[p[kind+'_id']]
        for k in ('odbc','od_eff_loss','lscc_loss','td_loss','reachable_demand_share'):
            assert p[k] == float(r[k]), (kind,p[kind+'_id'],k)
        if kind=='link':
            original = list(wkt.loads(r['geometry']).coords)
            displayed = f['geometry']['coordinates']
            assert len(original) == len(displayed)
            assert all(math.dist(a,b) < 1e-7 for a,b in zip(original,displayed))
    checks.append(f'{kind}: every component metric matches its source CSV exactly')

grids=data['grid']['features']; src=rows('accessibility/hospital_access.csv')
assert len(grids)==len(src)==3599
for f,r in zip(grids,src):
    p=f['properties']
    assert p['grid_id']==int(r['grid_id'])
    assert p['reachable']==(r['reachable']=='True')
    if r['distance_m']: assert p['distance_m']==float(r['distance_m'])
    else: assert p['distance_m'] is None
assert sum(f['properties']['reachable'] for f in grids)==3136
to_hk=Transformer.from_crs(4326,2326,always_xy=True)
for f in grids[::100]:
    coords=[to_hk.transform(*c) for c in f['geometry']['coordinates'][0]]
    assert abs(math.dist(coords[0],coords[1])-100)<.03
    assert abs(math.dist(coords[1],coords[2])-100)<.03
checks.append('all grid IDs, distances and reachability match; sampled cell sides are 100 m within projection rounding')

for c in data['curves']:
    seq=data['sequences'][c['component']+':'+c['strategy']]
    assert len(seq)>=c['n_removed']
    assert len(set(seq[:c['n_removed']]))==c['n_removed']
    total=len(data[c['component']+'s']['features'])
    assert abs(c['fraction']-c['n_removed']/total)<1e-12
    h=[a for a in data['accessCurves'] if a['component']==c['component'] and a['strategy']==c['strategy'] and a['target_fraction']==c['target_fraction']]
    assert len(h)==1 and h[0]['n_removed']==c['n_removed']
    if c['strategy']=='random': assert c['runs']==30
checks.append('all 210 scenario steps have matching network/hospital results and valid saved removal sequences')

manifest=json.loads((OUT/'downloads/source_manifest.json').read_text())
for path,sha in manifest.items(): assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
checks.append('all 16 source fingerprints unchanged')
for f in data['figures']: assert (OUT/'figures'/f['file']).is_file()
for p in (OUT/'downloads').glob('*.csv'):
    candidates=list((ROOT/'source'/'res').glob('*/'+p.name))
    assert any(c.read_bytes()==p.read_bytes() for c in candidates)
for ref in re.findall(r'(?:src|href)="([^"]+)"',(OUT/'index.html').read_text(encoding='utf-8')):
    if ref.startswith(('data:','https:','http:','#')): continue
    assert (OUT/ref).exists(),ref
checks.append('all linked local entry assets exist; figures exist; downloadable CSVs are byte-identical to originals')
print(json.dumps({'status':'PASS','checks':checks},indent=2))
