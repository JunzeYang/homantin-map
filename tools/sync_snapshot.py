"""Copy supplied data, code, and saved results into this website repository."""
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[1]
ORIGINAL = REPO.parent

assert (ORIGINAL/'web'/'dist'/'data.js').is_file()
assert (ORIGINAL/'res'/'graph_metric'/'link_metric.csv').is_file()

shutil.copytree(ORIGINAL/'web'/'dist', REPO/'docs', dirs_exist_ok=True)
shutil.copytree(ORIGINAL/'code', REPO/'source'/'code', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.idea'), dirs_exist_ok=True)
shutil.copytree(ORIGINAL/'res', REPO/'source'/'res', ignore=shutil.ignore_patterns('cache', '__pycache__', '*.pyc'), dirs_exist_ok=True)
shutil.copytree(ORIGINAL/'shp', REPO/'source'/'shp', dirs_exist_ok=True)
for filename in ('node.csv', 'link.csv'):
    shutil.copy2(ORIGINAL/filename, REPO/'source'/filename)
shutil.copytree(ORIGINAL/'mapbox_layers', REPO/'mapbox', ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
shutil.copy2(ORIGINAL/'mapbox_result_layers.zip', REPO/'mapbox_result_layers.zip')
(REPO/'mapbox'/'export_mapbox_layers.py').write_text(
    (REPO/'mapbox'/'export_mapbox_layers.py').read_text(encoding='utf-8').replace("here.parent / 'web' / 'dist' / 'data.js'", "here.parent / 'docs' / 'data.js'"),
    encoding='utf-8',
)

for filename in ('export_results.py', 'validate_results.py'):
    src=(ORIGINAL/'web'/'scripts'/filename).read_text(encoding='utf-8')
    src=src.replace("ROOT = Path(__file__).resolve().parents[2]", "ROOT = Path(__file__).resolve().parents[1]")
    src=src.replace("ROOT / 'web' / 'dist'", "ROOT / 'docs'")
    src=src.replace("ROOT / 'res'", "ROOT / 'source' / 'res'")
    src=src.replace("ROOT/'res'", "ROOT/'source'/'res'")
    (REPO/'tools'/filename).write_text(src,encoding='utf-8')

(REPO/'docs'/'.nojekyll').touch()
print('Copied website, input tables, analysis code, saved results, spatial files and Mapbox layers')
