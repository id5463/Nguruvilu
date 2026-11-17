# -*- coding: utf-8 -*-
"""Rebuild packs/pack-market/store + index.json from every pack directory.

Each pack directory is packed with `ngu pack` (the same code path an install
uses), unpacked, and re-zipped for the store — so what the market serves is
what the pack actually produces, not a hand-maintained copy that drifts.
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(ROOT, 'packs', 'pack-market', 'store')
INDEX = os.path.join(ROOT, 'packs', 'pack-market', 'index.json')
BASE = 'https://raw.githubusercontent.com/id5463/Nguruvilu/main/packs/pack-market/'
NGU = os.path.join(ROOT, 'target', 'release', 'ngu.exe')
SKIP = {'target', 'node_modules', '__pycache__', '.git', 'dist', '.venv'}

os.makedirs(STORE, exist_ok=True)
entries = []
packs_dir = os.path.join(ROOT, 'packs')
for name in sorted(os.listdir(packs_dir)):
    d = os.path.join(packs_dir, name)
    if not os.path.isdir(d):
        continue
    idx = os.path.join(d, 'ngu.index.json')
    if not os.path.isfile(idx):
        continue
    manifest = json.load(open(idx, encoding='utf-8'))
    if manifest.get('name') != name:
        print(f'SKIP {name}: manifest names {manifest.get("name")!r}')
        continue

    tmp = os.path.join(tempfile.gettempdir(), f'mk-{name}.ngupack')
    r = subprocess.run([NGU, 'pack', d, '--out', tmp], capture_output=True)
    if r.returncode != 0:
        print(f'SKIP {name}: pack failed {r.stderr.decode("utf-8","replace")[:90]}')
        continue

    staging = tempfile.mkdtemp()
    with zipfile.ZipFile(tmp) as z:
        z.extractall(staging)
    out = os.path.join(STORE, f'{name}-{manifest["versionId"]}.zip')
    if os.path.exists(out):
        os.remove(out)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for base, dirs, files in os.walk(staging):
            dirs[:] = [x for x in dirs if x not in SKIP]
            for f in files:
                full = os.path.join(base, f)
                z.write(full, os.path.relpath(full, staging))
    shutil.rmtree(staging, ignore_errors=True)
    os.remove(tmp)

    data = open(out, 'rb').read()
    entries.append({
        'name': name,
        'version': manifest['versionId'],
        'description': manifest.get('description', ''),
        'license': manifest.get('license', 'MIT'),
        'url': BASE + 'store/' + os.path.basename(out),
        'sha256': hashlib.sha256(data).hexdigest(),
        'size': len(data),
    })

json.dump({'version': 1, 'packs': entries},
          open(INDEX, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f'index: {len(entries)} packs')
for e in entries:
    print(f"  {e['name']:16} {e['version']:8} {e['size']:>8} B")
