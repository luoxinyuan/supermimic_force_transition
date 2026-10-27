"""Download only the 40 G1 clips at a pinned HF revision; verify git blob hashes."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
BASE = 'https://huggingface.co'
REPO = 'lvhaidong/LAFAN1_Retargeting_Dataset'
def get(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()

def main():
    dest = ROOT / 'data/raw/lafan1_g1'
    dest.mkdir(parents=True, exist_ok=True)
    manifest = dest / 'manifest.json'
    revision = json.loads(manifest.read_text())['revision'] if manifest.exists() else json.loads(get(f'{BASE}/api/datasets/{REPO}'))['sha']
    entries = json.loads(get(f'{BASE}/api/datasets/{REPO}/tree/{revision}/g1'))
    clips = [e for e in entries if e['path'].endswith('.csv')]
    assert len(clips) == 40, f'Expected 40 clips, got {len(clips)}'
    def download(entry):
        path = dest / entry['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes() if path.exists() else get(f'{BASE}/datasets/{REPO}/resolve/{revision}/{entry["path"]}')
        assert len(data) == entry['size']
        assert hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == entry['oid'], path
        path.write_bytes(data)
        return {**entry, 'sha256': hashlib.sha256(data).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(download, clips))
    for name in ('README.md', 'LICENSE'):
        (dest / name).write_bytes(get(f'{BASE}/datasets/{REPO}/resolve/{revision}/{name}'))
    manifest.write_text(json.dumps({'repo': REPO, 'revision': revision, 'clips': records}, indent=2) + '\n')
    print(f'Verified {len(records)} clips at {revision}', flush=True)

if __name__ == '__main__':
    main()
