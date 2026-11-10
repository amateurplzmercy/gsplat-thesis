"""Verify downloaded PyPI dependency wheels against their published digests."""

import hashlib
import json
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from email.parser import Parser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def inspect_wheel(path):
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))
        metadata = Parser().parsestr(archive.read(name).decode())
    url = f"https://pypi.org/pypi/{metadata['Name']}/{metadata['Version']}/json"
    with urllib.request.urlopen(url, timeout=40) as response:
        release = json.load(response)
    info = next(item for item in release['urls'] if item['filename'] == path.name)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    actual = digest.hexdigest()
    if actual != info['digests']['sha256']:
        raise RuntimeError(f"Published digest mismatch: {path.name}")
    return {'name': metadata['Name'], 'version': metadata['Version'],
            'filename': path.name, 'bytes': path.stat().st_size,
            'url': info['url'], 'sha256': actual, 'published_hash_verified': True}


if __name__ == '__main__':
    paths = sorted((ROOT / 'downloads' / 'wheels').glob('*.whl'))
    if not paths:
        raise RuntimeError('No dependency wheels found')
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(inspect_wheel, paths))
    (ROOT / 'reports' / 'dependency_wheel_manifest.json').write_text(
        json.dumps(records, indent=2) + '\n', encoding='utf-8')
    print(f'Verified {len(records)} dependency wheels against PyPI SHA256 digests.')
