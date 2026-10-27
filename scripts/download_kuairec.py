"""Download the official KuaiRec 2.0 archive and verify its published checksum."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

URL = 'https://zenodo.org/records/18164998/files/KuaiRec.zip'
EXPECTED_MD5 = '261550d472c48eff4990fb13c0e5bcf7'
EXPECTED_SIZE = 431964858


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('data/raw/kuairec_official'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir/'KuaiRec.zip'
    if not archive.exists():
        partial = archive.with_suffix('.zip.part')
        with urllib.request.urlopen(URL, timeout=120) as source, partial.open('wb') as target:
            shutil.copyfileobj(source, target)
        partial.rename(archive)
    with archive.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'md5').hexdigest()
    if archive.stat().st_size != EXPECTED_SIZE or checksum != EXPECTED_MD5:
        raise RuntimeError('Archive checksum mismatch. Remove the incomplete file and retry.')
    extracted = args.output_dir/'extracted'
    extracted.mkdir(exist_ok=True)
    records = []
    wanted = {'big_matrix.csv', 'item_categories.csv', 'LICENSE'}
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            name = Path(info.filename).name
            if name in wanted:
                target = extracted/name
                with bundle.open(info) as source, target.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
                with target.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                records.append(dict(name=name, bytes=target.stat().st_size, sha256=digest))
    if {row['name'] for row in records} != wanted:
        raise RuntimeError('Archive is missing required input files.')
    manifest = dict(source=URL, record='https://zenodo.org/records/18164998',
                    official_repository='https://github.com/chongminggao/KuaiRec',
                    checked_at_utc=datetime.now(timezone.utc).isoformat(), archive_md5=checksum,
                    archive_bytes=archive.stat().st_size, files=records)
    (args.output_dir/'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Verified KuaiRec inputs: {extracted}')


if __name__ == '__main__':
    main()
