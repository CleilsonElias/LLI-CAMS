"""Verify included file hashes. Integrity check, not an authorship certificate."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
failures = []
for relative, expected in manifest['files'].items():
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        failures.append(relative + ': missing or invalid path')
    elif hashlib.sha256(target.read_bytes()).hexdigest() != expected['sha256']:
        failures.append(relative + ': hash mismatch')
if failures:
    print('\n'.join(failures), file=sys.stderr)
    raise SystemExit(1)
print(f"Verified {len(manifest['files'])} files for LLI-CAMS {manifest['version']}.")
