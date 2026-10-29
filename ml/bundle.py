"""One versioned manifest binds inference, threshold, metrics and chart arrays."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LEGACY_DIRECTORY=ROOT/'artifacts/synthetic_ml'
ACTIVE=ROOT/'artifacts/ml_active.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def active_model_directory():
    if not ACTIVE.exists():
        return LEGACY_DIRECTORY
    pointer=json.loads(ACTIVE.read_text())
    #directory=(ROOT/pointer['directory']).resolve()
    directory=(ROOT / pointer['directory'].replace("\\", "/")).resolve()
    if not directory.is_relative_to((ROOT/'artifacts').resolve()):
        raise ValueError('Active model directory must stay inside project artifacts.')
    if digest(directory/'bundle_manifest.json') != pointer['manifest_sha256']:
        raise ValueError('Active model manifest hash mismatch.')
    return directory


def validate_bundle(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'bundle_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        path=directory/name
        if not path.is_file():
            raise ValueError('Required final-model artifact missing: '+name)
        if digest(path)!=expected:
            raise ValueError('Final-model artifact integrity mismatch: '+name)
    return manifest
