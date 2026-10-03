"""Atomic files and content identities; file existence is never enough to resume."""
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def save_npz(path, arrays, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays, meta_json=np.array(json.dumps(meta, allow_nan=False)))
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def load_npz(path, expected_id=None):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        arrays = {k: data[k] for k in data.files if k != "meta_json"}
    if expected_id is not None and meta.get("result_id") != expected_id:
        raise ValueError(f"Result settings do not match: {path}. Use another output directory.")
    return arrays, meta

def result_id(study_id, spec):
    return digest_json({"study_id": study_id, "spec": spec})

def ensure_manifest(path, identity, extra):
    study_id = digest_json(identity)
    path = Path(path)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("study_id") != study_id:
            raise ValueError(f"Study settings, code, data, or model changed: {path}. "
                             "Choose a new output_dir; old results will not be reused.")
        return saved
    manifest = {"study_id": study_id, "identity": identity, **extra}
    atomic_json(path, manifest)
    return manifest
