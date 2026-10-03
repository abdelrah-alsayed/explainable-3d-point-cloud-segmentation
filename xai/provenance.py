"""Run provenance for artifacts that outlive the machine that made them.

The exact tables and KernelSHAP outputs produced on a borrowed GPU are
analysed for months afterwards on a different machine. Recording what
produced each one -- torch build, GPU, checkpoint identity, when -- costs
microseconds and is impossible to reconstruct later.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_commit() -> str:
    """Short commit hash, or a reason it isn't available. This tree is not
    always a git checkout, so absence is normal and must not raise."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "not-a-git-checkout"
    except (OSError, subprocess.SubprocessError):
        return "git-unavailable"


def checkpoint_sha256(log_dir) -> str:
    """Hash of the .pth actually loaded. The single strongest guard against
    silently explaining a different checkpoint than you think."""
    path = Path(log_dir) / "checkpoints" / "best_model.pth"
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_provenance(log_dir=None) -> dict:
    """JSON-safe dict to embed in every saved artifact."""
    import torch

    gpu = "cpu"
    if torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(0)
        except Exception:
            gpu = "cuda-unknown"

    prov = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu,
        "python_version": sys.version.split()[0],
        "hostname": platform.node(),
        "git_commit": _git_commit(),
    }
    if log_dir is not None:
        prov["checkpoint_sha256"] = checkpoint_sha256(log_dir)
    return prov
