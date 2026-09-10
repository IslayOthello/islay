"""Atomic training checkpoints; only committed directories are resumable."""
import json
import math
import os
from pathlib import Path
import tempfile

import torch

from replay import sha256
from selfplay import sync_directory, write_json


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    return True


def save(root, state, metrics, max_mib=1024):
    root = Path(root)
    destination = root / f"step-{state['updates']:09d}"
    if destination.exists():
        raise ValueError("refusing to overwrite a committed checkpoint")
    reserve = 32 * 1024 * 1024
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if used + reserve > max_mib * 1024 * 1024:
        raise ValueError("checkpoint byte cap reached; nothing evicted")
    state = cpu_tree(state)
    if not finite_tree(state["state_dict"]) or not finite_tree(state["optimizer"]):
        raise ValueError("non-finite weights/optimizer cannot be checkpointed")
    with tempfile.TemporaryDirectory(prefix=".pending-", dir=root) as temporary:
        staging = Path(temporary)
        path = staging / "state.pt"
        with path.open("xb") as output:
            torch.save(state, output)
            output.flush()
            os.fsync(output.fileno())
        write_json(staging / "metrics.json", metrics)
        write_json(staging / "index.json", {"schema": "islay-train-commit-v1", "updates": state["updates"],
                                            "sha256": sha256(path), "metrics_sha256": sha256(staging / "metrics.json")})
        if sum(p.stat().st_size for p in staging.iterdir()) > reserve:
            raise ValueError("checkpoint exceeds reserved size")
        sync_directory(staging)
        os.rename(staging, destination)
        sync_directory(root)
    # The directory, not this convenience pointer, is the commit record.
    with tempfile.TemporaryDirectory(prefix=".latest-", dir=root) as temporary:
        path = Path(temporary) / "latest.json"
        write_json(path, {"directory": destination.name})
        os.replace(path, root / "latest.json")
        sync_directory(root)
    return destination / "state.pt"


def latest(root):
    entries = sorted(Path(root).glob("step-*"), key=lambda path: int(path.name.removeprefix("step-")))
    if not entries:
        raise ValueError("no committed checkpoint; incomplete initialization is not resumable")
    previous = -1
    for directory in entries:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("invalid checkpoint path")
        index = json.loads((directory / "index.json").read_text())
        if (index.get("schema") != "islay-train-commit-v1" or type(index.get("updates")) is not int
                or index["updates"] <= previous or directory.name != f"step-{index['updates']:09d}"):
            raise ValueError("invalid checkpoint index/order")
        path = directory / "state.pt"
        if path.is_symlink() or path.stat().st_size > 32 * 1024 * 1024 or sha256(path) != index["sha256"]:
            raise ValueError("checkpoint checksum mismatch")
        if sha256(directory / "metrics.json") != index["metrics_sha256"]:
            raise ValueError("checkpoint metrics checksum mismatch")
        previous = index["updates"]
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("updates") != previous:
        raise ValueError("checkpoint payload/index mismatch")
    return state, path
