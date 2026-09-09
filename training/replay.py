"""Versioned, checksummed, mmap-backed replay with vectorized batch decoding."""
import hashlib
import json
from pathlib import Path

import numpy as np

RECORD_BYTES = 560
DTYPE = np.dtype([("game_id", "<u8"), ("player", "<u8"), ("opponent", "<u8"), ("legal", "<u8"),
                  ("visits", "<u4", (65,)), ("priors", "<f4", (65,)), ("ply", "<u2"),
                  ("action", "u1"), ("stm", "u1"), ("outcome", "i1"), ("temperature", "u1"),
                  ("pass", "u1"), ("reserved", "u1")])
assert DTYPE.itemsize == RECORD_BYTES
SCHEMA = "islay-replay-v1"


def sha256(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def mix64(value):
    value &= (1 << 64) - 1
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def game_seed(master, game_id):
    return mix64(master ^ mix64(game_id + 0x9e3779b97f4a7c15))


def validation_game(game_id, seed):
    # Stable by game, before sampling/augmentation; resumed games cannot cross splits.
    return mix64(game_id ^ seed ^ 0x53504c4954) % 10 == 0


def manifest(root):
    data = json.loads((Path(root) / "manifest.json").read_text())
    if (data.get("schema") != SCHEMA or data.get("record_bytes") != RECORD_BYTES
            or data.get("rule") != "Othello" or data.get("encoding") != "relative-2x8x8-a1-v1"
            or data.get("policy") != "visits-tau1-prefix-tau0-lowest-v1"):
        raise ValueError("unsupported replay manifest")
    return data


def shards(root, meta, verify=True):
    root = Path(root)
    result, expected = [], 0
    identity = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    for directory in sorted(root.glob("shard-*")):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("invalid shard path")
        info = json.loads((directory / "index.json").read_text())
        path = directory / "samples.bin"
        if (info["start"] != expected or directory.name != f"shard-{expected:012d}"
                or info["dataset"] != identity or path.is_symlink() or not 0 < info["games"] <= 1024
                or not info["games"] <= info["samples"] <= info["games"] * 128
                or len(info["lengths"]) != info["games"] or sum(info["lengths"]) != info["samples"]
                or any(not 1 <= n <= 128 for n in info["lengths"])
                or path.stat().st_size != info["samples"] * RECORD_BYTES):
            raise ValueError("invalid shard index/length/identity")
        if verify and sha256(path) != info["sha256"]:
            raise ValueError("replay checksum mismatch")
        result.append((path, info))
        expected += info["games"]
    return result


def symmetry_maps():
    result = []
    for s in range(8):
        forward = []
        for a in range(64):
            x, y = a % 8, a // 8
            if s & 1: x = 7 - x
            if s & 2: y = 7 - y
            if s & 4: x, y = y, x
            forward.append(y * 8 + x)
        result.append(forward + [64])
    return np.asarray(result)


FORWARD = symmetry_maps()
INVERSE = np.argsort(FORWARD, axis=1)


def decode(records, symmetries=None):
    """Decode an entire gathered batch; bitboards stay packed on disk."""
    n = len(records)
    bits = np.stack((records["player"], records["opponent"]), axis=1).astype("<u8", copy=False)
    board = np.unpackbits(bits.view(np.uint8).reshape(n, 16), axis=1, bitorder="little").reshape(n, 2, 64)
    packed_legal = np.ascontiguousarray(records["legal"], dtype="<u8")
    legal = np.concatenate((np.unpackbits(packed_legal.view(np.uint8).reshape(n, 8), axis=1,
                                          bitorder="little"), records["pass"][:, None]), axis=1).astype(bool)
    visits = records["visits"].astype(np.float32)
    totals = visits.sum(axis=1, keepdims=True)
    if n and (np.any(totals <= 0) or np.any((records["temperature"] != 0) & (records["temperature"] != 1))):
        raise ValueError("invalid visits/temperature")
    policy = visits / totals
    cold = records["temperature"] == 0
    policy[cold] = 0
    policy[np.flatnonzero(cold), visits[cold].argmax(axis=1)] = 1
    priors = records["priors"].copy()
    actions = records["action"].astype(np.int64)
    if symmetries is not None:
        symmetries = np.asarray(symmetries)
        if symmetries.shape != (n,) or np.any(symmetries < 0) or np.any(symmetries > 7):
            raise ValueError("invalid batch symmetries")
        mapping = INVERSE[symmetries]
        board = np.take_along_axis(board, mapping[:, None, :64], axis=2)
        legal = np.take_along_axis(legal, mapping, axis=1)
        policy = np.take_along_axis(policy, mapping, axis=1)
        priors = np.take_along_axis(priors, mapping, axis=1)
        actions = FORWARD[symmetries, actions]
    return {"board": board.reshape(n, 2, 8, 8).astype(np.float32), "legal": legal, "policy": policy,
            "outcome": records["outcome"].astype(np.float32).reshape(n, 1), "priors": priors,
            "action": actions, "game_id": records["game_id"].copy(), "ply": records["ply"].copy()}


class Replay:
    def __init__(self, root):
        self.meta = manifest(root)
        self.shards = shards(root, self.meta)
        self.arrays = [np.memmap(path, dtype=DTYPE, mode="r") for path, _ in self.shards]
        self.ends = np.cumsum([len(a) for a in self.arrays], dtype=np.int64)
        train, validation = [], []
        offset = 0
        for _, info in self.shards:
            for game, length in enumerate(info["lengths"], info["start"]):
                dest = validation if validation_game(game, self.meta["seed"]) else train
                dest.extend(range(offset, offset + length))
                offset += length
        for array, (_, info) in zip(self.arrays, self.shards):
            cursor = 0
            for game, length in enumerate(info["lengths"], info["start"]):
                records = array[cursor:cursor + length]
                if (np.any(records["game_id"] != game) or np.any(records["ply"] != np.arange(length))
                        or np.any(records["reserved"] != 0) or np.any(np.abs(records["outcome"].astype(int)) > 1)):
                    raise ValueError("shard index/sample mismatch")
                cursor += length
        self.indices = {"train": np.asarray(train, dtype=np.int64), "validation": np.asarray(validation, dtype=np.int64)}

    def gather(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or (len(indices) and (not len(self.ends) or indices.min() < 0 or indices.max() >= self.ends[-1])):
            raise ValueError("replay index outside dataset")
        which = np.searchsorted(self.ends, indices, side="right")
        result = np.empty(len(indices), dtype=DTYPE)
        for shard in np.unique(which):
            mask = which == shard
            offset = 0 if shard == 0 else self.ends[shard - 1]
            result[mask] = self.arrays[shard][indices[mask] - offset]
        return result

    def sample(self, batch, rng, split="train", augment=True):
        if batch < 1 or batch > 65536 or split not in self.indices or not len(self.indices[split]):
            raise ValueError("invalid batch or empty replay split")
        indices = rng.choice(self.indices[split], size=batch, replace=True)
        return decode(self.gather(indices), rng.integers(0, 8, size=batch) if augment else None)

    def torch_batch(self, batch, rng, split="train", augment=True):
        import torch
        return {k: torch.from_numpy(v) for k, v in self.sample(batch, rng, split, augment).items()}
