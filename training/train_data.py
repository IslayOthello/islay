"""Bounded, immutable replay window; game splits precede sampling and D4 augmentation."""
import hashlib
import json
from pathlib import Path

import numpy as np

from replay import RECORD_BYTES, Replay, decode, validation_game
from selfplay import verify_shard


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class TrainingReplay:
    def __init__(self, roots, verifier, max_games=10000, max_mib=256):
        if not roots or max_games < 1 or max_mib < 1:
            raise ValueError("invalid replay window")
        # Preflight before checksums/maps/index allocations; no unbounded source scan.
        total = sum(p.stat().st_size for root in roots for p in Path(root).glob("shard-*/samples.bin"))
        if total > max_mib * 1024 * 1024:
            raise ValueError("replay inputs exceed max_replay_mib")
        self.sources = [Replay(root) for root in roots]
        total = sum(path.stat().st_size for source in self.sources for path, _ in source.shards)
        if total > max_mib * 1024 * 1024:
            raise ValueError("replay grew beyond max_replay_mib while opening snapshot")
        descriptions, games, seen, offset = [], [], set(), 0
        for i, source in enumerate(self.sources):
            identity = digest(source.meta)
            if identity in seen:
                raise ValueError("duplicate replay dataset identity")
            seen.add(identity)
            descriptions.append({"manifest": source.meta, "shards": [info for _, info in source.shards]})
            for path, info in source.shards:
                verify_shard(Path(verifier).resolve(), path, source.meta, info["start"], info["games"])
                for game, length in enumerate(info["lengths"], info["start"]):
                    games.append((i, game, offset, length, validation_game(game, source.meta["seed"])))
                    offset += length
        self.ends = np.cumsum([int(s.ends[-1]) if len(s.ends) else 0 for s in self.sources], dtype=np.int64)
        selected = games[-max_games:]
        if not selected:
            raise ValueError("empty replay window")
        self.indices = {}
        for split, validation in (("train", False), ("validation", True)):
            pieces = [np.arange(offset, offset + n, dtype=np.int64) for _, _, offset, n, val in selected if val == validation]
            self.indices[split] = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
            if not len(self.indices[split]):
                raise ValueError("replay window has empty " + split + " split; generate more games")
        self.snapshot = {"schema": "islay-training-data-v1", "sources_oldest_first": descriptions,
                         "selected_games": [[i, game] for i, game, _, _, _ in selected],
                         "max_games": max_games, "record_bytes": RECORD_BYTES}
        self.identity = digest(self.snapshot)
        self.statistics = {"games": len(selected), "samples": {k: len(v) for k, v in self.indices.items()},
                           "bytes": total, "phases": {}}
        # Labels/distributions come from the frozen replay only, never a heuristic teacher.
        for split, indices in self.indices.items():
            phases = {name: {"samples": 0, "loss": 0, "draw": 0, "win": 0} for name in ("opening", "middle", "end")}
            for start in range(0, len(indices), 8192):
                records = self.gather(indices[start:start + 8192])
                occupied = np.bitwise_count(records["player"] | records["opponent"])
                for name, mask in (("opening", occupied < 24), ("middle", (occupied >= 24) & (occupied < 44)),
                                   ("end", occupied >= 44)):
                    phases[name]["samples"] += int(mask.sum())
                    for result, value in (("loss", -1), ("draw", 0), ("win", 1)):
                        phases[name][result] += int(((records["outcome"] == value) & mask).sum())
            self.statistics["phases"][split] = phases

    def gather(self, indices):
        from replay import DTYPE
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or (len(indices) and (indices.min() < 0 or indices.max() >= self.ends[-1])):
            raise ValueError("training replay index outside snapshot")
        owners = np.searchsorted(self.ends, indices, side="right")
        result = np.empty(len(indices), dtype=DTYPE)
        for owner in np.unique(owners):
            mask = owners == owner
            base = self.ends[owner - 1] if owner else 0
            result[mask] = self.sources[owner].gather(indices[mask] - base)
        return result

    def sample(self, batch, rng, augment=True):
        indices = rng.choice(self.indices["train"], size=batch, replace=True)
        return decode(self.gather(indices), rng.integers(0, 8, size=batch) if augment else None)
