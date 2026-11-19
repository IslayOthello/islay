"""Bounded, immutable replay window; game splits precede sampling and D4 augmentation."""
import hashlib
import json
from pathlib import Path

import numpy as np

from replay import DTYPE, INVERSE, RECORD_BYTES, Replay, decode, validation_game


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class TrainingReplay:
    def __init__(self, roots, verifier, max_games=10000, max_mib=256, decoded_cache_mib=512):
        if not roots or max_games < 1 or max_mib < 1:
            raise ValueError("invalid replay window")
        paths = [Path(root) for root in roots]
        if any(path.is_symlink() or not (path.is_dir() or path.is_file()) for path in paths):
            raise ValueError("invalid replay source")

        total = sum(path.stat().st_size if path.is_file() else
                    sum(p.stat().st_size for p in path.glob("shard-*/samples.bin")) for path in paths)
        if total > max_mib * 1024 * 1024:
            raise ValueError("replay inputs exceed max_replay_mib")
        self.sources = []
        descriptions = []
        direct_bins = []
        for path in paths:
            if path.is_dir():
                source = Replay(path)
                self.sources.append(source)
                descriptions.append({"type": "shard", "manifest": source.meta,
                                     "shards": [info for _, info in source.shards]})
                direct_bins.append(None)
            else:
                if path.suffix.lower() != ".bin":
                    raise ValueError("direct replay source must be a .bin file")
                size = path.stat().st_size
                if size == 0 or size % RECORD_BYTES:
                    raise ValueError("direct replay .bin has invalid size")
                records = np.memmap(path, dtype=DTYPE, mode="r")
                if len(records) == 0:
                    raise ValueError("direct replay .bin is empty")
                if np.any(records["value_format"] > 1):
                    raise ValueError("direct replay .bin has unknown value format")
                direct_bins.append(records)
                self.sources.append(records)
                descriptions.append({"type": "direct-bin", "path": str(path.resolve()),
                                     "sha256": digest({"bytes": size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}),
                                     "records": len(records), "seed": 20260909})
        total = sum(path.stat().st_size for path in paths)
        if total > max_mib * 1024 * 1024:
            raise ValueError("replay grew beyond max_replay_mib while opening snapshot")
        games, seen, offset = [], set(), 0
        source_ends = []
        for i, source in enumerate(self.sources):
            if direct_bins[i] is not None:
                records = direct_bins[i]
                unique, inverse = np.unique(records["game_id"], return_inverse=True)
                if len(unique) == 0:
                    raise ValueError("direct replay .bin has no games")
                seed = 20260909
                start = offset
                for game_index, game_id in enumerate(unique):
                    mask = inverse == game_index
                    length = int(mask.sum())
                    game_records = records[mask]
                    if np.unique(game_records["ply"]).size != length or not np.all(game_records["ply"] == np.arange(length)):
                        raise ValueError("direct replay .bin has non-contiguous game plies")
                    validation = validation_game(int(game_id), seed)
                    games.append((i, int(game_id), start, length, validation))
                    start += length
                offset = start
                source_ends.append(len(records))
            else:
                identity = digest(source.meta)
                if identity in seen:
                    raise ValueError("duplicate replay dataset identity")
                seen.add(identity)
                source_start = offset
                for path, info in source.shards:
                    if verifier is None:
                        raise ValueError("--verifier is required for shard replay directories")
                    from selfplay import verify_shard
                    verify_shard(Path(verifier).resolve(), path, source.meta, info["start"], info["games"])
                    for game, length in enumerate(info["lengths"], info["start"]):
                        games.append((i, game, offset, length, validation_game(game, source.meta["seed"])))
                        offset += length
                source_ends.append(offset - source_start)
        self.ends = np.cumsum([int(n) for n in source_ends], dtype=np.int64)
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
        self.decoded_train = None
        decoded_bytes = len(self.indices["train"]) * (2 * 8 * 8 * 4 + 65 + 65 * 4 + 2 * 4)
        if decoded_cache_mib > 0 and decoded_bytes <= decoded_cache_mib * 1024 * 1024:
            cached = {key: [] for key in ("board", "legal", "policy", "wdl", "margin")}
            for start in range(0, len(self.indices["train"]), 8192):
                batch = decode(self.gather(self.indices["train"][start:start + 8192]))
                for key in cached:
                    cached[key].append(batch[key])
            self.decoded_train = {key: np.concatenate(value, axis=0) for key, value in cached.items()}
            self.statistics["decoded_cache_bytes"] = decoded_bytes
        else:
            self.statistics["decoded_cache_bytes"] = 0
        # Labels/distributions come from the frozen replay only, never a heuristic teacher.
        for split, indices in self.indices.items():
            phases = {name: {"samples": 0, "loss": 0, "draw": 0, "win": 0} for name in ("opening", "middle", "end")}
            for start in range(0, len(indices), 8192):
                records = self.gather(indices[start:start + 8192])
                occupied = np.bitwise_count(records["player"] | records["opponent"])
                for name, mask in (("opening", occupied < 24), ("middle", (occupied >= 24) & (occupied < 44)),
                                   ("end", occupied >= 44)):
                    phases[name]["samples"] += int(mask.sum())
                    results = np.sign(records["outcome"].astype(np.int16))
                    for result, value in (("loss", -1), ("draw", 0), ("win", 1)):
                        phases[name][result] += int(((results == value) & mask).sum())
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
            source = self.sources[owner]
            local = indices[mask] - base
            if isinstance(source, np.memmap):
                result[mask] = source[local]
            else:
                result[mask] = source.gather(local)
        return result

    def sample(self, batch, rng, augment=True):
        if self.decoded_train is not None:
            count = len(self.decoded_train["board"])
            selected = rng.integers(0, count, size=batch)
            result = {key: value[selected].copy() for key, value in self.decoded_train.items()}
            if augment:
                symmetries = rng.integers(0, 8, size=batch)
                mapping = INVERSE[symmetries]
                board = result["board"].reshape(batch, 2, 64)
                result["board"] = np.take_along_axis(board, mapping[:, None, :64], axis=2).reshape(batch, 2, 8, 8)
                result["legal"] = np.take_along_axis(result["legal"], mapping, axis=1)
                result["policy"] = np.take_along_axis(result["policy"], mapping, axis=1)
            return result
        indices = rng.choice(self.indices["train"], size=batch, replace=True)
        return decode(self.gather(indices), rng.integers(0, 8, size=batch) if augment else None)
