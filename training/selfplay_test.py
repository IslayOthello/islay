"""Real-network self-play, resume, independent replay oracle and corruption checks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import numpy as np

from replay import DTYPE, RECORD_BYTES, Replay, decode, game_seed, manifest, sha256, shards
from selfplay import FRAME, config_args, verify_shard

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from perft_study import START, play, scalar_moves


def generate(args, output, games, **overrides):
    options = {"games": games, "simulations": 16, "shard-games": 4, "cache-mib": 16, **overrides}
    command = [sys.executable, str(Path(__file__).with_name("selfplay.py")), "--engine", str(args.engine),
               "--model", str(args.model), "--output", str(output)]
    for key, value in options.items():
        command += ["--" + key, str(value)]
    return subprocess.run(command, capture_output=True, text=True, timeout=600)


def oracle(records):
    p, o = START
    outcomes, formats, game, ply, passes = [], [], None, 0, 0

    def check_targets(black_margin):
        for i, (target, value_format) in enumerate(zip(outcomes, formats)):
            margin = black_margin if i % 2 == 0 else -black_margin
            if value_format == 0:
                expected = (margin > 0) - (margin < 0)
            else:
                raw = np.tanh(margin / 15.0) * 127.0
                expected = int(np.sign(raw) * np.floor(abs(raw) + 0.5))
            assert target == expected, (target, expected, value_format)

    for r in records:
        if game is None or int(r["game_id"]) != game:
            if game is not None:
                assert not scalar_moves(p, o) and not scalar_moves(o, p)
                black_margin = (p.bit_count() - o.bit_count()) * (-1 if ply % 2 else 1)
                check_targets(black_margin)
            p, o = START
            game, ply, outcomes, formats = int(r["game_id"]), 0, [], []
        assert (int(r["player"]), int(r["opponent"])) == (p, o)
        moves = dict(scalar_moves(p, o))
        assert int(r["legal"]) == sum(1 << a for a in moves)
        action = int(r["action"])
        if action == 64:
            assert not moves and scalar_moves(o, p) and r["pass"] == 1
            p, o = o, p
            passes += 1
        else:
            assert action in moves and not r["pass"]
            p, o = play(p, o, (action, moves[action]))
        outcomes.append(int(r["outcome"]))
        formats.append(int(r["value_format"]))
        ply += 1
    assert not scalar_moves(p, o) and not scalar_moves(o, p)
    black_margin = (p.bit_count() - o.bit_count()) * (-1 if ply % 2 else 1)
    check_targets(black_margin)
    return passes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    args = parser.parse_args()
    args.engine, args.model = args.engine.resolve(), args.model.resolve()
    if args.games < 4: parser.error("need at least four games")
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output / "replay"
    first = generate(args, root, 3)
    assert first.returncode == 0, first.stderr
    committed_before = [(p, p.read_bytes()) for p in root.glob("shard-*/*")]
    resumed = generate(args, root, args.games)
    assert resumed.returncode == 0, resumed.stderr
    assert all(p.read_bytes() == data for p, data in committed_before)
    print(f"{args.games} neural games completed/resumed", flush=True)
    again = generate(args, root, args.games)
    assert again.returncode == 0 and '"new_games": 0' in again.stdout, again.stderr
    rejected = generate(args, root, args.games + 1, seed=42)
    assert rejected.returncode != 0 and "identical model" in rejected.stderr
    capacity = generate(args, args.output / "byte-cap", 16, **{"max-mib": 3, "shard-games": 16})
    assert capacity.returncode != 0 and "cap" in capacity.stderr
    assert not list((args.output / "byte-cap").glob("shard-*"))
    replay = Replay(root)
    all_records = replay.gather(np.arange(replay.ends[-1]))
    assert len(np.unique(all_records["game_id"])) == args.games
    passes = oracle(all_records)
    if args.games >= 100: assert passes > 0
    train = set(all_records["game_id"][replay.indices["train"]].tolist())
    validation = set(all_records["game_id"][replay.indices["validation"]].tolist())
    assert not train.intersection(validation) and len(train | validation) == args.games
    if args.games >= 100: assert train and validation

    meta = manifest(root)
    legacy_root = args.output / "legacy-wld"
    shutil.copytree(root, legacy_root)
    legacy_meta = dict(meta)
    legacy_meta["schema"] = "islay-replay-v1"
    legacy_meta.pop("value_target")
    legacy_id = hashlib.sha256(json.dumps(legacy_meta, sort_keys=True).encode()).hexdigest()
    for directory in legacy_root.glob("shard-*"):
        samples = directory / "samples.bin"
        old_records = np.memmap(samples, dtype=DTYPE, mode="r+")
        old_records["outcome"] = np.sign(old_records["outcome"]).astype(np.int8)
        old_records["value_format"] = 0
        old_records.flush()
        del old_records
        index = json.loads((directory / "index.json").read_text())
        index["dataset"], index["sha256"] = legacy_id, sha256(samples)
        (directory / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    (legacy_root / "manifest.json").write_text(json.dumps(legacy_meta, indent=2, sort_keys=True) + "\n")
    old_replay = Replay(legacy_root)
    old_records = old_replay.gather(np.arange(old_replay.ends[-1]))
    np.testing.assert_array_equal(decode(old_records)["outcome"][:, 0], np.sign(all_records["outcome"]))
    for source, info in shards(legacy_root, legacy_meta):
        verify_shard(args.engine, source, legacy_meta, info["start"], info["games"])

    direct = subprocess.run([str(args.engine), "--model", str(args.model), "--games", "4", "--cache-mib", "0",
                             *config_args(meta)], capture_output=True, check=True, timeout=120)
    position, plain = 0, []
    for expected in range(4):
        magic, game, seed, length, reserved = FRAME.unpack_from(direct.stdout, position)
        assert magic == b"ISLAGM01" and game == expected and seed == game_seed(meta["seed"], game) and not reserved
        position += FRAME.size
        plain.append(direct.stdout[position:position + length * RECORD_BYTES])
        position += length * RECORD_BYTES
    assert position == len(direct.stdout)
    assert b"".join(plain) == all_records[all_records["game_id"] < 4].tobytes()
    print("Cache on/off and resumed RNG: byte-identical samples", flush=True)

    from model import encode, transform_policy
    import torch
    chosen = all_records[np.linspace(0, len(all_records) - 1, 64, dtype=int)]
    base = decode(chosen)
    expected = encode([(int(r["player"]), int(r["opponent"])) for r in chosen]).numpy()
    np.testing.assert_array_equal(base["board"], expected)
    for s in range(8):
        batch = decode(chosen, np.full(len(chosen), s))
        board = torch.tensor(base["board"])
        if s & 1: board = board.flip(-1)
        if s & 2: board = board.flip(-2)
        if s & 4: board = board.transpose(-2, -1)
        np.testing.assert_array_equal(batch["board"], board.numpy())
        for key in ("policy", "legal", "priors"):
            expected = transform_policy(torch.tensor(base[key]), s).numpy()
            np.testing.assert_array_equal(batch[key], expected)
        assert np.all(batch["policy"][~batch["legal"]] == 0)
        np.testing.assert_allclose(batch["policy"].sum(1), 1, atol=1e-6)
    rng = np.random.default_rng(42)
    batch = replay.torch_batch(128, rng)
    from model import PolicyValueNet, policy_value_loss
    torch.set_num_threads(1)
    network = PolicyValueNet()
    p, v = network(batch["board"])
    loss = policy_value_loss(p, v, batch["policy"], batch["outcome"], batch["legal"])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())

    source, info = shards(root, meta)[0]
    original = source.read_bytes()
    mutations = {"outcome": (556, 128), "action": (554, 65), "value_format": (559, 2),
                 "board": (8, original[8] ^ 1), "legal": (24, original[24] ^ 1), "temperature": (557, 9)}
    for name, (offset, value) in mutations.items():
        modified = bytearray(original)
        modified[offset] = value
        path = args.output / ("bad-" + name + ".bin")
        path.write_bytes(modified)
        try: verify_shard(args.engine, path, meta, info["start"], info["games"])
        except subprocess.CalledProcessError: pass
        else: raise AssertionError("accepted corrupted " + name)
    for trim in (1, RECORD_BYTES):
        path = args.output / f"truncated-{trim}.bin"
        path.write_bytes(original[:-trim])
        try: verify_shard(args.engine, path, meta, info["start"], info["games"])
        except subprocess.CalledProcessError: pass
        else: raise AssertionError("accepted truncated game")
    corrupt = args.output / "corrupt-copy"
    shutil.copytree(root, corrupt)
    path = next(corrupt.glob("shard-*/samples.bin"))
    with path.open("r+b") as output:
        output.write(b"CORRUPT!")
    try: Replay(corrupt)
    except ValueError: pass
    else: raise AssertionError("accepted wrong checksum")
    result = {"result": "SELFPLAY/REPLAY TESTS PASSED", "games": args.games, "samples": len(all_records),
              "passes": passes, "train_games": len(train), "validation_games": len(validation),
              "resumed_metrics": json.loads(resumed.stdout[resumed.stdout.index("{\n"):])}
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
