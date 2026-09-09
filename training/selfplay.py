"""Run one C++ self-play owner; atomically publish bounded, resumable replay shards."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import tempfile
import time

import onnx

from replay import RECORD_BYTES, SCHEMA, game_seed, manifest, sha256, shards

FRAME = struct.Struct("<8sQQII")


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path, data):
    with open(path, "x") as output:
        json.dump(data, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def read_exact(stream, size):
    result = bytearray()
    while len(result) < size:
        data = stream.read(size - len(result))
        if not data:
            raise RuntimeError("self-play exited mid-frame; incomplete game not published")
        result.extend(data)
    return bytes(result)


def config_args(meta):
    return ["--seed", str(meta["seed"]), "--simulations", str(meta["simulations"]),
            "--tree-mib", str(meta["tree_mib"]), "--epsilon", str(meta["epsilon"]),
            "--alpha", str(meta["alpha"]), "--temperature-plies", str(meta["temperature_plies"])]


def verify_shard(binary, path, meta, start, games):
    result = subprocess.run([str(binary), "--verify", str(path), "--start", str(start), "--games", str(games),
                             *config_args(meta)], check=True, text=True, capture_output=True, timeout=120)
    return json.loads(result.stdout)


def run(args):
    root = args.output.resolve()
    binary, model = args.engine.resolve(), args.model.resolve()
    if not model.is_file() or not 0 < model.stat().st_size <= 64 * 1024 * 1024:
        raise ValueError("expected single ONNX file of at most 64 MiB")
    if model.stat().st_size + 32768 > args.max_mib * 1024 * 1024:
        raise ValueError("model snapshot exceeds replay byte cap")
    runtime = json.loads(subprocess.check_output([str(binary), "--identity"], text=True, timeout=20))
    if runtime["backend"] != "onnx-cpu":
        raise ValueError("self-play requires ISLAY_ONNX=ON")
    properties = {p.key: p.value for p in onnx.load(model, load_external_data=False).metadata_props}
    meta = {"schema": SCHEMA, "record_bytes": RECORD_BYTES, "rule": "Othello", "encoding": "relative-2x8x8-a1-v1",
            "architecture": "b8c64-v1", "policy": "visits-tau1-prefix-tau0-lowest-v1",
            "rng": "splitmix64-boxmuller-marsaglia-v1", "seed": args.seed, "simulations": args.simulations,
            "tree_mib": args.tree_mib, "epsilon": args.epsilon, "alpha": args.alpha,
            "temperature_plies": args.temperature_plies, "c_puct": 1.5, "model_sha256": sha256(model),
            "engine_sha256": sha256(binary), "platform": platform.platform(),
            "checkpoint_sha256": properties["islay.checkpoint_sha256"], "runtime": runtime,
            "backend": "onnx-cpu-fp32-one-thread", "split": "mix64-game-seed-mod10-v1"}
    identity = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".lock", "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "manifest.json").exists():
            if manifest(root) != meta or sha256(root / "model.onnx") != meta["model_sha256"]:
                raise ValueError("resume requires identical model, engine binary and generation config")
        else:
            if list(root.glob("shard-*")) or (root / "model.onnx").exists():
                raise ValueError("incomplete dataset initialization; preserve it and choose a fresh directory")
            with tempfile.TemporaryDirectory(prefix=".init-", dir=root) as temporary:
                staging = Path(temporary)
                shutil.copyfile(model, staging / "model.onnx")
                if sha256(staging / "model.onnx") != meta["model_sha256"]:
                    raise ValueError("model changed during snapshot")
                with open(staging / "model.onnx", "rb") as source:
                    os.fsync(source.fileno())
                write_json(staging / "manifest.json", meta)
                os.rename(staging / "model.onnx", root / "model.onnx")
                os.rename(staging / "manifest.json", root / "manifest.json")
                sync_directory(root)
        existing = shards(root, meta)
        completed = sum(info["games"] for _, info in existing)
        samples = sum(info["samples"] for _, info in existing)
        # Include incomplete data left by a killed producer in the hard byte budget.
        used = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        for path, info in existing:
            verify_shard(binary, path, meta, info["start"], info["games"])
        if completed >= args.games:
            return {"games": completed, "samples": samples, "new_games": 0}
        pending_games = args.games - completed
        command = [str(binary), "--model", str(root / "model.onnx"), "--start", str(completed),
                   "--games", str(pending_games), "--cache-mib", str(args.cache_mib), *config_args(meta)]
        started = time.perf_counter()
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
            try:
                while completed < args.games:
                    count = min(args.shard_games, args.games - completed)
                    # Reserve worst-case full shard and metadata before consuming any game.
                    reserve = count * 128 * RECORD_BYTES + 16384
                    if used + reserve > args.max_mib * 1024 * 1024:
                        raise ValueError("replay byte cap reached; raise --max-mib or use a new dataset (nothing evicted)")
                    with tempfile.TemporaryDirectory(prefix=".pending-", dir=root) as temporary:
                        staging = Path(temporary)
                        lengths = []
                        with open(staging / "samples.bin", "xb") as output:
                            for expected in range(completed, completed + count):
                                magic, game_id, seed, length, reserved = FRAME.unpack(read_exact(process.stdout, FRAME.size))
                                if (magic != b"ISLAGM01" or game_id != expected or seed != game_seed(args.seed, game_id)
                                        or not 1 <= length <= 128 or reserved != 0):
                                    raise ValueError("invalid self-play frame")
                                output.write(read_exact(process.stdout, length * RECORD_BYTES))
                                lengths.append(length)
                            output.flush()
                            os.fsync(output.fileno())
                        checked = verify_shard(binary, staging / "samples.bin", meta, completed, count)
                        if checked["samples"] != sum(lengths):
                            raise ValueError("frame/index sample mismatch")
                        if sha256(root / "model.onnx") != meta["model_sha256"]:
                            raise ValueError("loaded model snapshot was modified; shard rejected")
                        info = {"dataset": identity, "start": completed, "games": count, "samples": sum(lengths),
                                "lengths": lengths, "sha256": sha256(staging / "samples.bin")}
                        write_json(staging / "index.json", info)
                        sync_directory(staging)
                        destination = root / f"shard-{completed:012d}"
                        if destination.exists():
                            raise ValueError("refusing to replace a committed shard")
                        used += sum(p.stat().st_size for p in staging.iterdir())
                        os.rename(staging, destination)
                        sync_directory(root)
                    completed += count
                    samples += sum(lengths)
                    print(json.dumps({"committed_games": completed, "samples": samples}), flush=True)
                if process.stdout.read(1):
                    raise ValueError("unexpected trailing self-play frame")
                code = process.wait(timeout=20)
                errors.seek(0)
                diagnostic = errors.read().decode()
                if code:
                    raise RuntimeError(diagnostic)
                metrics = json.loads(diagnostic)
                metrics.update({"new_games": pending_games, "total_games": completed, "total_samples": samples,
                                "wall_seconds": time.perf_counter() - started, "dataset_bytes": used})
                return metrics
            except BaseException as failure:
                if process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                errors.seek(0)
                diagnostic = errors.read().decode()
                if diagnostic and not isinstance(failure, KeyboardInterrupt):
                    print(diagnostic, end="", file=__import__("sys").stderr)
                raise
            finally:
                process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100, help="total target, including already committed games")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--tree-mib", type=int, default=8)
    parser.add_argument("--cache-mib", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--temperature-plies", type=int, default=16)
    parser.add_argument("--shard-games", type=int, default=16)
    parser.add_argument("--max-mib", type=int, default=256)
    args = parser.parse_args()
    if (not 1 <= args.games <= 1000000000 or not 0 <= args.seed < 1 << 64 or not 1 <= args.simulations <= 1000000
            or not 1 <= args.tree_mib <= 4096 or not 0 <= args.cache_mib <= 4096
            or not 0 <= args.epsilon <= 1 or not 0.01 <= args.alpha <= 100
            or not 0 <= args.temperature_plies <= 128 or not 1 <= args.shard_games <= 1024 or args.max_mib < 1):
        parser.error("invalid bounded generation configuration")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
