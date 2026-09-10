"""Fixed-sample paired ONNX arena, atomic pair commits and conservative candidate selection."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time

import numpy as np
import onnx

from arena_engine import Engine
from arena_game import actions, advance, canonical, openings, verify_game
from arena_stats import summarize
from replay import sha256
from selfplay import sync_directory, write_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def model_info(path):
    if not path.is_file() or not 0 < path.stat().st_size <= 64 * 1024 * 1024:
        raise ValueError("expected trusted single-file ONNX, at most 64 MiB")
    model = onnx.load(path, load_external_data=False)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer):
        raise ValueError("external ONNX data not supported")
    props = {p.key: p.value for p in model.metadata_props}
    checkpoint = props.get("islay.checkpoint_sha256", "")
    if len(checkpoint) != 64 or any(c not in "0123456789abcdef" for c in checkpoint):
        raise ValueError("invalid model checkpoint metadata")
    steps = props.get("islay.training_steps", "")
    if not steps.isascii() or not steps.isdecimal(): raise ValueError("invalid training step metadata")
    return {"sha256": sha256(path), "checkpoint_sha256": checkpoint, "training_steps": int(steps)}


def make_manifest(args):
    directory = Path(__file__).parent
    names = [directory / n for n in ("arena.py", "arena_engine.py", "arena_game.py", "arena_stats.py",
                                     "selfplay.py", "replay.py")]
    names.append(directory.parent / "tools" / "perft_study.py")
    config = {k: getattr(args, k) for k in ("pairs", "seed", "opening_plies", "mode", "budget", "tree_mib",
                                           "timeout_ms", "grace_ms", "threshold", "alpha")}
    return {"schema": "islay-arena-v1", "config": config, "engine_sha256": sha256(args.engine),
            "models": {name: model_info(getattr(args, name)) for name in ("candidate", "champion")},
            "runtime": {"platform": platform.platform(), "machine": platform.machine(),
                        "python": sys.version, "numpy": np.__version__, "onnx": onnx.__version__,
                        "cpu_count": os.cpu_count()},
            "code": {p.name: sha256(p) for p in names}, "root_noise": False, "temperature": 0,
            "opening_policy": "iid-uniform-legal-playout-with-replacement-v1",
            "openings": openings(args.pairs, args.seed, args.opening_plies)}


def verify_pair(pair, manifest, index):
    if pair["id"] != index or pair["manifest"] != digest(manifest) or len(pair["games"]) != 2:
        raise ValueError("invalid pair identity")
    expected_colors = [index % 2, 1 - index % 2]
    if [g["candidate_color"] for g in pair["games"]] != expected_colors:
        raise ValueError("pair must exchange candidate color and alternate first game")
    config = manifest["config"]
    for game in pair["games"]:
        verify_game(manifest["openings"][index], game)
        for ply, row in enumerate(game["moves"]):
            side = (manifest["openings"][index]["side"] + ply) % 2
            owner = "candidate" if side == game["candidate_color"] else "champion"
            model = manifest["models"][owner]
            prefix = ("info string evaluator onnx-cpu b8c64-v1 checkpoint " + model["checkpoint_sha256"]
                      + " training_steps " + str(model["training_steps"]))
            expected = prefix + (" (untrained initialization)" if model["training_steps"] == 0 else "")
            if row["evaluator"] != expected: raise ValueError("search used wrong evaluator")
            if (type(row["nodes"]) is not int or row["nodes"] < 0
                    or type(row["evaluations"]) is not int or row["evaluations"] < 0
                    or type(row["engine_ms"]) is not int or row["engine_ms"] < 0
                    or not math.isfinite(row["elapsed_us"]) or row["elapsed_us"] <= 0):
                raise ValueError("invalid search metrics")
            if config["mode"] == "nodes":
                if row["reason"] != "nodes" or row["nodes"] != config["budget"]:
                    raise ValueError("unequal simulation budget")
                deadline = config["timeout_ms"]
            else:
                if row["reason"] != "time": raise ValueError("unequal time budget")
                deadline = config["budget"] + config["grace_ms"]
            if row["elapsed_us"] > deadline * 1000: raise ValueError("arena time forfeit; match invalid")


def load_pairs(root, manifest):
    result = []
    for index, directory in enumerate(sorted(root.glob("pair-*"))):
        if directory.is_symlink() or directory.name != f"pair-{index:06d}" or index >= manifest["config"]["pairs"]:
            raise ValueError("noncontiguous or invalid pair commit")
        path = directory / "pair.json"
        entry = json.loads((directory / "index.json").read_text())
        if path.is_symlink() or path.stat().st_size > 262144 or entry != {"sha256": sha256(path)}:
            raise ValueError("pair checksum mismatch")
        pair = json.loads(path.read_text())
        verify_pair(pair, manifest, index)
        result.append(pair)
    return result


def play_game(engines, opening, color, config):
    for engine in engines.values(): engine.new_game()
    board, side, moves = tuple(opening["board"]), opening["side"], []
    while actions(board):
        if len(moves) >= 128: raise ValueError("arena game exceeded ply bound")
        owner = "candidate" if side == color else "champion"
        row = engines[owner].search(board, config)
        board = advance(board, row["action"])
        moves.append(row)
        side = 1 - side
    margin = board[0].bit_count() - board[1].bit_count()
    black_margin = margin if side == 0 else -margin
    candidate_margin = black_margin if color == 0 else -black_margin
    game = {"candidate_color": color, "moves": moves, "black_margin": black_margin,
            "score": (1 + (candidate_margin > 0) - (candidate_margin < 0)) / 2}
    verify_game(opening, game)
    return game


def publish(root, name, filename, value, reserve):
    if (root / name).exists(): raise ValueError("refusing to overwrite committed arena data")
    with tempfile.TemporaryDirectory(prefix=".pending-", dir=root) as temporary:
        staging = Path(temporary)
        write_json(staging / filename, value)
        write_json(staging / "index.json", {"sha256": sha256(staging / filename)})
        size = sum(p.stat().st_size for p in staging.iterdir())
        if size > reserve: raise ValueError("arena record exceeds reserved storage")
        sync_directory(staging)
        os.rename(staging, root / name)
        sync_directory(root)
    return size


def report(pairs, manifest):
    config = manifest["config"]
    unique = len({canonical(tuple(o["board"])) for o in manifest["openings"][:len(pairs)]})
    result = summarize(pairs, config["pairs"], config["seed"], unique, config["threshold"], config["alpha"])
    if manifest["models"]["candidate"]["sha256"] == manifest["models"]["champion"]["sha256"]:
        result["decision"] = "retain_champion"
    selected = "candidate" if result["decision"] == "candidate_eligible" else "champion"
    result.update(manifest=digest(manifest), selected_model=manifest["models"][selected], selected_role=selected,
                  deployment="none; selection artifact only", mode=config["mode"], budget=config["budget"])
    for owner in ("candidate", "champion"):
        rows = [row for pair in pairs for game in pair["games"] for ply, row in enumerate(game["moves"])
                if (((manifest["openings"][pair["id"]]["side"] + ply) % 2 == game["candidate_color"])
                    == (owner == "candidate"))]
        result[owner + "_search"] = {"moves": len(rows), "nodes": sum(r["nodes"] for r in rows),
                                     "elapsed_seconds": sum(r["elapsed_us"] for r in rows) / 1e6,
                                     "median_move_us": float(np.median([r["elapsed_us"] for r in rows]))}
    return result


def run(args):
    manifest = make_manifest(args)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    engines = {}
    with open(root / ".lock", "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assets = root / "assets"
        if args.resume:
            if json.loads((assets / "manifest.json").read_text()) != manifest:
                raise ValueError("resume requires identical models, engine, openings, runtime/code and fixed plan")
        else:
            reserve = args.candidate.stat().st_size + args.champion.stat().st_size + len(json.dumps(manifest)) * 3 + 32768
            if reserve > args.max_mib * 1048576: raise ValueError("arena assets exceed storage cap")
            with tempfile.TemporaryDirectory(prefix=".init-", dir=root) as temporary:
                staging = Path(temporary)
                for owner in ("candidate", "champion"):
                    path = staging / (owner + ".onnx")
                    shutil.copyfile(getattr(args, owner), path)
                    if sha256(path) != manifest["models"][owner]["sha256"]: raise ValueError("model changed during copy")
                    with path.open("rb") as source: os.fsync(source.fileno())
                write_json(staging / "manifest.json", manifest)
                sync_directory(staging)
                os.rename(staging, assets)
                sync_directory(root)

        def check_assets():
            if sha256(args.engine) != manifest["engine_sha256"]: raise ValueError("engine binary changed during arena")
            for owner in ("candidate", "champion"):
                if sha256(assets / (owner + ".onnx")) != manifest["models"][owner]["sha256"]:
                    raise ValueError("arena model snapshot modified")

        check_assets()
        pairs = load_pairs(root, manifest)
        if (root / "failure.json").exists():
            raise ValueError("recorded arena failure; investigate and start a new run, no silent retry")
        if (root / "final").exists():
            expected = report(pairs, manifest)
            path = root / "final" / "report.json"
            if (json.loads((root / "final" / "index.json").read_text()) != {"sha256": sha256(path)}
                    or json.loads(path.read_text()) != expected or not expected["complete"]):
                raise ValueError("invalid final arena report")
            return expected
        used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        try:
            for owner in ("candidate", "champion"):
                engines[owner] = Engine(args.engine)
                engines[owner].preflight()
                engines[owner].configure(assets / (owner + ".onnx"), args.tree_mib)
            stop = min(args.pairs, len(pairs) + (args.max_new_pairs or args.pairs))
            while len(pairs) < stop:
                if used + 524288 > args.max_mib * 1048576: raise ValueError("arena storage cap; nothing evicted")
                index = len(pairs)
                started = time.perf_counter()
                pair = {"id": index, "manifest": digest(manifest), "games": []}
                for color in (index % 2, 1 - index % 2):
                    pair["games"].append(play_game(engines, manifest["openings"][index], color, manifest["config"]))
                verify_pair(pair, manifest, index)
                check_assets()
                used += publish(root, f"pair-{index:06d}", "pair.json", pair, 262144)
                pairs.append(pair)
                print(json.dumps({"committed_pairs": len(pairs), "target_pairs": args.pairs,
                                  "pair_seconds": time.perf_counter() - started}), flush=True)
            for engine in engines.values(): engine.close()
            engines.clear()
            check_assets()
            result = report(pairs, manifest)
            if result["complete"]:
                if used + 262144 > args.max_mib * 1048576: raise ValueError("arena report storage cap")
                publish(root, "final", "report.json", result, 262144)
            return result
        except Exception as failure:
            write_json(root / "failure.json", {"committed_pairs": len(pairs), "error": str(failure),
                                                "type": type(failure).__name__, "time_ns": time.time_ns()})
            sync_directory(root)
            raise
        finally:
            for engine in engines.values(): engine.close(abort=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("engine", "candidate", "champion", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=200, help="fixed total; cannot grow on resume")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--opening-plies", type=int, default=12)
    parser.add_argument("--mode", choices=("nodes", "movetime"), default="nodes")
    parser.add_argument("--budget", type=int, default=128, help="simulations or milliseconds per move")
    parser.add_argument("--tree-mib", type=int, default=8)
    parser.add_argument("--timeout-ms", type=int, default=30000, help="nodes-mode watchdog, not a chess clock")
    parser.add_argument("--grace-ms", type=int, default=250, help="movetime transport/in-flight inference allowance")
    parser.add_argument("--threshold", type=float, default=0.55, help="required lower confidence bound on score")
    parser.add_argument("--alpha", type=float, default=0.05, help="one-sided fixed-sample gate error budget")
    parser.add_argument("--max-mib", type=int, default=256)
    parser.add_argument("--max-new-pairs", type=int, help="administrative pause; never an early selection gate")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (not 1 <= args.pairs <= 10000 or not 0 <= args.seed < 1 << 63 or not 8 <= args.opening_plies <= 40
            or not 1 <= args.budget <= 1000000 or not 1 <= args.tree_mib <= 4096
            or not 100 <= args.timeout_ms <= 3600000 or not 1 <= args.grace_ms <= 60000
            or not 0.5 <= args.threshold < 1 or not 0 < args.alpha <= 0.05 or args.max_mib < 1
            or (args.max_new_pairs is not None and not 1 <= args.max_new_pairs <= args.pairs)):
        parser.error("invalid bounded arena configuration")
    print(json.dumps(run(args), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
