#!/usr/bin/env python3
"""Reproducible single-thread perft experiments; artifacts stay in --directory.

Generate frozen experiment inputs, then build, verify and measure:
  python3 tools/perft_study.py snapshot
  python3 tools/perft_study.py prepare
  python3 tools/perft_study.py build baseline probe_first threshold4 threshold5 raw_key bucket2 depth_preferred full_neon
  python3 tools/perft_study.py verify baseline probe_first
  python3 tools/perft_study.py run probe_first --rounds 7 --milliseconds 150
  python3 tools/perft_study.py report
All traversal timings exclude process startup, TT allocation, and cold TT clears.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import time

MASK = (1 << 64) - 1
START = ((1 << 28) | (1 << 35), (1 << 27) | (1 << 36))
DIRECTIONS = [(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if x or y]


def snapshot(directory):
    """Generate isolated experiment inputs; never edit production sources."""
    base = directory / "base"
    if base.exists():
        raise SystemExit(f"Refusing to overwrite {base}; choose a fresh --directory")
    base.mkdir(parents=True)
    revision = "4e7b75cda6b197117bf0382d0e357a9cb693b0d8"
    names = ["perft.cpp", "perft.hpp", "board.cpp", "board.hpp", "movegen.cpp", "movegen.hpp",
             "common.hpp", "bitboard.hpp", "hash.hpp", "options.hpp"]
    for name in names:
        content = subprocess.check_output(["git", "show", f"{revision}:src/{name}"])
        (base / name).write_bytes(content)
    patches = Path(__file__).resolve().parent / "perft_variants"
    for variant in ["baseline", "full_neon"] + [p.stem for p in sorted(patches.glob("*.patch"))]:
        target = directory / variant
        target.mkdir()
        shutil.copyfile(base / "perft.cpp", target / "perft.cpp")
        patch = patches / (variant + ".patch")
        if patch.exists():
            subprocess.run(["git", "apply", "--unidiff-zero", "--directory=" + str(target), str(patch)], check=True)
    for name in ["pgo", "pgo-instrument", "profiles"]:
        (directory / name).mkdir()
    print("Snapshot", revision, "created at", directory, flush=True)


def scalar_moves(p, o):
    """Independent, square-scanning oracle: no bitboard move-generation kernel."""
    result = []
    for sq in range(64):
        if ((p | o) >> sq) & 1:
            continue
        flips = 0
        for dx, dy in DIRECTIONS:
            x, y = sq % 8 + dx, sq // 8 + dy
            ray = 0
            while 0 <= x < 8 and 0 <= y < 8 and (o >> (y * 8 + x)) & 1:
                ray |= 1 << (y * 8 + x)
                x, y = x + dx, y + dy
            if ray and 0 <= x < 8 and 0 <= y < 8 and (p >> (y * 8 + x)) & 1:
                flips |= ray
        if flips:
            result.append((sq, flips))
    return result


def play(p, o, move):
    sq, flips = move
    return o ^ flips, p ^ flips ^ (1 << sq)


def oracle(p, o, depth, rule):
    if depth == 0:
        return 1
    moves = scalar_moves(p, o)
    if not moves:
        return oracle(o, p, depth - 1, rule) if rule == "Othello" and scalar_moves(o, p) else 0
    if depth == 1:
        return len(moves)
    return sum(oracle(*play(p, o, move), depth - 1, rule) for move in moves)


def prepare(directory):
    rng = random.Random(20260909)
    positions = [{"id": "start", "p": START[0], "o": START[1], "depth": 11}]
    snapshots = {}
    passes = []
    validation = [START]
    for game in range(40):
        p, o = START
        for ply in range(70):
            moves = scalar_moves(p, o)
            if not moves:
                if not scalar_moves(o, p):
                    break
                passes.append((p, o))
                p, o = o, p
                continue
            discs = (p | o).bit_count()
            if game < 4 and discs in (12, 24, 40, 52, 60):
                snapshots[game, discs] = (p, o)
                validation.append((p, o))
            p, o = play(p, o, rng.choice(moves))
    holdout = []
    for game in (0, 1, 2, 3):
        for discs, depth in ((12, 8), (24, 7), (40, 7), (52, 9)):
            p, o = snapshots[game, discs]
            (positions if game < 2 else holdout).append(
                {"id": f"g{game}_discs{discs}", "p": p, "o": o, "depth": depth})
    # Bias a separate legal playout toward low opponent mobility to obtain
    # a nontrivial forced-pass benchmark, not a one-node endgame microbenchmark.
    found = False
    for game in range(200):
        p, o = START
        for ply in range(70):
            moves = scalar_moves(p, o)
            if not moves:
                other_moves = scalar_moves(o, p)
                if not other_moves:
                    break
                if 12 <= 64 - (p | o).bit_count() <= 30 and len(other_moves) >= 3:
                    passes.append((p, o))
                    found = True
                    break
                p, o = o, p
                continue
            if rng.random() < 0.7:
                moves.sort(key=lambda move: len(scalar_moves(*play(p, o, move))))
                move = moves[0]
            else:
                move = rng.choice(moves)
            p, o = play(p, o, move)
        if found:
            break
    if not found:
        raise RuntimeError("No nontrivial forced-pass position generated")
    passes = sorted(set(passes), key=lambda board: (board[0] | board[1]).bit_count())
    p, o = passes[0]
    empties = 64 - (p | o).bit_count()
    positions.append({"id": "forced_pass", "p": p, "o": o, "depth": min(9, empties + 1)})
    validation += passes[:3] + passes[-1:] + [(MASK, 0), (0, 0), (2, 1)]
    checks = []
    for p, o in validation:
        for rule in ("Othello", "Reversi"):
            for depth in range(4):
                checks.append([p, o, depth, rule, oracle(p, o, depth, rule)])
    known = [1, 4, 12, 56, 244, 1396, 8200, 55092, 390216]
    for d, nodes in enumerate(known):
        checks.append([*START, d, "Othello", nodes])
    # Generic dispatcher (>12), forced passes and termination with only four empties.
    for game in range(4):
        p, o = snapshots[game, 60]
        for rule in ("Othello", "Reversi"):
            for depth in (4, 5, 13):
                checks.append([p, o, depth, rule, oracle(p, o, depth, rule)])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "corpus.json").write_text(json.dumps(
        {"seed": 20260909, "positions": positions, "holdout": holdout, "checks": checks}, indent=2) + "\n")
    print(f"Prepared {len(positions)} timed positions, {len(validation)} oracle positions, {len(checks)} oracle cases", flush=True)


def binary(directory, variant):
    return directory / variant / "bench"


def build(directory, variants):
    common = ["clang++", "-std=c++20", "-O3", "-DNDEBUG", "-march=native", "-flto=thin"]
    sources = ["tools/perft_bench.cpp", str(directory / "base/board.cpp"), str(directory / "base/movegen.cpp")]
    for variant in variants:
        if variant == "baseline_control":
            (directory / variant).mkdir(exist_ok=True)
            shutil.copy2(binary(directory, "baseline"), binary(directory, variant))
            print("COPIED baseline_control from baseline", flush=True)
            continue
        extra = ["-DISLAY_USE_NEON"] if variant == "full_neon" else []
        source_variant = variant
        if variant == "pgo-instrument":
            extra += ["-fprofile-instr-generate"]
            source_variant = "baseline"
        elif variant == "pgo":
            extra += ["-fprofile-instr-use=" + str(directory / "training.profdata")]
            source_variant = "baseline"
        cmd = common + extra + ["-I" + str(directory / "base")] + sources + [str(directory / source_variant / "perft.cpp"), "-o", str(binary(directory, variant))]
        print("BUILD", variant, flush=True)
        start = time.monotonic()
        subprocess.run(cmd, check=True)
        (directory / variant / "build.json").write_text(json.dumps({"command": cmd, "seconds": time.monotonic() - start,
            "sha256": hashlib.sha256(binary(directory, variant).read_bytes()).hexdigest()}, indent=2) + "\n")
        print("BUILT", variant, round(time.monotonic() - start, 1), flush=True)


def corpus(directory):
    return json.loads((directory / "corpus.json").read_text())


def verify(directory, variants):
    data = corpus(directory)
    text = "".join(" ".join(map(str, row)) + "\n" for row in data["checks"])
    for variant in variants:
        result = subprocess.run([str(binary(directory, variant)), "--verify"], input=text,
                                text=True, capture_output=True, timeout=180, check=True)
        print(variant, result.stdout.strip(), flush=True)
        (directory / variant / "verification.txt").write_text(result.stdout)


def measure(directory, variant, position, rule, mode, mib, milliseconds):
    cmd = [str(binary(directory, variant)), str(position["p"]), str(position["o"]),
           str(position["depth"]), rule, mode, str(mib), str(milliseconds), "1024"]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180)
    return json.loads(result.stdout)


def measure_uci(directory, variant, position, rule, mode, mib):
    if mode != "cold":
        raise ValueError("UCI cross-check currently supports cold TT only")
    diagram = "".join("X" if (position["p"] >> sq) & 1 else "O" if (position["o"] >> sq) & 1 else "-"
                      for sq in range(64))
    commands = f"setoption name PerftHash value {mib}\nsetoption name Rule value {rule}\n"
    commands += (f"ucinewgame\nposition fen {diagram} X\ngo perft {position['depth']}\n") * 5 + "quit\n"
    result = subprocess.run([str(directory / variant / "islay")], input=commands,
                            check=True, capture_output=True, text=True, timeout=180)
    if "info error:" in result.stdout:
        raise RuntimeError(result.stdout)
    counts = [int(x.replace(",", "")) for x in re.findall(r"Nodes searched: ([0-9,]+)", result.stdout)]
    speeds = [float(x.replace(",", "")) for x in re.findall(r"Speed: ([0-9,.]+) N/s", result.stdout)]
    if len(counts) != 5 or len(speeds) != 5 or len(set(counts)) != 1:
        raise RuntimeError(result.stdout)
    # Discard one warmup; derive elapsed time from full-precision Speed, not integer Time.
    elapsed_ms = sum(counts[0] / nps * 1000 for nps in speeds[1:])
    return {"nodes": counts[0], "iterations": 4, "elapsed_ms": elapsed_ms,
            "nps": counts[0] * 4000 / elapsed_ms, "ns_per_call": elapsed_ms * 1e6 / 4,
            "checksum": counts[0] * 4}


def run(directory, variants, rounds, milliseconds, modes, mib, tag, holdout, uci):
    data = corpus(directory)
    path = directory / f"results-{tag}.jsonl"
    if path.exists():
        raise SystemExit(f"Refusing to overwrite {path}; use a new --tag")
    rng = random.Random(20260909)
    with path.open("x") as journal:
        for variant in variants:
            print("MEASURE", variant, flush=True)
            for round_index in range(rounds):
                positions = list(data["positions"]) + (data["holdout"] if holdout else [])
                rng.shuffle(positions)
                for position in positions:
                    for rule in ("Othello", "Reversi"):
                        # Zero-node terminal Reversi roots belong in correctness tests, not NPS summaries.
                        if rule == "Reversi" and not scalar_moves(position["p"], position["o"]):
                            continue
                        for mode in modes:
                            order = ["baseline", variant] if round_index % 2 == 0 else [variant, "baseline"]
                            results = {}
                            for name in order:
                                results[name] = (measure_uci(directory, name, position, rule, mode, mib) if uci else
                                                 measure(directory, name, position, rule, mode, mib, milliseconds))
                            assert results["baseline"]["nodes"] == results[variant]["nodes"], (variant, position, results)
                            row = {"variant": variant, "round": round_index, "position": position["id"],
                                   "timestamp": time.time(),
                                   "depth": position["depth"], "rule": rule,
                                   "mode": "uci-" + mode if uci else mode, "hash_mib": mib,
                                   "baseline": results["baseline"], "candidate": results[variant],
                                   "ratio": results[variant]["nps"] / results["baseline"]["nps"] if results["baseline"]["nps"] else None}
                            journal.write(json.dumps(row) + "\n")
                            journal.flush()
                print("ROUND", variant, round_index + 1, "/", rounds, flush=True)
    print("SAVED", path, flush=True)


def report(directory):
    grouped = {}
    for path in sorted(directory.glob("results-*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["ratio"] is None:
                continue
            key = (path.stem, row["variant"], row["mode"], row["hash_mib"])
            grouped.setdefault(key, []).append(row)
    summary = []
    for key, rows in grouped.items():
        cells = {}
        for row in rows:
            cells.setdefault((row["position"], row["rule"]), []).append(row)
        cell_report = []
        for (position, rule), samples in sorted(cells.items()):
            ratios = [x["ratio"] for x in samples]
            cell_report.append({"position": position, "rule": rule, "rounds": len(samples),
                "baseline_nps": statistics.median(x["baseline"]["nps"] for x in samples),
                "candidate_nps": statistics.median(x["candidate"]["nps"] for x in samples),
                "median_paired_ratio": statistics.median(ratios)})
        ratios = [x["median_paired_ratio"] for x in cell_report]
        # Resample whole A/B rounds together to retain cross-position timing drift.
        round_ids = sorted({row["round"] for row in rows})
        round_rng = random.Random(20260909)
        resampled = []
        by_cell = [{row["round"]: row["ratio"] for row in samples} for samples in cells.values()]
        if all(len(cell) == len(round_ids) for cell in by_cell):
            for _ in range(2000):
                draw = round_rng.choices(round_ids, k=len(round_ids))
                medians = [statistics.median(cell[r] for r in draw) for cell in by_cell]
                resampled.append(100 * (math.exp(statistics.mean(math.log(x) for x in medians)) - 1))
        resampled.sort()
        item = {"run": key[0], "variant": key[1], "mode": key[2], "hash_mib": key[3],
                "median_position_gain_pct": 100 * (statistics.median(ratios) - 1),
                "geomean_gain_pct": 100 * (math.exp(statistics.mean(math.log(x) for x in ratios)) - 1),
                "min_gain_pct": 100 * (min(ratios) - 1), "max_gain_pct": 100 * (max(ratios) - 1),
                "geomean_round_bootstrap_95_pct": [resampled[50], resampled[1949]] if resampled else None,
                "cells": cell_report}
        summary.append(item)
        print(f"{key}: median {item['median_position_gain_pct']:+.2f}% geo {item['geomean_gain_pct']:+.2f}% "
              f"range [{item['min_gain_pct']:+.2f}, {item['max_gain_pct']:+.2f}]%")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def train_pgo(directory):
    rng = random.Random(20260910)
    positions = []
    for game in range(3):
        p, o = START
        for ply in range(60):
            if game == 0 and ply == 2:
                positions.append({"id": "training_deep_opening", "p": p, "o": o, "depth": 11})
            if ply in (8, 20, 36, 48):
                positions.append({"id": f"training_{game}_{ply}", "p": p, "o": o,
                                  "depth": 6 if ply < 40 else 7})
            moves = scalar_moves(p, o)
            if not moves:
                if not scalar_moves(o, p):
                    break
                p, o = o, p
                continue
            p, o = play(p, o, rng.choice(moves))
    (directory / "pgo-training.json").write_text(json.dumps(positions, indent=2) + "\n")
    environment = dict(os.environ)
    environment["LLVM_PROFILE_FILE"] = str(directory.resolve() / "profiles/training-%p.profraw")
    for position in positions:
        for rule in ("Othello", "Reversi"):
            for mode in ("cold", "nocache", "warm"):
                cmd = [str(binary(directory, "pgo-instrument")), str(position["p"]), str(position["o"]),
                       str(position["depth"]), rule, mode, "256", "0", "1024"]
                subprocess.run(cmd, env=environment, check=True, capture_output=True, timeout=180)
        print("TRAINED", position["id"], flush=True)
    profiles = sorted((directory / "profiles").glob("training-*.profraw"))
    subprocess.run(["xcrun", "llvm-profdata", "merge", "-o", str(directory / "training.profdata")] +
                   [str(p) for p in profiles], check=True)
    print("PGO profile merged", len(profiles), "files", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["snapshot", "prepare", "build", "verify", "run", "report", "train-pgo"])
    parser.add_argument("variants", nargs="*")
    parser.add_argument("--directory", type=Path, default=Path("_build/perft-study"))
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--milliseconds", type=float, default=150)
    parser.add_argument("--modes", nargs="+", default=["cold"])
    parser.add_argument("--hash-mib", type=int, default=256)
    parser.add_argument("--tag", default="pilot")
    parser.add_argument("--include-holdout", action="store_true")
    parser.add_argument("--uci", action="store_true", help="measure frozen UCI binaries named islay, four traversals per sample")
    args = parser.parse_args()
    if args.action == "run" and (args.rounds < 1 or args.milliseconds <= 0):
        parser.error("rounds and milliseconds must be positive")
    if args.action == "snapshot":
        snapshot(args.directory)
    elif args.action == "prepare":
        prepare(args.directory)
    elif args.action == "build":
        build(args.directory, args.variants)
    elif args.action == "verify":
        verify(args.directory, args.variants)
    elif args.action == "run":
        run(args.directory, args.variants, args.rounds, args.milliseconds, args.modes, args.hash_mib, args.tag,
            args.include_holdout, args.uci)
    elif args.action == "train-pgo":
        train_pgo(args.directory)
    else:
        report(args.directory)


if __name__ == "__main__":
    main()
