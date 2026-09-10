"""Shared inference parity, native reference, exact cohort resume and replay corruption gates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from replay import DTYPE, RECORD_BYTES, Replay
from selfplay import FRAME
from selfplay_test import oracle
from shm_region import Region


def payload(root):
    replay = Replay(root)
    return replay.gather(np.arange(replay.ends[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worker", "verifier", "model", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    common = [sys.executable, str(Path(__file__).with_name("batched_selfplay.py")), "--worker", str(args.worker),
              "--verifier", str(args.verifier), "--model", str(args.model), "--checkpoint", str(args.checkpoint),
              "--simulations", "8"]

    def launch(root, backend, workers, games, *extra, success=True):
        result = subprocess.run([*common, "--output", str(root), "--backend", backend, "--workers", str(workers),
                                 "--games", str(games), *extra], capture_output=True, text=True, timeout=180)
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        return result

    with Region(8192) as region:
        fd = region.fd
        assert not os.get_inheritable(fd)
        region.map[:4] = b"test"
        assert region.map[:4] == b"test"
    try: os.fstat(fd)
    except OSError: pass
    else: raise AssertionError("shared descriptor leaked")

    reference = args.output / "ort-single"
    launch(reference, "ort-cpu", 1, 4)
    direct = subprocess.run([str(args.verifier), "--model", str(args.model), "--games", "4", "--simulations", "8",
                             "--cache-mib", "0", "--tree-mib", "1"], capture_output=True, timeout=120, check=True)
    offset, records = 0, []
    for expected in range(4):
        magic, game, _, length, flags = FRAME.unpack_from(direct.stdout, offset)
        assert magic == b"ISLAGM01" and game == expected and not flags
        offset += FRAME.size
        records.append(direct.stdout[offset:offset + length * RECORD_BYTES])
        offset += length * RECORD_BYTES
    assert offset == len(direct.stdout)
    assert payload(reference).tobytes() == b"".join(records), "ORT shm batch1 differs from native evaluator"
    print("Native C++ ORT vs shm ORT batch1: byte-identical replay", flush=True)

    import torch
    backends = ["ort-cpu", "torch-cpu"] + (["torch-mps"] if torch.backends.mps.is_available() else [])
    results = {}
    for backend in backends:
        split, whole = args.output / (backend + "-split"), args.output / (backend + "-whole")
        launch(split, backend, 4, 4)
        before = {p: p.read_bytes() for p in split.glob("shard-*/*")}
        (split / ".pending-test").mkdir()
        (split / ".pending-test" / "samples.bin").write_bytes(b"partial")
        launch(split, backend, 4, 8)
        assert all(p.read_bytes() == content for p, content in before.items())
        launch(whole, backend, 4, 8)
        a, b = payload(split), payload(whole)
        assert a.tobytes() == b.tobytes(), backend + " cohort resume changed replay"
        passes = oracle(a)
        assert len(np.unique(a["game_id"])) == 8
        no_work = launch(split, backend, 4, 8)
        assert json.loads(no_work.stdout)["new_games"] == 0
        launch(split, backend, 8, 16, success=False)
        launch(split, backend, 4, 12, "--seed", "42", success=False)
        launch(args.output / (backend + "-partial"), backend, 4, 5, success=False)
        launch(args.output / (backend + "-capacity"), backend, 4, 4, "--max-mib", "1", success=False)
        launch(args.output / (backend + "-memory"), backend, 4, 4, "--worker-mib", "1", success=False)
        path = next(split.glob("shard-*/samples.bin"))
        with path.open("r+b") as file: file.write(b"CORRUPT!")
        launch(split, backend, 4, 8, success=False)
        results[backend] = {"resume": "byte-identical", "samples": len(a), "passes": passes}
        print(backend + " resume/oracle/rejection tests passed", flush=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("BATCHED SELFPLAY TESTS PASSED")


if __name__ == "__main__": main()
