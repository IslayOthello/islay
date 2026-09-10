"""Hard-kill a private shm producer cohort; require exact full-cohort resume and lock exclusion."""
import argparse
import fcntl
from pathlib import Path
import os
import signal
import subprocess
import sys
import tempfile
import time

from batched_selfplay_test import payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worker", "verifier", "model", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--backend", choices=("ort-cpu", "torch-mps"), default="ort-cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output / "interrupted"
    common = [sys.executable, str(Path(__file__).with_name("batched_selfplay.py")), "--worker", str(args.worker),
              "--verifier", str(args.verifier), "--model", str(args.model), "--checkpoint", str(args.checkpoint),
              "--backend", args.backend, "--workers", "4", "--simulations", "8"]
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen([*common, "--output", str(root), "--games", "128"], stdout=log, stderr=log,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 60
            while not (root / "shard-000000000000").exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    log.seek(0)
                    raise AssertionError(log.read().decode())
                time.sleep(0.01)
            # Only this test's orchestrator/C++ workers belong to this process group.
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    before = {p: p.read_bytes() for p in root.glob("shard-*/*")}
    assert before and not (root / "shard-000000000004").exists()
    with (root / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run([*common, "--output", str(root), "--games", "8"], capture_output=True, timeout=60)
        assert result.returncode != 0 and b"BlockingIOError" in result.stderr
    for target in (root, args.output / "continuous"):
        result = subprocess.run([*common, "--output", str(target), "--games", "8"], text=True, capture_output=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    assert all(p.read_bytes() == value for p, value in before.items())
    assert payload(root).tobytes() == payload(args.output / "continuous").tobytes()
    print("SHM HARD-KILL / LOCK / BYTE-EXACT RESUME PASSED: " + args.backend)


if __name__ == "__main__": main()
