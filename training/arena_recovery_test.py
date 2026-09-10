"""Kill the isolated arena process group; preserve pair commits and verify exact node-mode resume."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from arena_test import trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output / "interrupted"
    common = [sys.executable, str(Path(__file__).with_name("arena.py")), "--engine", str(args.engine),
              "--candidate", str(args.model), "--champion", str(args.model), "--pairs", "4", "--budget", "4"]
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen([*common, "--output", str(root)], stdout=log, stderr=log, start_new_session=True)
        try:
            deadline = time.monotonic() + 60
            while not (root / "pair-000000").is_dir():
                if process.poll() is not None or time.monotonic() > deadline:
                    log.seek(0)
                    raise AssertionError(log.read().decode())
                time.sleep(0.01)
            # The group contains only this test's orchestrator and its owned engines.
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    committed = {p: p.read_bytes() for p in root.glob("pair-*/*")}
    with open(root / ".lock", "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = subprocess.run([*common, "--output", str(root), "--resume"], text=True, capture_output=True, timeout=60)
        assert attempt.returncode != 0 and "BlockingIOError" in attempt.stderr
    for output, extra in ((root, ["--resume"]), (args.output / "continuous", [])):
        result = subprocess.run([*common, "--output", str(output), *extra], text=True, capture_output=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    assert all(p.read_bytes() == data for p, data in committed.items())
    assert trajectory(root) == trajectory(args.output / "continuous")
    print(json.dumps({"result": "ARENA HARD-KILL / LOCK / RESUME PASSED", "pairs": 4}))


if __name__ == "__main__":
    main()
