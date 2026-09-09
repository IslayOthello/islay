"""Kill an isolated producer process group mid-shard; verify lock and exact resume."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

from replay import Replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).with_name("selfplay.py")), "--engine", str(args.engine),
               "--model", str(args.model), "--output", str(args.output), "--simulations", "16", "--shard-games", "1"]
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen([*command, "--games", "100"], stdout=log, stderr=log, start_new_session=True)
        try:
            deadline = time.monotonic() + 30
            while not list(args.output.glob("shard-*/index.json")) or not list(args.output.glob(".pending-*")):
                if process.poll() is not None or time.monotonic() > deadline:
                    log.seek(0)
                    raise AssertionError(log.read().decode())
                time.sleep(0.01)
            # This group contains only the test wrapper and its C++ producer.
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    before = [(p, p.read_bytes()) for p in args.output.glob("shard-*/*")]
    assert before
    with open(args.output / ".lock", "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = subprocess.run([*command, "--games", "4"], capture_output=True, timeout=30)
        assert locked.returncode != 0
    resumed = subprocess.run([*command, "--games", "4"], capture_output=True, text=True, timeout=120)
    assert resumed.returncode == 0, resumed.stderr
    assert all(p.read_bytes() == data for p, data in before)
    recovered, reference = Replay(args.output), Replay(args.reference)
    actual = recovered.gather(np.arange(recovered.ends[-1]))
    expected = reference.gather(np.arange(reference.ends[-1]))
    expected = expected[expected["game_id"] < 4]
    assert actual.tobytes() == expected.tobytes()
    assert len(np.unique(actual["game_id"])) == 4
    print("HARD-KILL / LOCK / RESUME TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()
