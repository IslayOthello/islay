"""Hard-kill a training process and verify exact CPU continuation from committed state."""
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

import checkpoint
from train import DEFAULT_CONFIG
from train_test import equal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(DEFAULT_CONFIG.read_text())
    config.update(batch_size=8, checkpoint_every=2, milestones=[4], device="cpu", memory_format="contiguous")
    cfg = args.output / "config.json"
    cfg.write_text(json.dumps(config))
    root = args.output / "interrupted"
    common = [sys.executable, str(Path(__file__).with_name("train.py")), "--replay", str(args.replay),
              "--verifier", str(args.verifier), "--config", str(cfg)]
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen([*common, "--output", str(root), "--init", str(args.init), "--steps", "100000"],
                                   stdout=log, stderr=log, start_new_session=True)
        try:
            deadline = time.monotonic() + 60
            while not (root / "step-000000002").is_dir():
                if process.poll() is not None or time.monotonic() > deadline:
                    log.seek(0)
                    raise AssertionError(log.read().decode())
                time.sleep(0.01)
            # This process group belongs solely to this isolated test run.
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    state, _ = checkpoint.latest(root)
    target = state["updates"] + 4
    committed = [(p, p.read_bytes()) for p in root.glob("step-*/*")]
    with open(root / ".lock", "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = subprocess.run([*common, "--output", str(root), "--resume", "--steps", str(target)],
                                 text=True, capture_output=True, timeout=60)
        assert attempt.returncode != 0 and "BlockingIOError" in attempt.stderr
    for output, source in ((root, ["--resume"]), (args.output / "continuous", ["--init", str(args.init)])):
        result = subprocess.run([*common, "--output", str(output), "--steps", str(target), *source],
                                text=True, capture_output=True, timeout=120)
        assert result.returncode == 0, result.stderr
    assert all(p.read_bytes() == data for p, data in committed)
    equal(checkpoint.latest(root)[0], checkpoint.latest(args.output / "continuous")[0])
    print(json.dumps({"result": "TRAINING HARD-KILL / LOCK / RESUME PASSED", "target_updates": target}))


if __name__ == "__main__":
    main()
