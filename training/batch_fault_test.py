"""Private worker IPC faults; test evaluators never publish training datasets."""
import argparse
from pathlib import Path
import signal

import numpy as np

from batch_workers import Workers


class TestBackend:
    def __init__(self, invalid=False): self.invalid = invalid
    def forward(self, inputs):
        if self.invalid == "raise": raise ValueError("injected model failure")
        return np.full((len(inputs), 66), np.nan if self.invalid else 0, np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    config = {"workers": 2, "seed": 7, "simulations": 4, "tree_mib": 1, "cache_mib": 0,
              "epsilon": 0.25, "alpha": 0.3, "temperature_plies": 16, "timeout_ms": 30000}
    for fault in ("nan", "raise", "crash", "stall", "server-close"):
        workers = Workers(args.worker.resolve(), TestBackend("raise" if fault == "raise" else fault == "nan"), dict(config))
        try:
            try:
                if fault == "crash": workers.processes[0].kill()
                if fault == "stall":
                    workers.processes[0].send_signal(signal.SIGSTOP)
                    workers.config["timeout_ms"] = 200
                if fault == "server-close":
                    workers.sockets[0].sendall(b"G" + bytes(8))
                    assert workers.sockets[0].recv(1) == b"R"
                    workers.sockets[0].close()
                    assert workers.processes[0].wait(timeout=5) != 0
                    raise RuntimeError("expected server-close exit")
                workers.cohort(0)
            except (RuntimeError, ValueError, TimeoutError, OSError):
                if fault == "stall": workers.processes[0].send_signal(signal.SIGCONT)
            else: raise AssertionError("worker fault accepted: " + fault)
        finally:
            diagnostic = workers.diagnostics()
            workers.close(abort=True)
            assert "AddressSanitizer" not in diagnostic and "runtime error:" not in diagnostic
        print(fault + " rejected", flush=True)
    print("BATCH IPC FAULT TESTS PASSED")


if __name__ == "__main__": main()
