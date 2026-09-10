"""Warm-cache model-byte reading versus ORT session construction; shm is not a shared session."""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import onnxruntime as ort

from shm_region import Region
from replay import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.model.is_file() or not 0 < args.model.stat().st_size <= 64 * 1048576:
        parser.error("expected a single ONNX file of at most 64 MiB")
    args.output.mkdir(parents=True, exist_ok=False)
    data = args.model.read_bytes()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    board = np.zeros((1, 2, 8, 8), np.float32)
    rounds = []
    with Region(len(data)) as region:
        region.map[:] = data
        for round_index in range(8):
            row = {}
            for source in (("file", "shm") if round_index % 2 == 0 else ("shm", "file")):
                reads, loads = [], []
                for _ in range(3):
                    start = time.perf_counter_ns()
                    content = args.model.read_bytes() if source == "file" else region.map[:]
                    read_us = (time.perf_counter_ns() - start) / 1000
                    assert content == data
                    start = time.perf_counter_ns()
                    session = ort.InferenceSession(content, options, providers=["CPUExecutionProvider"])
                    session.run(None, {"board": board})
                    load_us = (time.perf_counter_ns() - start) / 1000
                    del session
                    reads.append(read_us)
                    loads.append(read_us + load_us)
                row[source] = {"read_us": statistics.median(reads), "read_session_warmup_us": statistics.median(loads)}
            if round_index: rounds.append(row)
    result = {"model_sha256": sha256(args.model), "bytes": len(data), "ort": ort.__version__, "rounds": rounds,
              "paired_load_speedup": statistics.median(r["file"]["read_session_warmup_us"] /
                                                        r["shm"]["read_session_warmup_us"] for r in rounds)}
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
