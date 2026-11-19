"""Model identity and native ONNX parity gate, without training or timing benchmarks."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from arena import model_info
from arena_engine import Engine
from batch_inference import Backend
from export import load_checkpoint
from model import encode
from replay import sha256
from selfplay import write_json
from verify import corpus, probe, transformed


def check(checkpoint, model, binary, engine_binary, backend, workers):
    identity = model_info(model)
    if identity["checkpoint_sha256"] != sha256(checkpoint): raise ValueError("checkpoint/model mismatch")
    engine = Engine(engine_binary)
    try:
        engine.preflight()
        engine.configure(model, 8)
        row = engine.search((1 << 28 | 1 << 35, 1 << 27 | 1 << 36),
                            {"mode": "nodes", "budget": 1, "timeout_ms": 30000})
        evaluator_backend = row["evaluator"].split(" ")[3]
        if evaluator_backend not in ("onnx-cpu", "onnx-coreml"):
            raise ValueError("native engine used unsupported evaluator")
        expected = ("info string evaluator " + evaluator_backend + " b8c64-v2 checkpoint "
                    + identity["checkpoint_sha256"] + " training_steps " + str(identity["training_steps"]))
        if not identity["training_steps"]: expected += " (untrained initialization)"
        if row["evaluator"] != expected: raise ValueError("native engine used wrong evaluator")
        engine.close()
    finally:
        engine.close(abort=True)
    backend_check = Backend(checkpoint, model, backend, workers)
    network, steps = load_checkpoint(checkpoint)
    boards = [(transformed(p["p"], s), transformed(p["o"], s)) for p in corpus() for s in range(8)]
    boards += [(1 << sq, 0) for sq in range(64)] + [(0, 1 << sq) for sq in range(64)]
    batches = [boards[i:i + 32] for i in range(0, len(boards), 32)]
    batches += [boards[:n] for n in (1, 8, 32, 128)]
    error = 0.0
    with torch.inference_mode():
        for batch in batches:
            expected = torch.cat(network(encode(batch)), dim=1).numpy()
            actual = np.asarray(probe(binary, model, batch)["outputs"])
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)
            error = max(error, float(np.max(np.abs(actual - expected))))
    return {"model": identity, "training_steps": steps, "positions": len(boards),
            "max_abs_error": error, "backend": backend_check.identity, "startup": backend_check.startup,
            "native_preflight": {"selftests": "passed", "perft8": 390216, "evaluator": row["evaluator"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "model", "probe", "engine", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--backend", choices=("ort-cpu", "torch-cpu", "torch-mps"), required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    result = check(args.checkpoint, args.model, args.probe, args.engine, args.backend, args.workers)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__": main()
