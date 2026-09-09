"""Python/ONNX/C++ parity, reload/error/stop checks and batch latency measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import onnx
import torch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mcts_study import corpus
from uci_search_test import Engine, fen, search
from export import load_checkpoint
from model import ARCHITECTURE, encode, policy_value_loss, transform_action


def transformed(bits, symmetry):
    return sum(1 << transform_action(sq, symmetry) for sq in range(64) if bits >> sq & 1)


def probe(binary, model, boards, repeats=1, optimization="all"):
    inputs = "".join(f"{p} {o}\n" for p, o in boards)
    result = subprocess.run([str(binary), str(model), str(repeats), optimization], input=inputs,
                            text=True, capture_output=True, check=True, timeout=120)
    return json.loads(result.stdout)


def statistics_us(samples, batch):
    median = statistics.median(samples)
    return {"median_us": median, "p95_us": sorted(samples)[int((len(samples) - 1) * 0.95)],
            "positions_per_second": batch * 1e6 / median}


def protocol(binary, model, invalid):
    engine = Engine(binary)
    latencies = []
    try:
        for _ in range(2):
            engine.send(f"setoption name EvalFile value {model}")
            lines = engine.barrier()
            assert any("option EvalFile =" in line for line in lines), lines
            engine.send("position startpos\ngo nodes 8")
            lines = engine.until("bestmove ")
            assert any("onnx-cpu b8c64-v1" in line for line in lines), lines
            assert any(line.startswith("info nodes 8 ") for line in lines), lines
            engine.barrier()
        for position in corpus():
            search(engine, (position["p"], position["o"]), "go nodes 16", 16, evaluator="onnx-cpu b8c64-v1")
        search(engine, (2, 1), "go nodes 8", 8, "pass", evaluator="onnx-cpu b8c64-v1")
        engine.send("go infinite")
        engine.barrier()
        engine.send(f"setoption name EvalFile value {invalid}")
        lines = engine.barrier()
        assert any("previous evaluator retained" in line for line in lines), lines
        assert not any(line.startswith("bestmove ") for line in lines), lines
        engine.send("go nodes 1")
        lines = engine.until("bestmove ")
        assert any("onnx-cpu b8c64-v1" in line for line in lines), lines
        for _ in range(20):
            engine.send("position startpos\ngo infinite")
            engine.barrier()
            # Allow inference to begin; cancellation is not restricted to pre-start jobs.
            time.sleep(0.002)
            start = time.perf_counter_ns()
            engine.send("stop\nisready")
            lines = engine.until("readyok")
            latencies.append((time.perf_counter_ns() - start) / 1000)
            assert sum(line.startswith("bestmove ") for line in lines) == 1, lines
            assert not any("search error" in line for line in lines), lines
        search(engine, (1 << 28 | 1 << 35, 1 << 27 | 1 << 36), "go movetime 20",
               evaluator="onnx-cpu b8c64-v1")
        engine.send("go infinite")
        engine.barrier()
        engine.send("setoption name EvalFile value <empty>\ngo nodes 1")
        lines = engine.until("bestmove ")
        assert any("no trained network" in line for line in lines), lines
        engine.barrier()
        engine.send(f"setoption name EvalFile value {model}\nposition fen {fen(1, 0)} X\ngo nodes 8")
        lines = engine.until("bestmove ")
        assert lines[-1] == "bestmove 0000" and any("evaluations 0" in line for line in lines), lines
        engine.send("position startpos\ngo perft 8")
        assert "Nodes searched: 390,216" in engine.barrier()
        engine.close()
    finally:
        engine.abort()
    measured = statistics_us(latencies, 1)
    del measured["positions_per_second"]
    return measured


def updated_checkpoint_parity(args, boards):
    """Exercise non-identity BatchNorm and the checkpoint export path; not strength training."""
    model, _ = load_checkpoint(args.checkpoint)
    model.train()
    inputs = encode(boards[:8])
    legal = torch.ones(8, 65, dtype=torch.bool)
    target = legal.float() / 65
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        p, v = model(inputs)
        policy_value_loss(p, v, target, torch.zeros(8, 1), legal).backward()
        optimizer.step()
    path = args.output / "synthetic-smoke.pt"
    torch.save({"schema": 1, "architecture": ARCHITECTURE, "training_steps": 4,
                "state_dict": model.state_dict()}, path)
    exported = args.output / "synthetic-export"
    subprocess.run([sys.executable, str(Path(__file__).with_name("export.py")), "--checkpoint", str(path),
                    "--output", str(exported)], check=True, capture_output=True, text=True, timeout=120)
    model.eval()
    error = 0.0
    with torch.inference_mode():
        for size in (1, 8, 32, 128):
            expected = torch.cat(model(encode(boards[:size])), dim=1).numpy()
            actual = np.asarray(probe(args.probe, exported / "model.onnx", boards[:size])["outputs"])
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)
            error = max(error, float(np.max(np.abs(actual - expected))))
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=31)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("need at least three repeats")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model, _ = load_checkpoint(args.checkpoint)
    boards = [(transformed(p["p"], s), transformed(p["o"], s)) for p in corpus() for s in range(8)]
    boards += [(1 << sq, 0) for sq in range(64)] + [(0, 1 << sq) for sq in range(64)]
    max_error = 0.0
    with torch.inference_mode():
        batches = [boards[i:i + 32] for i in range(0, len(boards), 32)]
        batches += [boards[:size] for size in (1, 8, 32, 128)]
        for batch in batches:
            p, v = model(encode(batch))
            expected = torch.cat((p, v), dim=1).numpy()
            actual = np.asarray(probe(args.probe, args.model, batch)["outputs"])
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    print(f"Parity passed: {len(boards)} positions, max abs error {max_error:.3g}", flush=True)
    updated_error = updated_checkpoint_parity(args, boards)
    print(f"Updated checkpoint/BatchNorm parity passed: {updated_error:.3g}", flush=True)

    rejected = []
    for kind in ("metadata", "shape", "nan", "corrupt", "missing"):
        path = args.output / f"invalid-{kind}.onnx"
        if kind in ("metadata", "shape", "nan"):
            graph = onnx.load(args.model)
            if kind == "metadata":
                metadata = {p.key: p.value for p in graph.metadata_props}
                metadata["islay.encoding"] = "wrong-encoding"
                onnx.helper.set_model_props(graph, metadata)
            elif kind == "shape":
                graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param = ""
                graph.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
            else:
                tensor = graph.graph.initializer[0]
                values = onnx.numpy_helper.to_array(tensor).copy()
                values.fill(np.nan)
                tensor.CopyFrom(onnx.numpy_helper.from_array(values, tensor.name))
            onnx.save(graph, path)
        elif kind == "corrupt":
            path.write_bytes(b"invalid ONNX")
        attempt = subprocess.run([str(args.probe), str(path)], input="1 2\n", text=True,
                                 capture_output=True, timeout=30)
        assert attempt.returncode != 0, kind
        rejected.append(kind)

    stop = protocol(args.engine.resolve(), args.model.resolve(), (args.output / "invalid-metadata.onnx").resolve())
    results = []
    with torch.inference_mode():
        for batch in (1, 8, 32, 128):
            inputs = encode(boards[:batch])
            for device in ("cpu", "mps"):
                if device == "mps" and not torch.backends.mps.is_available():
                    continue
                net = model.to(device)
                tensor = inputs.to(device)
                for _ in range(3):
                    net(tensor)
                if device == "mps":
                    torch.mps.synchronize()
                times = []
                for _ in range(args.repeats):
                    start = time.perf_counter_ns()
                    result = net(tensor)
                    if device == "mps":
                        torch.mps.synchronize()
                    times.append((time.perf_counter_ns() - start) / 1000)
                if device == "mps":
                    actual = torch.cat(result, dim=1).cpu().numpy()
                    model.cpu()
                    expected = torch.cat(model(inputs), dim=1).numpy()
                    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
                results.append({"backend": "torch-" + device, "batch": batch, **statistics_us(times, batch)})
            for optimization in ("off", "all"):
                measured = probe(args.probe, args.model, boards[:batch], args.repeats, optimization)
                model.cpu()
                expected = torch.cat(model(inputs), dim=1).numpy()
                np.testing.assert_allclose(measured.pop("outputs"), expected, rtol=1e-4, atol=2e-5)
                measured["positions_per_second"] = batch * 1e6 / measured["median_us"]
                results.append({"backend": "ort-cpp-" + optimization, "batch": batch, **measured})
            print(f"batch {batch} measured", flush=True)
    paired = []
    for round_index in range(5):
        row = {}
        for optimization in (("off", "all") if round_index % 2 == 0 else ("all", "off")):
            row[optimization] = probe(args.probe, args.model, boards[:1], args.repeats, optimization)["median_us"]
        row["speedup_percent"] = (row["off"] / row["all"] - 1) * 100
        paired.append(row)
    summary = {"max_abs_error": max_error, "updated_checkpoint_max_abs_error": updated_error,
               "positions": len(boards), "rejected": rejected, "paired_batch1": paired,
               "stop_roundtrip_us": stop, "benchmark": results, "repeats": args.repeats,
               "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest()}
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
