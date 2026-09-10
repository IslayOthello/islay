"""Paired full-update FP32 training benchmark and CPU/MPS/layout numerical checks."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np
import torch

import checkpoint
from replay import decode, sha256
from train import DEFAULT_CONFIG, Trainer
from train_data import TrainingReplay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.rounds < 3 or args.repeats < 3: parser.error("need at least three rounds/repeats")
    args.output.mkdir(parents=True, exist_ok=False)
    data = TrainingReplay([args.replay], args.verifier)
    config = json.loads(DEFAULT_CONFIG.read_text())
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    errors, rejected, supported = [], [], []
    fixed = decode(data.gather(data.indices["train"][:32]))
    reference = Trainer(data, {**config, "device": "cpu", "memory_format": "contiguous"}, initial=args.init)
    reference.step(fixed)
    expected = checkpoint.cpu_tree(reference.model.state_dict())
    for device in devices:
        for layout in ("contiguous", "channels_last"):
            try:
                model = Trainer(data, {**config, "device": device, "memory_format": layout}, initial=args.init)
                model.step(fixed)
            except (ValueError, RuntimeError) as failure:
                if layout == "contiguous": raise
                rejected.append({"device": device, "layout": layout, "reason": str(failure)})
                continue
            supported.append((device, layout))
            actual = checkpoint.cpu_tree(model.model.state_dict())
            maximum = 0
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=1e-3, atol=2e-4)
                maximum = max(maximum, float((actual[key] - expected[key]).abs().max()))
            errors.append({"device": device, "layout": layout, "one_update_max_abs_error": maximum})
    print("CPU/MPS/layout one-update parameter/BatchNorm parity passed", flush=True)
    results = []
    for batch in (32, 128):
        pairs = []
        for round_index in range(args.rounds):
            row = {}
            variants = supported.copy()
            if round_index % 2: variants.reverse()
            for device, layout in variants:
                trainer = Trainer(data, {**config, "batch_size": batch, "device": device, "memory_format": layout},
                                  initial=args.init)
                for _ in range(3): trainer.step()
                samples = [trainer.step()["step_us"] for _ in range(args.repeats)]
                row[device + "/" + layout] = {"median_us": statistics.median(samples),
                                               "samples_per_second": batch * 1e6 / statistics.median(samples)}
            pairs.append(row)
            print(f"batch {batch} round {round_index + 1} complete", flush=True)
        gains = {}
        for device in devices:
            if (device, "channels_last") not in supported: continue
            gains[device + "_layout_gain_percent"] = statistics.median(
                (row[device + "/contiguous"]["median_us"] / row[device + "/channels_last"]["median_us"] - 1) * 100
                for row in pairs)
        if "mps" in devices:
            gains["mps_vs_cpu_contiguous_speedup"] = statistics.median(
                row["cpu/contiguous"]["median_us"] / row["mps/contiguous"]["median_us"] for row in pairs)
        results.append({"batch": batch, "pairs": pairs, "gains": gains})
    summary = {"initial_sha256": sha256(args.init), "data_identity": data.identity, "torch": str(torch.__version__),
               "rounds": args.rounds, "repeats": args.repeats, "parity": errors, "rejected": rejected, "results": results}
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
