"""Export a trusted checkpoint or explicitly untrained initialization to single-file ONNX."""
import argparse
import hashlib
import json
from pathlib import Path

import onnx
import torch

from model import ARCHITECTURE, ENCODING, PARAMETERS, POLICY, PolicyValueNet


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != 1 or checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("unsupported checkpoint schema/architecture")
    steps = checkpoint.get("training_steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("invalid training_steps")
    model = PolicyValueNet()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if not all(torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
        raise ValueError("non-finite checkpoint")
    return model.eval(), steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--init-random", action="store_true")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--output", type=Path, required=True, help="fresh artifact directory")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.checkpoint
    if args.init_random:
        model = PolicyValueNet()
        checkpoint_path = args.output / "random-init.pt"
        torch.save({"schema": 1, "architecture": ARCHITECTURE, "training_steps": 0,
                    "state_dict": model.state_dict()}, checkpoint_path)
    model, steps = load_checkpoint(checkpoint_path)
    count = sum(p.numel() for p in model.parameters())
    assert count == PARAMETERS, count
    identity = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    metadata = {"islay.arch": ARCHITECTURE, "islay.encoding": ENCODING, "islay.policy": POLICY,
                "islay.value": "relative-wdl-v1", "islay.checkpoint_sha256": identity,
                "islay.training_steps": str(steps)}
    path = args.output / "model.onnx"
    torch.onnx.export(model, (torch.zeros(2, 2, 8, 8),), str(path), input_names=["board"],
                      output_names=["policy_logits", "value"], opset_version=18, dynamo=True,
                      dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=128)},),
                      external_data=False)
    graph = onnx.load(path)
    onnx.helper.set_model_props(graph, metadata)
    onnx.checker.check_model(graph, full_check=True)
    onnx.save_model(graph, path, save_as_external_data=False)
    manifest = {"metadata": metadata, "parameters": count, "torch": str(torch.__version__),
                "onnx": onnx.__version__, "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "checkpoint": str(checkpoint_path.resolve()), "status": "untrained" if steps == 0 else "trained"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
