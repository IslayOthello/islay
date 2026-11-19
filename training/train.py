#!/usr/bin/env python3
"""Train an Islay b8c64-v2 network from a raw replay .bin file."""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from model import ARCHITECTURE, ENCODING, PARAMETERS, POLICY, PolicyValueNet, policy_value_terms
from replay import DTYPE, RECORD_BYTES, decode


def split_games(records):
    rng = random.Random(20260909)
    train_indices = []
    validation_indices = []
    for game_id in np.unique(records["game_id"]):
        indices = np.flatnonzero(records["game_id"] == game_id)
        indices = indices[np.argsort(records["ply"][indices])]
        destination = validation_indices if rng.randrange(10) == 0 else train_indices
        destination.extend(indices.tolist())
    if not train_indices or not validation_indices:
        raise ValueError("dataset has no train/validation split; generate more games")
    return np.asarray(train_indices, dtype=np.int64), np.asarray(validation_indices, dtype=np.int64)


def export_onnx(model, training_steps, output_path):
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("ONNX export requires the 'onnx' package") from error

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".onnx":
        raise ValueError("--output-network must end in .onnx")
    if output_path.exists():
        raise ValueError("network output already exists: " + str(output_path))

    model_cpu = model.cpu().eval()
    torch.onnx.export(
        model_cpu,
        torch.zeros(2, 2, 8, 8),
        str(output_path),
        input_names=["board"],
        output_names=["policy_logits", "wdl", "margin"],
        opset_version=18,
        dynamo=True,
        dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=128)},),
        external_data=False,
    )

    graph = onnx.load(str(output_path))
    state = model_cpu.state_dict()
    identity = hashlib.sha256()
    for name in sorted(state):
        identity.update(name.encode())
        identity.update(state[name].detach().cpu().numpy().tobytes())

    metadata = {
        "islay.arch": ARCHITECTURE,
        "islay.encoding": ENCODING,
        "islay.policy": POLICY,
        "islay.value": "relative-wdl-margin-v2",
        "islay.checkpoint_sha256": identity.hexdigest(),
        "islay.training_steps": str(training_steps),
    }
    onnx.helper.set_model_props(graph, metadata)
    onnx.checker.check_model(graph, full_check=True)
    onnx.save_model(graph, str(output_path), save_as_external_data=False)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", "--replay", dest="data", type=Path, required=True,
                        help="raw replay .bin file")
    parser.add_argument("--epochs", type=int, required=True, help="full passes over training samples")
    parser.add_argument("--batch", "--batch-size", dest="batch", type=int, default=256,
                        help="mini-batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=0.01, help="initial SGD learning rate (default: 0.01)")
    parser.add_argument("--output-network", "--output-name", dest="output_network", type=Path,
                        default=Path("model.onnx"), help="output ONNX network")
    args = parser.parse_args()

    if args.epochs < 1 or args.epochs > 1000000:
        parser.error("--epochs must be in 1..1000000")
    if args.batch < 1 or args.batch > 65536:
        parser.error("--batch must be in 1..65536")
    if not math.isfinite(args.lr) or args.lr <= 0 or args.lr > 1:
        parser.error("--lr must be finite and in (0,1]")
    path = args.data
    if not path.is_file():
        parser.error("--data file not found: " + str(path))
    size = path.stat().st_size
    if size == 0 or size % RECORD_BYTES:
        parser.error("--data must be a non-empty replay .bin divisible by 560 bytes")

    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    device = torch.device(device_name)

    records = np.memmap(path, dtype=DTYPE, mode="r")
    if np.any(records["value_format"] > 1):
        parser.error("dataset contains unknown replay value format")
    train_indices, validation_indices = split_games(records)
    batch_size = args.batch
    steps_per_epoch = (len(train_indices) + batch_size - 1) // batch_size
    target_steps = args.epochs * steps_per_epoch
    seed = 20260910
    random.seed(seed)
    np_rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device_name != "mps":
        torch.use_deterministic_algorithms(True)

    model = PolicyValueNet().to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=0.0001, foreach=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[target_steps // 3, 2 * target_steps // 3], gamma=0.1)

    print(json.dumps({
        "device": device_name,
        "data": str(path),
        "samples": len(records),
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch": batch_size,
        "steps": target_steps,
    }), flush=True)

    started = time.perf_counter()
    for step in range(1, target_steps + 1):
        selected = np_rng.choice(train_indices, size=batch_size, replace=True)
        symmetries = np_rng.integers(0, 8, size=batch_size)
        batch = decode(records[selected], symmetries)

        board = torch.from_numpy(batch["board"]).to(device)
        policy = torch.from_numpy(batch["policy"]).to(device)
        wdl = torch.from_numpy(batch["wdl"]).to(device)
        margin = torch.from_numpy(batch["margin"]).to(device)
        legal = torch.from_numpy(batch["legal"]).to(device)

        optimizer.zero_grad(set_to_none=True)
        pred_policy, pred_wdl, pred_margin = model(board)
        policy_loss, wdl_loss, margin_loss, entropy = policy_value_terms(
            pred_policy, pred_wdl, pred_margin, policy, wdl, margin, legal)
        loss = policy_loss.mean() + wdl_loss.mean() + 0.25 * margin_loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 10 == 0 or step == target_steps:
            elapsed = time.perf_counter() - started
            speed = step / elapsed
            current_epoch = (step - 1) // steps_per_epoch + 1
            pct = 100.0 * step / target_steps
            print(
                f"[Epoch {current_epoch}/{args.epochs} | Step {step}/{target_steps} ({pct:.1f}%)] "
                f"Loss: {float(loss.detach()):.4f} (Pol: {float(policy_loss.mean().detach()):.4f}, "
                f"WDL: {float(wdl_loss.mean().detach()):.4f}, Margin: {float(margin_loss.mean().detach()):.4f}) | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f} | Speed: {speed:.1f} steps/s",
                flush=True,
            )

    metadata = export_onnx(model, target_steps, args.output_network)
    print(json.dumps({
        "network": str(Path(args.output_network).resolve()),
        "training_steps": target_steps,
        "metadata": metadata,
    }), flush=True)


if __name__ == "__main__":
    main()
