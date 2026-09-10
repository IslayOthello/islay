"""FP32 policy/value training with frozen replay, full-state resume and atomic checkpoints."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

import checkpoint
from export import load_checkpoint
from model import ARCHITECTURE, policy_value_terms
from replay import decode, sha256
from selfplay import write_json
from train_data import TrainingReplay, digest

DEFAULT_CONFIG = Path(__file__).with_name("config_b.json")


def validate_config(config):
    expected = set(json.loads(DEFAULT_CONFIG.read_text()))
    if set(config) != expected or config["schema"] != 1:
        raise ValueError("unknown training config fields/schema")
    for name, low, high in (("batch_size", 1, 1024), ("seed", 0, (1 << 63) - 1), ("max_games", 1, 1000000),
                            ("max_replay_mib", 1, 65536), ("cpu_threads", 1, 256), ("checkpoint_every", 1, 1000000)):
        if type(config[name]) is not int or not low <= config[name] <= high:
            raise ValueError("invalid " + name)
    for name, low, high in (("learning_rate", 1e-8, 1), ("momentum", 0, 0.9999), ("weight_decay", 0, 1),
                            ("lr_gamma", 1e-6, 1), ("grad_clip", 1e-6, 1000000)):
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("invalid " + name)
    milestones = config["milestones"]
    if (not isinstance(milestones, list) or any(type(n) is not int or n < 1 for n in milestones)
            or milestones != sorted(set(milestones))):
        raise ValueError("milestones must be distinct increasing update counts")
    if (config["device"] not in ("cpu", "mps", "cuda") or config["memory_format"] not in ("contiguous", "channels_last")
            or type(config["augment"]) is not bool):
        raise ValueError("invalid device/layout/augmentation")
    if config["device"] == "mps" and config["memory_format"] == "channels_last":
        raise ValueError("MPS channels_last backward fails on pinned PyTorch; use contiguous")
    return config


def configure(config):
    validate_config(config)
    if config["device"] == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
            raise ValueError("deterministic CUDA requires a supported CUBLAS_WORKSPACE_CONFIG")
    if config["device"] == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])


def runtime(config):
    directory = Path(__file__).parent
    code = {name: sha256(directory / name) for name in ("train.py", "train_data.py", "checkpoint.py", "model.py",
                                                       "replay.py", "selfplay.py", "export.py")}
    return {"torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.platform(),
            "device": config["device"], "cuda": torch.version.cuda, "code": code,
            "cuda_device": torch.cuda.get_device_name() if config["device"] == "cuda" else None}


def synchronize(device):
    if device == "mps": torch.mps.synchronize()
    if device == "cuda": torch.cuda.synchronize()


class Trainer:
    def __init__(self, data, config, initial=None, state=None):
        configure(config)
        self.data, self.config = data, config
        self.device = config["device"]
        self.rng = np.random.default_rng(config["seed"])
        self.runtime = runtime(config)
        if state is None:
            if initial is None: raise ValueError("initial checkpoint required")
            self.model, self.parent_steps = load_checkpoint(initial)
            self.initial_sha = sha256(initial)
            self.updates = 0
        else:
            # Optimizer loading can otherwise alias CPU momentum tensors in the caller's state.
            state = checkpoint.cpu_tree(state)
            if (state.get("trainer_schema") != "islay-trainer-v1" or state.get("config") != config
                    or state.get("runtime") != self.runtime or state.get("data") != data.snapshot
                    or state.get("schema") != 1 or state.get("architecture") != ARCHITECTURE
                    or type(state.get("updates")) is not int or state["updates"] < 0
                    or state.get("training_steps") != state.get("parent_steps", -1) + state["updates"]):
                raise ValueError("resume requires identical runtime/code/config/replay and valid counters")
            from model import PolicyValueNet
            self.model = PolicyValueNet()
            self.model.load_state_dict(state["state_dict"], strict=True)
            if not checkpoint.finite_tree(state["state_dict"]) or not checkpoint.finite_tree(state["optimizer"]):
                raise ValueError("non-finite resume state")
            self.parent_steps, self.updates, self.initial_sha = state["parent_steps"], state["updates"], state["initial_sha256"]
        self.format = torch.channels_last if config["memory_format"] == "channels_last" else torch.contiguous_format
        self.model.to(device=self.device, memory_format=self.format).train()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config["learning_rate"], momentum=config["momentum"],
                                         weight_decay=config["weight_decay"], foreach=False)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config["milestones"],
                                                              gamma=config["lr_gamma"])
        if state is not None:
            self.scheduler.load_state_dict(state["scheduler"])
            self.optimizer.load_state_dict(state["optimizer"])
            if self.scheduler.last_epoch != self.updates:
                raise ValueError("scheduler/update mismatch")
            expected_lr = config["learning_rate"] * config["lr_gamma"] ** sum(m <= self.updates for m in config["milestones"])
            if not math.isclose(self.optimizer.param_groups[0]["lr"], expected_lr, rel_tol=1e-12):
                raise ValueError("optimizer learning rate/schedule mismatch")
            random.setstate(state["rng"]["python"])
            self.rng.bit_generator.state = state["rng"]["numpy"]
            torch.set_rng_state(state["rng"]["torch_cpu"])
            if self.device == "mps": torch.mps.set_rng_state(state["rng"]["accelerator"])
            if self.device == "cuda": torch.cuda.set_rng_state_all(state["rng"]["accelerator"])

    def tensors(self, batch):
        result = {key: torch.from_numpy(batch[key]).to(self.device) for key in ("legal", "policy", "outcome")}
        result["board"] = torch.from_numpy(batch["board"]).to(device=self.device, memory_format=self.format)
        return result

    def step(self, batch=None):
        started = time.perf_counter_ns()
        if batch is None:
            batch = self.data.sample(self.config["batch_size"], self.rng, self.config["augment"])
        tensors = self.tensors(batch)
        self.optimizer.zero_grad(set_to_none=True)
        p, v = self.model(tensors["board"])
        policy, value, entropy = policy_value_terms(p, v, tensors["policy"], tensors["outcome"], tensors["legal"])
        loss = policy.mean() + value.mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"],
                                              error_if_nonfinite=True, foreach=False)
        metrics = torch.stack((policy.mean().detach(), value.mean().detach(), entropy.mean().detach(),
                               v.detach().abs().mean(), norm.detach(), (p.isfinite().all() & v.isfinite().all()).float())).cpu()
        if not torch.isfinite(metrics).all() or metrics[-1] != 1:
            raise ValueError("non-finite training output/loss/gradient")
        lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.step()
        self.scheduler.step()
        self.updates += 1
        synchronize(self.device)
        elapsed_us = (time.perf_counter_ns() - started) / 1000
        return {"update": self.updates, "training_steps": self.parent_steps + self.updates, "lr": lr,
                "policy_loss": float(metrics[0]), "value_loss": float(metrics[1]), "entropy": float(metrics[2]),
                "mean_abs_value": float(metrics[3]), "grad_norm": float(metrics[4]), "step_us": elapsed_us,
                "samples_per_second": len(batch["board"]) * 1e6 / elapsed_us}

    def validate(self):
        self.model.eval()
        sums = np.zeros(3, dtype=np.float64)
        phases = {name: {"sum": np.zeros(3, dtype=np.float64), "samples": 0} for name in ("opening", "middle", "end")}
        count = 0
        try:
            with torch.inference_mode():
                indices = self.data.indices["validation"]
                for start in range(0, len(indices), self.config["batch_size"]):
                    batch = decode(self.data.gather(indices[start:start + self.config["batch_size"]]))
                    t = self.tensors(batch)
                    p, v = self.model(t["board"])
                    if not bool((p.isfinite().all() & v.isfinite().all()).cpu()):
                        raise ValueError("non-finite validation output")
                    terms = policy_value_terms(p, v, t["policy"], t["outcome"], t["legal"])
                    values = torch.stack(terms, dim=1).cpu().numpy()
                    if not np.isfinite(values).all(): raise ValueError("non-finite validation loss")
                    sums += values.sum(axis=0, dtype=np.float64)
                    occupied = batch["board"].sum(axis=(1, 2, 3))
                    for name, mask in (("opening", occupied < 24), ("middle", (occupied >= 24) & (occupied < 44)),
                                       ("end", occupied >= 44)):
                        phases[name]["sum"] += values[mask].sum(axis=0, dtype=np.float64)
                        phases[name]["samples"] += int(mask.sum())
                    count += len(batch["board"])
        finally:
            self.model.train()
        names = ("policy_loss", "value_loss", "entropy")
        phase_metrics = {name: {"samples": row["samples"], **dict(zip(names, (row["sum"] / row["samples"]).tolist()))}
                         for name, row in phases.items() if row["samples"]}
        return dict(zip(names, (sums / count).tolist())) | {"samples": count, "phases": phase_metrics}

    def state(self):
        accelerator = None
        if self.device == "mps": accelerator = torch.mps.get_rng_state()
        if self.device == "cuda": accelerator = torch.cuda.get_rng_state_all()
        return {"schema": 1, "architecture": ARCHITECTURE, "trainer_schema": "islay-trainer-v1",
                "state_dict": self.model.state_dict(), "training_steps": self.parent_steps + self.updates,
                "parent_steps": self.parent_steps, "updates": self.updates, "initial_sha256": self.initial_sha,
                "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "config": self.config, "runtime": self.runtime, "data": self.data.snapshot,
                "rng": {"python": random.getstate(), "numpy": self.rng.bit_generator.state,
                        "torch_cpu": torch.get_rng_state(), "accelerator": accelerator}}

    def run_manifest(self):
        return {"schema": "islay-training-run-v1", "config": self.config, "runtime": self.runtime,
                "data_identity": self.data.identity, "statistics": self.data.statistics,
                "initial_sha256": self.initial_sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="append", type=Path, required=True, help="oldest to newest; repeatable")
    parser.add_argument("--verifier", type=Path, required=True, help="islay-selfplay binary for full replay checks")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--init", type=Path)
    source.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--memory-format", choices=("contiguous", "channels_last"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000, help="total updates in this run, not additional updates")
    parser.add_argument("--max-mib", type=int, default=1024, help="checkpoint storage cap; no automatic eviction")
    args = parser.parse_args()
    if not 0 <= args.steps <= 1000000000 or args.max_mib < 32: parser.error("invalid steps/storage cap")
    config = json.loads(args.config.read_text())
    if args.device: config["device"] = args.device
    if args.memory_format: config["memory_format"] = args.memory_format
    validate_config(config)
    data = TrainingReplay(args.replay, args.verifier, config["max_games"], config["max_replay_mib"])
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    with open(root / ".lock", "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            state, last = checkpoint.latest(root)
            trainer = Trainer(data, config, state=state)
            expected = json.loads((root / "run.json").read_text())
            if expected != trainer.run_manifest():
                raise ValueError("run manifest does not match checkpoint/data")
        else:
            trainer = Trainer(data, config, initial=args.init)
            write_json(root / "run.json", trainer.run_manifest())
            last = checkpoint.save(root, trainer.state(), {"steps": [], "validation": trainer.validate()}, args.max_mib)
        pending = []
        while trainer.updates < args.steps:
            row = trainer.step()
            pending.append(row)
            if trainer.updates % 10 == 0:
                print(json.dumps(row), flush=True)
            if trainer.updates % config["checkpoint_every"] == 0 or trainer.updates == args.steps:
                metrics = {"steps": pending, "validation": trainer.validate()}
                last = checkpoint.save(root, trainer.state(), metrics, args.max_mib)
                pending = []
                print(json.dumps({"checkpoint": str(last), "validation": metrics["validation"]}), flush=True)
        print(json.dumps({"checkpoint": str(last), "updates": trainer.updates,
                          "training_steps": trainer.parent_steps + trainer.updates}), flush=True)


if __name__ == "__main__":
    main()
