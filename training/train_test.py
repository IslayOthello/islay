"""Deterministic resume, validation isolation, overfit and atomic checkpoint rejection tests."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import checkpoint
from model import policy_value_loss, policy_value_terms
from replay import decode
from train import DEFAULT_CONFIG, Trainer, validate_config
from train_data import TrainingReplay


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b), "tensor mismatch"
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else:
        assert a == b, (a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--new-replay", type=Path, help="optional second generation for whole-game window checks")
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(DEFAULT_CONFIG.read_text())
    config.update(batch_size=16, checkpoint_every=2, milestones=[4], device="cpu", memory_format="contiguous")
    data = TrainingReplay([args.replay], args.verifier)
    assert not set(data.indices["train"]).intersection(data.indices["validation"])
    trainer = Trainer(data, config, initial=args.init)
    for _ in range(3): trainer.step()
    state_before_validation = checkpoint.cpu_tree(trainer.state())
    validation = trainer.validate()
    equal(state_before_validation, checkpoint.cpu_tree(trainer.state()))
    root = args.output / "checkpoint"
    root.mkdir()
    saved = checkpoint.save(root, trainer.state(), {"validation": validation})
    loaded, path = checkpoint.latest(root)
    assert path == saved
    iteration = Trainer(data, config, initial=saved)
    assert iteration.updates == 0 and iteration.parent_steps == loaded["training_steps"]
    assert not iteration.optimizer.state and iteration.scheduler.last_epoch == 0
    equal(iteration.model.state_dict(), loaded["state_dict"])
    # Restore global RNG state after constructing the independent iteration.
    trainer = Trainer(data, config, state=loaded)
    for _ in range(3): trainer.step()
    uninterrupted = checkpoint.cpu_tree(trainer.state())
    resumed = Trainer(data, config, state=loaded)
    for _ in range(3): resumed.step()
    equal(uninterrupted, checkpoint.cpu_tree(resumed.state()))
    assert resumed.optimizer.param_groups[0]["lr"] == config["learning_rate"] * config["lr_gamma"]
    print("CPU resume: model, optimizer, scheduler and all RNG states bit-identical", flush=True)

    # Full CLI path crosses a schedule boundary and an extra final checkpoint.
    cfg = args.output / "config.json"
    cfg.write_text(json.dumps(config))
    common = [sys.executable, str(Path(__file__).with_name("train.py")), "--replay", str(args.replay),
              "--verifier", str(args.verifier), "--config", str(cfg)]
    split = args.output / "cli-split"
    whole = args.output / "cli-whole"
    for destination, steps, source in ((split, 3, ["--init", str(args.init)]), (split, 6, ["--resume"]),
                                        (whole, 6, ["--init", str(args.init)])):
        r = subprocess.run([*common, "--output", str(destination), "--steps", str(steps), *source],
                           text=True, capture_output=True, timeout=120)
        assert r.returncode == 0, r.stderr
    equal(checkpoint.latest(split)[0], checkpoint.latest(whole)[0])
    print("CLI split/uninterrupted trajectory identical", flush=True)
    layout_config = {**config, "memory_format": "channels_last"}
    layout_trainer = Trainer(data, layout_config, initial=args.init)
    for _ in range(3): layout_trainer.step()
    layout_saved = checkpoint.cpu_tree(layout_trainer.state())
    for _ in range(3): layout_trainer.step()
    layout_expected = checkpoint.cpu_tree(layout_trainer.state())
    layout_resumed = Trainer(data, layout_config, state=layout_saved)
    for _ in range(3): layout_resumed.step()
    equal(layout_expected, checkpoint.cpu_tree(layout_resumed.state()))
    if args.new_replay:
        combined = TrainingReplay([args.replay, args.new_replay], args.verifier)
        window = TrainingReplay([args.replay, args.new_replay], args.verifier, max_games=40)
        assert window.snapshot["selected_games"] == combined.snapshot["selected_games"][-40:]
        assert {source for source, _ in window.snapshot["selected_games"]} == {0, 1}
        for owner, source in enumerate(combined.sources):
            base = int(combined.ends[owner - 1]) if owner else 0
            indices = np.array([0, int(source.ends[-1]) - 1])
            np.testing.assert_array_equal(combined.gather(base + indices), source.gather(indices))
        assert not set(window.indices["train"]).intersection(window.indices["validation"])
    try: TrainingReplay([args.replay], args.verifier, max_mib=1)
    except ValueError: pass
    else: raise AssertionError("ignored replay byte cap")

    for field, bad in (("config", {**config, "batch_size": 32}), ("runtime", {}), ("data", {})):
        invalid = copy.deepcopy(loaded)
        invalid[field] = bad
        try: Trainer(data, config, state=invalid)
        except ValueError: pass
        else: raise AssertionError("accepted resume " + field + " mismatch")
    for field in ("learning_rate", "weight_decay", "grad_clip"):
        invalid = dict(config)
        invalid[field] = float("nan")
        try: validate_config(invalid)
        except ValueError: pass
        else: raise AssertionError("accepted non-finite config")
    try: TrainingReplay([args.replay, args.replay], args.verifier)
    except ValueError: pass
    else: raise AssertionError("accepted duplicate replay sources")
    # Directory is the commit record, so loss of latest.json is recoverable.
    (root / "latest.json").write_text("interrupted pointer")
    equal(checkpoint.latest(root)[0], loaded)
    (root / ".pending-test").mkdir()
    (root / ".pending-test" / "state.pt").write_bytes(b"partial")
    equal(checkpoint.latest(root)[0], loaded)
    try: checkpoint.save(root, trainer.state(), {}, max_mib=32)
    except ValueError: pass
    else: raise AssertionError("ignored checkpoint byte cap")
    with saved.open("r+b") as file:
        file.seek(100)
        file.write(b"CORRUPT")
    try: checkpoint.latest(root)
    except ValueError: pass
    else: raise AssertionError("ignored checkpoint corruption")
    numeric = args.output / "numeric-order"
    numeric.mkdir()
    for update in (999999999, 1000000000):
        copy_state = checkpoint.cpu_tree(loaded)
        copy_state["updates"] = update
        checkpoint.save(numeric, copy_state, {})
    assert checkpoint.latest(numeric)[0]["updates"] == 1000000000

    # Checked and optimized loss paths must agree, including forced pass.
    p = torch.randn(4, 65, requires_grad=True)
    v = torch.tanh(torch.randn(4, 1, requires_grad=True))
    legal = torch.zeros(4, 65, dtype=torch.bool)
    legal[0, 64] = True
    legal[1:, :4] = True
    target = legal.float() / legal.sum(1, keepdim=True)
    z = torch.tensor([[1.0], [0.0], [-1.0], [0.0]])
    checked = policy_value_loss(p, v, target, z, legal)
    ce, mse, _ = policy_value_terms(p, v, target, z, legal)
    torch.testing.assert_close(checked, ce.mean() + mse.mean(), rtol=0, atol=0)
    checked.backward()
    assert torch.isfinite(p.grad).all() and (p.grad[~legal] == 0).all()

    # Deliberately overfit one fixed *training* batch. This is not validation or strength.
    overfit_config = {**config, "batch_size": 8, "milestones": [], "grad_clip": 10.0}
    overfit = Trainer(data, overfit_config, initial=args.init)
    indices = np.random.default_rng(9).choice(data.indices["train"], size=8, replace=False)
    fixed = decode(data.gather(indices))
    losses = []
    for _ in range(120):
        row = overfit.step(fixed)
        losses.append(row["policy_loss"] + row["value_loss"])
    assert losses[-1] < losses[0] * 0.4, (losses[0], losses[-1])
    overfit.model.eval()
    with torch.inference_mode():
        t = overfit.tensors(fixed)
        p, v = overfit.model(t["board"])
        eval_loss = float(policy_value_loss(p, v, t["policy"], t["outcome"], t["legal"]))
    assert eval_loss < losses[0] * 0.5, eval_loss
    result = {"result": "TRAINING TESTS PASSED", "resume": "CPU bit-identical", "validation": validation,
              "overfit_initial": losses[0], "overfit_final_train": losses[-1], "overfit_final_eval": eval_loss}
    if torch.backends.mps.is_available():
        mps_config = {**config, "device": "mps", "memory_format": "contiguous"}
        gpu = Trainer(data, mps_config, initial=args.init)
        for _ in range(3): gpu.step()
        saved_gpu = checkpoint.cpu_tree(gpu.state())
        for _ in range(3): gpu.step()
        expected_gpu = checkpoint.cpu_tree(gpu.state())
        gpu_resumed = Trainer(data, mps_config, state=saved_gpu)
        for _ in range(3): gpu_resumed.step()
        actual_gpu = checkpoint.cpu_tree(gpu_resumed.state())
        maximum = 0
        for key, tensor in expected_gpu["state_dict"].items():
            other = actual_gpu["state_dict"][key]
            torch.testing.assert_close(other, tensor, rtol=1e-5, atol=1e-6)
            maximum = max(maximum, float((tensor - other).abs().max()))
        equal(expected_gpu["rng"], actual_gpu["rng"])
        equal(expected_gpu["scheduler"], actual_gpu["scheduler"])
        result["mps_resume_max_abs_error"] = maximum
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
