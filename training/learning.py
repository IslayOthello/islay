"""Frozen, resumable multi-generation self-play/training/arena; no UCI deployment."""
import argparse
import fcntl
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import signal
import sys

import torch

from arena import load_pairs, model_info, report
import checkpoint
from learning_run import Journal, Paused, atomic_json, atomic_plain_json, digest, python_command, read_json
from replay import RECORD_BYTES, manifest, sha256, shards
from train import validate_config

DIRECTORY = Path(__file__).resolve().parent
DEFAULT = DIRECTORY / "learning_config.json"


def validate_plan(plan, training):
    defaults = json.loads(DEFAULT.read_text())
    if set(plan) != set(defaults): raise ValueError("unknown/missing learning configuration fields")
    limits = {"schema": (1, 1), "generations": (1, 100), "seed": (0, (1 << 62) - 1),
              "games": (1, 1000000), "simulations": (1, 1000000), "workers": (1, 128),
              "steps": (1, 1000000), "replay_generations": (1, 100), "arena_pairs": (1, 10000),
              "arena_budget": (1, 1000000), "anchor_pairs": (1, 10000), "replay_mib": (32, 65536),
              "checkpoint_mib": (32, 65536), "arena_mib": (16, 65536), "total_mib": (64, 1048576),
              "stage_timeout_seconds": (10, 604800)}
    for key, (lo, hi) in limits.items():
        if type(plan[key]) is not int or not lo <= plan[key] <= hi: raise ValueError("invalid " + key)
    for key, lo, hi in (("family_alpha", 0, .05), ("threshold", .5, 1)):
        if (type(plan[key]) not in (int, float) or not math.isfinite(plan[key])
                or not lo <= plan[key] <= hi): raise ValueError("invalid " + key)
    if plan["family_alpha"] == 0 or plan["threshold"] == 1: raise ValueError("invalid gate")
    if plan["backend"] not in ("ort-cpu", "torch-cpu", "torch-mps") or plan["device"] not in ("cpu", "mps", "cuda"):
        raise ValueError("invalid learning backend/device")
    if plan["games"] % plan["workers"]: raise ValueError("games must be whole worker cohorts")
    validate_config({**training, "device": plan["device"]})
    # Bound the full inputs, not just the trainer's selected suffix. Actual plies are <=128.
    payload = plan["games"] * 128 * RECORD_BYTES
    if payload + 40 * 1048576 > plan["replay_mib"] * 1048576:
        raise ValueError("replay cap cannot hold worst-case games and model snapshots")
    window = min(plan["generations"], plan["replay_generations"])
    if payload * window > training["max_replay_mib"] * 1048576:
        raise ValueError("worst-case replay window exceeds trainer max_replay_mib")


def seed(plan, generation, role):
    return int(digest([plan["seed"], generation, role])[:15], 16)


def frozen_manifest(args):
    plan = json.loads(args.config.read_text())
    training = json.loads(args.train_config.read_text())
    validate_plan(plan, training)
    if (plan["device"] == "mps" or plan["backend"] == "torch-mps") and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    if plan["device"] == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA requested but unavailable")
    paths = {key: getattr(args, key).resolve() for key in ("engine", "worker", "verifier", "probe", "model", "checkpoint")}
    if any(not p.is_file() for p in paths.values()): raise ValueError("expected regular input files")
    if not 0 < paths["checkpoint"].stat().st_size <= 32 * 1048576: raise ValueError("invalid initial checkpoint size")
    identity = model_info(paths["model"])
    if identity["checkpoint_sha256"] != sha256(paths["checkpoint"]): raise ValueError("initial model/checkpoint mismatch")
    initial_bytes = paths["model"].stat().st_size + paths["checkpoint"].stat().st_size
    if initial_bytes + plan["games"] * 128 * RECORD_BYTES + 1048576 > plan["replay_mib"] * 1048576:
        raise ValueError("initial model snapshots and worst-case replay exceed cap")
    code = [p for p in DIRECTORY.glob("*.py") if not p.stem.endswith(("_test", "_bench"))]
    code += [DIRECTORY.parent / "tools" / n for n in ("mcts_study.py", "uci_search_test.py", "perft_study.py")]
    return {"schema": "islay-learning-v1", "plan": plan, "training": training,
            "inputs": {k: {"path": str(p), "sha256": sha256(p)} for k, p in paths.items()},
            "code": {str(p.relative_to(DIRECTORY.parent)): sha256(p) for p in code},
            "runtime": {"python": sys.version, "platform": platform.platform(),
                        "packages": {n: importlib.metadata.version(n) for n in
                                     ("torch", "numpy", "onnx", "onnxruntime", "onnxscript")}},
            "optimizer_policy": "reset-for-new-generation; full-state-resume-with-frozen-replay",
            "gate_alpha_per_generation": plan["family_alpha"] / plan["generations"],
            "seeds": [{role: seed(plan, g, role) for role in ("selfplay", "train", "arena", "anchor")}
                      for g in range(1, plan["generations"] + 1)]}


def model_ref(model, state):
    info = model_info(model)
    if info["checkpoint_sha256"] != sha256(state): raise ValueError("selected model/checkpoint mismatch")
    return {"model": str(model), "checkpoint": str(state), **info}


def can_resume(kind, output):
    if kind == "replay": return (output / "manifest.json").exists()
    if kind == "train": return bool(list(output.glob("step-*")))
    if kind == "arena":
        if (output / "failure.json").exists(): raise ValueError("arena failure recorded; no automatic retry")
        return (output / "assets").exists()
    return False


def inspect_replay(output, games):
    meta = manifest(output)
    entries = shards(output, meta)
    count = sum(info["games"] for _, info in entries)
    if count != games: raise ValueError("incomplete/oversized generation replay")
    return {"games": count, "samples": sum(info["samples"] for _, info in entries), "manifest": digest(meta)}


def inspect_train(output, steps, parent):
    state, path = checkpoint.latest(output)
    if state["updates"] != steps or state["initial_sha256"] != parent["checkpoint_sha256"]:
        raise ValueError("training target or parent mismatch")
    if state["training_steps"] != parent["training_steps"] + steps: raise ValueError("invalid cumulative updates")
    return {"checkpoint": str(path), "sha256": sha256(path), "updates": steps,
            "training_steps": state["training_steps"], "validation": json.loads(path.with_name("metrics.json").read_text())["validation"]}


def inspect_arena(output):
    meta = json.loads((output / "assets" / "manifest.json").read_text())
    if (output / "failure.json").exists(): raise ValueError("failed arena cannot select a champion")
    path = output / "final" / "report.json"
    if json.loads(path.with_name("index.json").read_text()) != {"sha256": sha256(path)}:
        raise ValueError("arena report checksum mismatch")
    result = report(load_pairs(output, meta), meta)
    if not result["complete"] or json.loads(path.read_text()) != result: raise ValueError("invalid/incomplete arena")
    return result


def selection(candidate, champion, arena):
    if not arena["complete"]: raise ValueError("cannot select from partial arena")
    chosen = candidate if arena["decision"] == "candidate_eligible" else champion
    expected = {key: chosen[key] for key in ("sha256", "checkpoint_sha256", "training_steps")}
    if arena["selected_model"] != expected: raise ValueError("arena selected model mismatch")
    return chosen


def pipeline(journal, frozen):
    plan, inputs = frozen["plan"], frozen["inputs"]
    paths = {k: Path(v["path"]) for k, v in inputs.items()}
    anchor = model_ref(paths["model"], paths["checkpoint"])
    champion, replay_roots = anchor, []

    def check_inputs():
        # Do not continue a run if code, binaries or initial artifacts changed in flight.
        for key, row in inputs.items():
            if sha256(row["path"]) != row["sha256"]: raise ValueError("modified input: " + key)
        for name_in_repo, expected in frozen["code"].items():
            if sha256(DIRECTORY.parent / name_in_repo) != expected: raise ValueError("learning code changed during run")

    def checked_stage(name, request, command, inspect, kind=None):
        check_inputs()
        def checked_inspect(output):
            result = inspect(output)
            check_inputs()
            return result
        result = journal.stage(name, request, command, checked_inspect,
                               (lambda output: can_resume(kind, output)) if kind else None)
        check_inputs()
        return result

    def parity(name, ref):
        return checked_stage(name, ref, lambda out, _: python_command("learning_check.py",
            "--checkpoint", ref["checkpoint"], "--model", ref["model"], "--probe", paths["probe"],
            "--engine", paths["engine"],
            "--backend", plan["backend"], "--workers", plan["workers"], "--output", out),
            lambda out: json.loads((out / "result.json").read_text()))

    parity("initial-parity", anchor)
    for generation in range(1, plan["generations"] + 1):
        prefix = f"generation-{generation:04d}"
        seeds = frozen["seeds"][generation - 1]
        request = {"generation": generation, "parent": champion, "plan": digest(frozen)}
        name = prefix + "/selfplay"
        replay = checked_stage(name, request, lambda out, _: python_command("batched_selfplay.py",
            "--worker", paths["worker"], "--verifier", paths["verifier"], "--model", champion["model"],
            "--checkpoint", champion["checkpoint"], "--output", out, "--games", plan["games"],
            "--workers", plan["workers"], "--backend", plan["backend"], "--seed", seeds["selfplay"],
            "--simulations", plan["simulations"], "--max-mib", plan["replay_mib"]),
            lambda out: inspect_replay(out, plan["games"]), "replay")
        replay_roots.append(journal.output(name, replay))
        selected_replays = replay_roots[-plan["replay_generations"]:]
        config_path = journal.root / prefix / "train-config.json"
        config = {**frozen["training"], "seed": seeds["train"], "device": plan["device"]}
        # Trainer expects plain JSON, not a journal envelope. Its digest is in the stage request.
        if config_path.exists():
            if json.loads(config_path.read_text()) != config: raise ValueError("changed generation training config")
        else:
            atomic_plain_json(config_path, config)
        train_request = {**request, "replays": [str(p) for p in selected_replays], "training": config}
        trained = checked_stage(prefix + "/train", train_request, lambda out, resume: python_command("train.py",
            *[x for p in selected_replays for x in ("--replay", p)], "--verifier", paths["verifier"],
            *(["--resume"] if resume else ["--init", champion["checkpoint"]]), "--config", config_path,
            "--steps", plan["steps"], "--max-mib", plan["checkpoint_mib"], "--output", out),
            lambda out: inspect_train(out, plan["steps"], champion), "train")
        state = Path(trained["result"]["checkpoint"])
        exported = checked_stage(prefix + "/export", trained["result"], lambda out, _: python_command("export.py",
            "--checkpoint", state, "--output", out), lambda out: model_ref(out / "model.onnx", state))
        candidate = exported["result"]
        parity(prefix + "/parity", candidate)

        def arena_stage(role, opponent, pairs):
            return checked_stage(prefix + "/" + role, {"candidate": candidate, "opponent": opponent,
                "pairs": pairs, "seed": seeds[role], "plan": digest(frozen)},
                lambda out, resume: python_command("arena.py", "--engine", paths["engine"],
                    "--candidate", candidate["model"], "--champion", opponent["model"], "--output", out,
                    "--pairs", pairs, "--seed", seeds[role], "--mode", "nodes", "--budget", plan["arena_budget"],
                    "--alpha", frozen["gate_alpha_per_generation"], "--threshold", plan["threshold"],
                    "--max-mib", plan["arena_mib"], *(["--resume"] if resume else [])), inspect_arena, "arena")

        arena = arena_stage("arena", champion, plan["arena_pairs"])
        baseline = arena_stage("anchor", anchor, plan["anchor_pairs"])
        chosen = selection(candidate, champion, arena["result"])
        decision = {"generation": generation, "parent": champion, "candidate": candidate, "champion": chosen,
                    "arena": arena["result"], "anchor_diagnostic": baseline["result"],
                    "selfplay": replay["result"], "training": trained["result"],
                    "deployment": "learning run only; UCI defaults unchanged"}
        committed = journal.root / prefix / "selection.json"
        if committed.exists():
            if read_json(committed) != decision: raise ValueError("modified generation selection")
        else: atomic_json(committed, decision)
        champion = chosen
        # Convenience pointer is never trusted for resume; rebuild from validated stage history.
        atomic_json(journal.root / "champion.json", {"generation": generation, "model": champion})
        print(json.dumps({"generation": generation, "decision": arena["result"]["decision"],
                          "champion_sha256": champion["sha256"], "anchor_score": baseline["result"]["score"]}), flush=True)
    return champion


def run(args):
    frozen = frozen_manifest(args)
    if args.dry_run:
        print(json.dumps(frozen, indent=2))
        return
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    with (root / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / "run.json"
        if args.resume:
            if read_json(path) != frozen: raise ValueError("resume requires identical frozen plan, inputs, code and runtime")
        else: atomic_json(path, frozen)
        journal = Journal(root, lock, frozen["plan"]["stage_timeout_seconds"],
                          frozen["plan"]["total_mib"] * 1048576, args.stop_after_stages)
        try:
            champion = pipeline(journal, frozen)
            atomic_json(root / "complete.json", {"plan": digest(frozen), "champion": champion})
        except Paused:
            print(json.dumps({"paused": True, "resume": "repeat the same command with --resume"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("engine", "worker", "verifier", "probe", "model", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--train-config", type=Path, default=DIRECTORY / "config_b.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate/print frozen plan; do not create output")
    parser.add_argument("--stop-after-stages", type=int, help="pause after N newly completed stages; not a selection gate")
    args = parser.parse_args()
    if args.stop_after_stages is not None and args.stop_after_stages < 1: parser.error("invalid stage limit")
    def interrupt(signum, frame): raise KeyboardInterrupt("learning run interrupted")
    signal.signal(signal.SIGTERM, interrupt)
    run(args)


if __name__ == "__main__": main()
