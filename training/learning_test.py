"""Journal/fault tests and optional two-generation real-pipeline smoke (not a strength test)."""
import argparse
import copy
import contextlib
import fcntl
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from arena_stats import summarize
from learning import DEFAULT, can_resume, run, seed, selection, validate_plan
from learning_run import Journal, Paused, atomic_json, read_json
from replay import sha256
from selfplay import write_json


class LearningTests(unittest.TestCase):
    def test_configuration(self):
        plan = json.loads(DEFAULT.read_text())
        training = json.loads(DEFAULT.with_name("config_b.json").read_text())
        validate_plan(plan, training)
        for change in ({"games": 33}, {"generations": True}, {"family_alpha": float("nan")},
                       {"family_alpha": 0}, {"threshold": 1}, {"surprise": 1}, {"workers": 129},
                       {"games": 1000000}, {"backend": "automatic"}):
            with self.assertRaises(ValueError): validate_plan({**plan, **change}, training)
        with self.assertRaises(ValueError): validate_plan(plan, {**training, "max_replay_mib": 1})
        seeds = [seed(plan, g, role) for g in range(1, 101) for role in ("train", "selfplay", "arena", "anchor")]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertTrue(all(0 <= s < 1 << 63 for s in seeds))

    def test_selection(self):
        champion = {"sha256": "old", "checkpoint_sha256": "old-pt", "training_steps": 80}
        candidate = {"sha256": "new", "checkpoint_sha256": "new-pt", "training_steps": 82}
        for decision, expected in (("retain_champion", champion), ("candidate_eligible", candidate)):
            result = {"decision": decision, "complete": True, "selected_model": expected}
            self.assertEqual(selection(candidate, champion, result), expected)
            with self.assertRaises(ValueError): selection(candidate, champion, {**result, "complete": False})
            with self.assertRaises(ValueError): selection(candidate, champion, {**result, "selected_model": {}})

    def test_fixed_gate(self):
        wins = [{"games": [{"score": 1}, {"score": 1}]} for _ in range(200)]
        self.assertEqual(summarize(wins[:2], 2, 42, 2)["decision"], "retain_champion")
        self.assertEqual(summarize(wins[:199], 200, 42, 199)["decision"], "retain_champion")
        self.assertEqual(summarize(wins, 200, 42, 200, alpha=.05 / 3)["decision"], "candidate_eligible")
        self.assertEqual(summarize(wins, 200, 42, 100)["decision"], "retain_champion")

    def test_journal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (root / ".lock").open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                journal = Journal(root, lock, 5, 1048576, 1)
                command = lambda out, resume: [sys.executable, "-c",
                    "from pathlib import Path; import sys; p=Path(sys.argv[1]); p.mkdir(); (p/'data').write_text('ok')", out]
                inspect = lambda out: (out / "data").read_text()
                record = journal.stage("one", {"parent": 1}, command, inspect)
                self.assertEqual(journal.stage("one", {"parent": 1}, command, inspect), record)
                with self.assertRaises(Paused): journal.stage("two", {}, command, inspect)
                with self.assertRaises(ValueError): journal.stage("one", {"parent": 2}, command, inspect)
                output = journal.output("one", record)
                (output / "data").write_text("changed")
                with self.assertRaises(ValueError): journal.stage("one", {"parent": 1}, command, inspect)
                (output / "data").write_text("ok")
                (output / "extra").write_text("injected")
                with self.assertRaises(ValueError): journal.stage("one", {"parent": 1}, command, inspect)
                atomic_json(root / "test.json", {"valid": 1})
                raw = json.loads((root / "test.json").read_text())
                raw["value"]["valid"] = 2
                (root / "test.json").write_text(json.dumps(raw))
                with self.assertRaises(ValueError): read_json(root / "test.json")
                (output / "failure.json").write_text("{}")
                with self.assertRaises(ValueError): can_resume("arena", output)

    def test_partial_retry_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (root / ".lock").open("a+b") as lock:
                journal = Journal(root, lock, 5, 1048576)
                def failed(out, resume):
                    return [sys.executable, "-c", "from pathlib import Path; import sys; p=Path(sys.argv[1]); "
                            "p.mkdir(); (p/'partial').write_text('keep'); sys.exit(1)", out]
                with self.assertRaises(RuntimeError): journal.stage("export", {}, failed, lambda p: None)
                first = read_json(root / "export" / "active.json")["attempt"]
                def successful(out, resume):
                    return [sys.executable, "-c", "from pathlib import Path; import sys; p=Path(sys.argv[1]); "
                            "p.mkdir(exist_ok=True); (p/'complete').write_text('done')", out]
                done = journal.stage("export", {}, successful, lambda p: "done")
                self.assertNotEqual(first, done["attempt"])
                self.assertTrue((root / "export" / first / "output" / "partial").exists())
                with self.assertRaises(RuntimeError): journal.stage("resume", {}, failed, lambda p: None)
                first = read_json(root / "resume" / "active.json")["attempt"]
                done = journal.stage("resume", {}, successful, lambda p: "done", lambda p: (p / "partial").exists())
                self.assertEqual(first, done["attempt"])
                self.assertTrue((journal.output("resume", done) / "partial").exists())

    def test_timeout_and_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (root / ".lock").open("a+b") as lock:
                journal = Journal(root, lock, .1, 1048576)
                with self.assertRaises(TimeoutError):
                    journal.execute([sys.executable, "-c", "import time; time.sleep(10)"], root / "timeout.log")
                (root / "large").write_bytes(b"123")
                journal.max_bytes = 1
                with self.assertRaises(ValueError): journal.capacity()

    def test_orphan_holds_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = subprocess.Popen([sys.executable, __file__, "--hold-child", str(root)])
            try:
                deadline = time.monotonic() + 30
                while not (root / "ready").exists():
                    if process.poll() is not None or time.monotonic() > deadline: self.fail("child never became ready")
                    time.sleep(.05)
                process.kill()
                process.wait(timeout=5)
                with (root / ".lock").open("a+b") as lock:
                    with self.assertRaises(BlockingIOError): fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    deadline = time.monotonic() + 10
                    while True:
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            if time.monotonic() > deadline: self.fail("child did not release lock")
                            time.sleep(.05)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()


def hold_child(root):
    with (root / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = Journal(root, lock, 20, 1048576)
        journal.execute([sys.executable, "-c", "from pathlib import Path; import sys,time; "
                         "Path(sys.argv[1]).write_text('ready'); time.sleep(2)", str(root / "ready")], root / "log")


def smoke(args):
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(DEFAULT.read_text())
    plan.update(generations=2, games=32, simulations=8, workers=8, steps=2,
                arena_pairs=2, arena_budget=4, anchor_pairs=1, backend=args.backend, device=args.device)
    config = args.output / "config.json"
    write_json(config, plan)
    options = argparse.Namespace(**{name: getattr(args, name) for name in
        ("engine", "worker", "verifier", "probe", "model", "checkpoint")},
        config=config, train_config=DEFAULT.with_name("config_b.json"), output=args.output / "run",
        resume=False, dry_run=False, stop_after_stages=3)
    options.dry_run = True
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        run(options)
    assert json.loads(captured.getvalue())["gate_alpha_per_generation"] == plan["family_alpha"] / 2
    assert not options.output.exists()
    options.dry_run = False
    run(options)
    # Simulate supervisor loss after the trainer committed, before its stage journal commit.
    marker = options.output / "generation-0001" / "train" / "complete.json"
    trainer_files = read_json(marker)["files"]
    marker.replace(marker.with_name(".unpublished-complete.json"))
    committed_before = {str(p): sha256(p) for p in options.output.rglob("complete.json")}
    options.resume, options.stop_after_stages = True, None
    run(options)
    assert all(sha256(Path(p)) == identity for p, identity in committed_before.items())
    assert read_json(marker)["files"] == trainer_files
    assert "--resume" in read_json(marker)["command"]
    run(options)  # Finished resume verifies history and launches no training or games.
    decisions = [read_json(options.output / f"generation-{g:04d}" / "selection.json") for g in (1, 2)]
    for decision in decisions:
        assert decision["arena"]["decision"] == "retain_champion"
        assert decision["champion"] == decision["parent"]
        assert decision["training"]["training_steps"] == decision["parent"]["training_steps"] + 2
        assert decision["selfplay"]["games"] == 32
    bad = copy.copy(options)
    bad.config = args.output / "changed.json"
    write_json(bad.config, {**plan, "arena_budget": 5})
    try: run(bad)
    except ValueError: pass
    else: raise AssertionError("changed plan accepted")
    # Losing only the convenience pointer is harmless.
    (options.output / "champion.json").write_text("broken pointer")
    run(options)
    assert read_json(options.output / "champion.json")["generation"] == 2
    print("LEARNING TWO-GENERATION SMOKE PASSED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hold-child", type=Path)
    parser.add_argument("--smoke", action="store_true")
    for name in ("engine", "worker", "verifier", "probe", "model", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--backend", default="ort-cpu")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.hold_child: hold_child(args.hold_child)
    elif args.smoke:
        if any(getattr(args, name) is None for name in ("engine", "worker", "verifier", "probe", "model", "checkpoint", "output")):
            parser.error("smoke requires binaries, model/checkpoint and fresh output")
        smoke(args)
    else: unittest.main(argv=[sys.argv[0]])


if __name__ == "__main__": main()
