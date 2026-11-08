"""Durable stage journal and single-owner subprocess supervision for learning runs."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from replay import sha256
from selfplay import sync_directory, write_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_plain_json(path, value):
    path = Path(path)
    with tempfile.TemporaryDirectory(prefix=".journal-", dir=path.parent) as temporary:
        source = Path(temporary) / "record.json"
        write_json(source, value)
        os.replace(source, path)
        sync_directory(path.parent)


def atomic_json(path, value):
    atomic_plain_json(path, {"sha256": digest(value), "value": value})


def read_json(path):
    envelope = json.loads(Path(path).read_text())
    if set(envelope) != {"sha256", "value"} or envelope["sha256"] != digest(envelope["value"]):
        raise ValueError(f"journal checksum mismatch: {path}")
    return envelope["value"]


def inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts) or path.name == "latest.json":
            continue
        if path.is_symlink(): raise ValueError("symlink in committed stage")
        if path.is_file(): result[str(relative)] = sha256(path)
    return result


class Paused(Exception):
    pass


class Journal:
    def __init__(self, root, lock, timeout, max_bytes, stop_after=None):
        self.root, self.lock = root, lock
        self.timeout, self.max_bytes = timeout, max_bytes
        self.remaining = stop_after

    def capacity(self):
        used = 0
        for path in self.root.rglob("*"):
            # A producer can atomically rename a pending directory during this advisory scan.
            try:
                if path.is_file(): used += path.stat().st_size
            except FileNotFoundError:
                continue
        if used > self.max_bytes: raise ValueError("learning storage cap reached; nothing evicted")

    def execute(self, command, log):
        # A surviving stage retains the supervisor lock after SIGKILL. Resume cannot race it.
        with log.open("xb") as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True, pass_fds=(self.lock.fileno(),))
            started = time.monotonic()
            checked = started - 5
            try:
                while child.poll() is None:
                    if time.monotonic() - started > self.timeout: raise TimeoutError("learning stage timeout")
                    if time.monotonic() - checked >= 5:
                        self.capacity()
                        checked = time.monotonic()
                    time.sleep(0.2)
                if child.returncode:
                    raise RuntimeError(f"stage exited {child.returncode}; inspect {log}")
            except BaseException:
                try: os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
            output.flush()
            os.fsync(output.fileno())

    def stage(self, name, request, command, inspect, resumable=None):
        directory = self.root / name
        directory.mkdir(parents=True, exist_ok=True)
        committed = directory / "complete.json"
        if committed.exists():
            record = read_json(committed)
            if record["request"] != request:
                raise ValueError(f"changed stage dependency: {name}")
            output = directory / record["attempt"] / "output"
            if inventory(output) != record["files"]: raise ValueError(f"modified committed stage: {name}")
            return record
        if self.remaining == 0: raise Paused()
        self.capacity()
        active = directory / "active.json"
        attempt = read_json(active) if active.exists() else None
        resume = False
        if attempt is not None:
            if attempt["request"] != request: raise ValueError("changed active stage dependency")
            output = directory / attempt["attempt"] / "output"
            resume = resumable(output) if resumable else False
        if not resume:
            attempt = {"attempt": "attempt-" + uuid.uuid4().hex, "request": request}
            (directory / attempt["attempt"]).mkdir()
            atomic_json(active, attempt)
        work = directory / attempt["attempt"]
        output = work / "output"
        argv = [str(x) for x in command(output, resume)]
        log = work / ("log-" + uuid.uuid4().hex + ".txt")
        print(json.dumps({"stage": name, "resume": resume, "log": str(log)}), flush=True)
        started = time.perf_counter_ns()
        self.execute(argv, log)
        result = inspect(output)
        self.capacity()
        if not output.is_dir() or output.is_symlink(): raise ValueError("missing/invalid stage output")
        # CLI export is not itself atomic; the journal commits only fully durable artifacts.
        for path in output.rglob("*"):
            if path.is_file():
                with path.open("rb") as file: os.fsync(file.fileno())
        for path in sorted((p for p in output.rglob("*") if p.is_dir()), reverse=True):
            sync_directory(path)
        sync_directory(output)
        record = {**attempt, "command": argv, "result": result, "files": inventory(output),
                  "last_attempt_seconds": (time.perf_counter_ns() - started) / 1e9}
        atomic_json(committed, record)
        if self.remaining is not None: self.remaining -= 1
        return record

    def output(self, name, record):
        return self.root / name / record["attempt"] / "output"


def python_command(script, *args):
    return [sys.executable, str(Path(__file__).with_name(script)), *args]
