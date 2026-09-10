"""Bounded synchronous UCI transport: persistent model, one outstanding search."""
import os
from pathlib import Path
import re
import selectors
import subprocess
import time

from arena_game import advance, fen, parse_token


class Engine:
    def __init__(self, binary):
        self.process = subprocess.Popen([str(Path(binary).resolve())], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = b""
        try:
            self.send("uci\nisready")
            lines = self.until("readyok", 30)
            if "uciok" not in lines: raise ValueError("missing UCI handshake")
            self.send("setoption name PerftHash value 1")
            self.barrier()
        except BaseException:
            self.close(abort=True)
            raise

    def send(self, command):
        data = (command + "\n").encode()
        while data:
            sent = self.process.stdin.write(data)
            if not sent: raise RuntimeError("engine stdin closed")
            data = data[sent:]

    def until(self, prefix, timeout):
        deadline, result = time.monotonic() + timeout, []
        while True:
            if time.monotonic() >= deadline: raise TimeoutError("engine deadline waiting for " + prefix)
            if b"\n" not in self.buffer:
                if not self.selector.select(max(0, deadline - time.monotonic())):
                    raise TimeoutError("engine deadline waiting for " + prefix)
                data = os.read(self.process.stdout.fileno(), 8192)
                if not data: raise RuntimeError("engine exited: " + repr(result[-8:]))
                self.buffer += data
                if len(self.buffer) > 65536: raise ValueError("oversized engine line")
                continue
            raw, self.buffer = self.buffer.split(b"\n", 1)
            line = raw.decode().rstrip("\r")
            result.append(line)
            if len(result) > 4096: raise ValueError("excessive engine output")
            if line.startswith("info error:") or line.startswith("info string search error"):
                raise RuntimeError(line)
            if line.startswith(prefix): return result

    def barrier(self):
        self.send("isready")
        lines = self.until("readyok", 30)
        if any(line.startswith("bestmove ") for line in lines): raise ValueError("stale bestmove")
        return lines

    def configure(self, model, tree_mib):
        if "\n" in str(model) or "\r" in str(model): raise ValueError("invalid model path")
        self.send(f"setoption name Rule value Othello\nsetoption name MctsHash value {tree_mib}\n"
                  f"setoption name EvalFile value {model}")
        self.barrier()

    def preflight(self):
        self.send("debug on\ntest\nposition startpos\ngo perft 8\nisready")
        lines = self.until("readyok", 60)
        if "ALL TESTS PASSED" not in lines or "Nodes searched: 390,216" not in lines:
            raise ValueError("engine correctness preflight failed")
        self.send("debug off")
        self.barrier()

    def new_game(self):
        self.send("ucinewgame")
        self.barrier()

    def search(self, board, config):
        limit = f"nodes {config['budget']}" if config["mode"] == "nodes" else f"movetime {config['budget']}"
        started = time.perf_counter_ns()
        self.send(f"position fen {fen(board)} X\ngo {limit}")
        timeout = config["timeout_ms"] / 1000
        if config["mode"] == "movetime": timeout = (config["budget"] + config["grace_ms"]) / 1000
        lines = self.until("bestmove ", timeout)
        elapsed_us = (time.perf_counter_ns() - started) / 1000
        info = [re.fullmatch(r"info nodes (\d+) nps (\d+) time (\d+)(?: pv (.*))?", s)
                for s in lines if s.startswith("info nodes ")]
        reason = [re.fullmatch(r"info string search (\w+) value ([-\d.]+) evaluations (\d+)", s)
                  for s in lines if s.startswith("info string search ")]
        evaluator = [s for s in lines if s.startswith("info string evaluator onnx-cpu b8c64-v1 checkpoint ")]
        if len(info) != 1 or not info[0] or len(reason) != 1 or not reason[0] or len(evaluator) != 1:
            raise ValueError("invalid search response: " + repr(lines))
        info, reason = info[0], reason[0]
        if not -1 <= float(reason[2]) <= 1: raise ValueError("invalid search value")
        if reason[1] != ("nodes" if config["mode"] == "nodes" else "time"):
            raise ValueError("arena search stopped early: " + reason[1])
        if config["mode"] == "nodes" and int(info[1]) != config["budget"]:
            raise ValueError("unequal simulation budget")
        pv = board
        for move in (info[4] or "").split(): pv = advance(pv, parse_token(move))
        fields = lines[-1].split()
        if len(fields) != 2: raise ValueError("invalid bestmove response")
        action = parse_token(fields[1])
        if info[4] and parse_token(info[4].split()[0]) != action: raise ValueError("PV/bestmove mismatch")
        advance(board, action)
        # Same-stream barrier detects duplicate results without a sleep/poll interval.
        self.barrier()
        return {"action": action, "nodes": int(info[1]), "engine_ms": int(info[3]),
                "elapsed_us": elapsed_us, "evaluations": int(reason[3]), "reason": reason[1],
                "evaluator": evaluator[0]}

    def close(self, abort=False):
        try:
            if self.process.poll() is None:
                if abort: self.process.kill()
                else: self.send("quit")
                try: self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
                    if not abort: raise RuntimeError("engine did not exit")
            if not abort and self.process.returncode: raise RuntimeError("engine exited unsuccessfully")
        finally:
            self.selector.close()
            self.process.stdin.close()
            self.process.stdout.close()
