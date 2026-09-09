#!/usr/bin/env python3
"""Black-box P2 protocol/lifecycle tests, with one engine at a time."""
import argparse
import json
from pathlib import Path
import queue
import re
import statistics
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
from perft_study import START, scalar_moves, play
from mcts_study import corpus


def fen(p, o):
    return "".join("X" if p >> sq & 1 else "O" if o >> sq & 1 else "-" for sq in range(64))


class Engine:
    def __init__(self, binary):
        self.process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, bufsize=1)
        self.lines = queue.Queue()
        self.errors = []
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.stderr_reader = threading.Thread(target=self.read_errors, daemon=True)
        self.reader.start()
        self.stderr_reader.start()
        self.send("uci\nisready")
        self.until("readyok")

    def read(self):
        for line in self.process.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)

    def read_errors(self):
        self.errors.extend(self.process.stderr.readlines())

    def send(self, command):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def until(self, prefix, timeout=20):
        result = []
        deadline = time.monotonic() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.001, deadline - time.monotonic()))
            except queue.Empty:
                raise AssertionError(f"timeout waiting for {prefix}: {result}") from None
            assert line is not None, f"unexpected EOF: {result} {self.errors}"
            result.append(line)
            if line.startswith(prefix):
                return result

    def barrier(self):
        self.send("isready")
        return self.until("readyok")

    def close(self, eof=False):
        if self.process.poll() is None:
            if eof:
                self.process.stdin.close()
            else:
                self.send("quit")
            self.process.wait(timeout=20)
        self.reader.join(timeout=2)
        self.stderr_reader.join(timeout=2)
        assert self.process.returncode == 0, self.errors
        assert not self.errors, self.errors

    def abort(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()


def search(engine, board, command, expected_nodes=None, expected_move=None):
    engine.send(f"position fen {fen(*board)} X\n{command}")
    lines = engine.until("bestmove ")
    assert sum(line.startswith("bestmove ") for line in lines) == 1, lines
    assert any("no trained network" in line for line in lines), lines
    infos = [line for line in lines if line.startswith("info nodes ")]
    assert len(infos) == 1, lines
    info = re.fullmatch(r"info nodes (\d+) nps (\d+) time (\d+)(?: pv (.*))?", infos[0])
    assert info, infos
    if expected_nodes is not None:
        assert int(info[1]) == expected_nodes, lines
    p, o = board
    for action in (info[4] or "").split():
        moves = dict(scalar_moves(p, o))
        if action == "pass":
            assert not moves and scalar_moves(o, p), lines
            p, o = o, p
        else:
            sq = ord(action[0]) - ord("a") + (int(action[1]) - 1) * 8
            assert sq in moves, lines
            p, o = play(p, o, (sq, moves[sq]))
    move = lines[-1].split()[1]
    moves = dict(scalar_moves(*board))
    if moves:
        sq = ord(move[0]) - ord("a") + (int(move[1]) - 1) * 8
        assert sq in moves, lines
    else:
        assert move == ("pass" if scalar_moves(board[1], board[0]) else "0000"), lines
    if expected_move is not None:
        assert move == expected_move, lines
    assert not any(line.startswith("bestmove ") for line in engine.barrier())
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    binary = parser.parse_args().binary.resolve()
    engine = Engine(binary)
    latencies = []
    try:
        engine.send("setoption name MctsHash value 1")
        assert any("MctsHash = 1" in line for line in engine.barrier())
        for board, move, nodes in [(START, None, 128), ((2, 1), "pass", 128),
                                    ((1, 2), "c1", 128), ((1, 0), "0000", 0),
                                    ((0xffffffff, 0xffffffff00000000), "0000", 0)]:
            search(engine, board, "go nodes 128", nodes, move)
        search(engine, START, "go nodes 0", 0)
        search(engine, START, "go movetime 0", 0)
        search(engine, START, "go nodes 64 movetime 10000", 64)
        search(engine, START, "go movetime 10000 nodes 64", 64)
        for position in corpus():
            search(engine, (position["p"], position["o"]), "go nodes 512", 512)
        lines = search(engine, START, "go nodes 1000000000")
        assert any("search memory" in line for line in lines), lines
        engine.send("setoption name MctsHash value 64")
        engine.barrier()
        lines = search(engine, (2, 1), "go movetime 20")
        assert any("search time" in line for line in lines), lines
        for command in ["go", "go depth 2", "go nodes -1", "go nodes 1x", "go nodes",
                        "go nodes 18446744073709551616", "go movetime 86400001",
                        "go infinite nodes 1", "go nodes 1 infinite", "go infinite infinite",
                        "go nodes 1 nodes 2", "go movetime 1 movetime 2", "go wtime 100"]:
            engine.send(command)
            lines = engine.barrier()
            assert any(line.startswith("info error:") for line in lines), (command, lines)
            assert not any(line.startswith("bestmove ") for line in lines), lines
        engine.send("setoption name Rule value Reversi\ngo nodes 1")
        assert any("supports Rule=Othello only" in line for line in engine.barrier())
        engine.send("position startpos\ngo perft 8")
        assert "Nodes searched: 390,216" in engine.barrier()
        engine.send("setoption name Rule value Othello")
        engine.barrier()
        for _ in range(10):
            engine.send("position startpos\ngo nodes 1000000000\nstop\nisready")
            lines = engine.until("readyok")
            assert sum(line.startswith("bestmove ") for line in lines) == 1, lines
        engine.send(f"position fen {fen(1, 0)} X\ngo infinite")
        engine.barrier()
        time.sleep(0.05)
        assert not any(line.startswith("bestmove ") for line in engine.barrier())
        engine.send("stop")
        assert sum(line.startswith("bestmove ") for line in engine.barrier()) == 1
        for iteration in range(20):
            board = (1, 0) if iteration % 2 else START
            engine.send(f"position fen {fen(*board)} X\ngo infinite")
            lines = engine.barrier()
            assert not any(line.startswith("bestmove ") for line in lines), lines
            started = time.perf_counter_ns()
            engine.send("stop\nstop\nisready")
            lines = engine.until("readyok")
            latencies.append((time.perf_counter_ns() - started) / 1e6)
            assert sum(line.startswith("bestmove ") for line in lines) == 1, lines
        for command in ["position startpos", "position invalid", "ucinewgame",
                        "setoption name MctsHash value 1", "setoption name bad value 1",
                        "go perft 2", "go nodes -1"]:
            engine.send("position startpos\ngo infinite")
            engine.barrier()
            engine.send(command + "\nisready")
            lines = engine.until("readyok")
            assert not any(line.startswith("bestmove ") for line in lines), (command, lines)
        engine.send("position startpos\ngo infinite\nuci")
        lines = engine.barrier()
        assert "uciok" in lines and sum(line.startswith("option name ") for line in lines) == 3, lines
        engine.send("stop")
        assert sum(line.startswith("bestmove ") for line in engine.barrier()) == 1
        engine.send("position startpos\ngo infinite")
        engine.barrier()
        search(engine, (1, 0), "go nodes 1", 0, "0000")
        engine.send("position startpos\ngo infinite")
        engine.barrier()
        engine.send("go nodes 32")
        lines = engine.until("bestmove ")
        assert any(line.startswith("info nodes 32 ") for line in lines), lines
        engine.send("stop")
        assert not any(line.startswith("bestmove ") for line in engine.barrier())
        engine.send("position startpos\ngo infinite")
        engine.barrier()
        engine.close()
        assert not any(line and line.startswith("bestmove ") for line in list(engine.lines.queue))
    finally:
        engine.abort()
    eof = Engine(binary)
    try:
        eof.send("go infinite")
        eof.barrier()
        eof.close(eof=True)
        assert not any(line and line.startswith("bestmove ") for line in list(eof.lines.queue))
    finally:
        eof.abort()
    print(json.dumps({"result": "ALL PROTOCOL TESTS PASSED", "stop_samples": len(latencies),
                      "stop_median_ms": statistics.median(latencies),
                      "stop_p95_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))]}, indent=2))


if __name__ == "__main__":
    main()
