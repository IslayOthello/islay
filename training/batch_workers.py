"""Fixed cohorts of single-owner C++ trees; shm tensors/replay, socket notifications, one model."""
import json
import selectors
import socket
import struct
import subprocess
import tempfile
import time

import numpy as np

from replay import RECORD_BYTES, game_seed
from selfplay import FRAME, config_args
from shm_region import Region

INPUT_BYTES, OUTPUT_BYTES = 512, 264
FRAME_BYTES = FRAME.size + 128 * RECORD_BYTES
SLOT_BYTES = INPUT_BYTES + OUTPUT_BYTES + FRAME_BYTES + 16


class Workers:
    def __init__(self, binary, backend, config):
        self.backend, self.config = backend, config
        self.count = config["workers"]
        self.region = Region(self.count * SLOT_BYTES)
        self.inputs = np.ndarray((self.count, 2, 8, 8), np.float32, buffer=self.region.map)
        self.outputs = np.ndarray((self.count, 66), np.float32, buffer=self.region.map, offset=self.count * INPUT_BYTES)
        self.selector = selectors.DefaultSelector()
        self.processes, self.sockets, self.logs = [], [], []
        self.metrics = {"rounds": 0, "neural_positions": 0, "padded_positions": 0,
                        "inference_us": 0.0, "cache_hits": 0, "games": 0}
        try:
            for slot in range(self.count):
                parent, child = socket.socketpair()
                self.sockets.append(parent)
                log = tempfile.TemporaryFile()
                self.logs.append(log)
                try:
                    command = [str(binary), "--shm-fd", str(self.region.fd), "--control-fd", str(child.fileno()),
                               "--workers", str(self.count), "--slot", str(slot), "--cache-mib", str(config["cache_mib"]),
                               "--timeout-ms", str(config["timeout_ms"]), *config_args(config)]
                    self.processes.append(subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                                           stderr=log, pass_fds=(self.region.fd, child.fileno())))
                finally: child.close()
                self.selector.register(parent, selectors.EVENT_READ, slot)
            for channel in self.sockets:
                channel.settimeout(config["timeout_ms"] / 1000)
                if channel.recv(1) != b"V": raise RuntimeError("invalid worker handshake")
                channel.settimeout(None)
        except BaseException as failure:
            failure.add_note(self.diagnostics())
            self.close(abort=True)
            raise

    def diagnostics(self):
        messages = []
        for slot, log in enumerate(self.logs):
            log.seek(0)
            message = log.read(65536).decode(errors="replace")
            if message: messages.append(f"worker {slot}: {message}")
        return "\n".join(messages)

    def cohort(self, start):
        if start % self.count: raise ValueError("cohorts must align to fixed worker count")
        self.inputs.fill(0)
        self.outputs.fill(np.nan)
        for slot, channel in enumerate(self.sockets): channel.sendall(b"G" + struct.pack("<Q", start + slot))
        active, pending, frames = set(range(self.count)), set(), {}
        requests = [0] * self.count
        while active:
            deadline = time.monotonic() + self.config["timeout_ms"] / 1000
            while pending != active:
                events = self.selector.select(max(0, deadline - time.monotonic()))
                if not events: raise TimeoutError("cohort worker stalled or exited")
                for key, _ in events:
                    slot = key.data
                    message = key.fileobj.recv(1)
                    if slot not in active or slot in pending: raise ValueError("unexpected worker notification")
                    if message == b"R":
                        pending.add(slot)
                        requests[slot] += 1
                    elif message == b"D":
                        offset = self.count * (INPUT_BYTES + OUTPUT_BYTES) + slot * FRAME_BYTES
                        magic, game, seed, length, flags = FRAME.unpack_from(self.region.map, offset)
                        if (magic != b"ISLAGM01" or game != start + slot or seed != game_seed(self.config["seed"], game)
                                or not 1 <= length <= 128 or flags):
                            raise ValueError("invalid shared replay frame")
                        frames[slot] = bytes(self.region.map[offset + FRAME.size:offset + FRAME.size + length * RECORD_BYTES])
                        metrics_offset = self.count * (INPUT_BYTES + OUTPUT_BYTES + FRAME_BYTES) + slot * 16
                        hits, misses = struct.unpack_from("<QQ", self.region.map, metrics_offset)
                        self.metrics["cache_hits"] += hits
                        if requests[slot] != misses: raise ValueError("worker cache/inference count mismatch")
                        active.remove(slot)
                        self.inputs[slot].fill(0)
                    else: raise RuntimeError(f"worker {slot} failed/closed control channel: {message!r}")
            if not active: break
            if not np.isfinite(self.inputs).all() or not np.isin(self.inputs, [0, 1]).all():
                raise ValueError("invalid shared neural input")
            if (self.inputs[:, 0] + self.inputs[:, 1] > 1).any(): raise ValueError("overlapping shared neural input")
            started = time.perf_counter_ns()
            self.outputs[:] = self.backend.forward(self.inputs)
            self.metrics["inference_us"] += (time.perf_counter_ns() - started) / 1000
            self.metrics["rounds"] += 1
            self.metrics["neural_positions"] += len(active)
            self.metrics["padded_positions"] += self.count
            for slot in sorted(active): self.sockets[slot].sendall(b"A")
            pending.clear()
        self.metrics["games"] += self.count
        return [frames[slot] for slot in range(self.count)]

    def close(self, abort=False):
        failure = None
        try:
            for channel in self.sockets:
                try:
                    if not abort: channel.sendall(b"Q")
                except OSError:
                    if not abort: failure = RuntimeError("worker closed before clean shutdown")
                channel.close()
            for process, log in zip(self.processes, self.logs):
                if abort and process.poll() is None: process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                    if not abort: failure = RuntimeError("worker failed to stop")
                log.seek(0)
                diagnostic = log.read(65536).decode(errors="replace")
                if not abort and (process.returncode or diagnostic): failure = RuntimeError(diagnostic or "worker failed")
            if failure: raise failure
        finally:
            for log in self.logs: log.close()
            self.selector.close()
            del self.inputs, self.outputs
            self.region.close()

    def __enter__(self): return self
    def __exit__(self, kind, failure, _):
        if failure is not None: failure.add_note(self.diagnostics())
        self.close(abort=kind is not None)
