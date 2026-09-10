"""One FP32 model owner for fixed-shape batches; CPU ORT reference and explicit Torch CPU/MPS."""
import time

import numpy as np
import onnxruntime as ort
import torch

from export import load_checkpoint
from model import encode
from replay import sha256


def ort_session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])


class Backend:
    def __init__(self, checkpoint, model, kind, batch):
        if kind not in ("ort-cpu", "torch-cpu", "torch-mps") or not 1 <= batch <= 128:
            raise ValueError("invalid inference backend/batch")
        if checkpoint.stat().st_size > 32 * 1048576 or model.stat().st_size > 64 * 1048576:
            raise ValueError("oversized inference assets")
        started = time.perf_counter_ns()
        self.kind, self.batch = kind, batch
        self.device = "mps" if kind == "torch-mps" else "cpu"
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS explicitly requested but unavailable")
        network, steps = load_checkpoint(checkpoint)
        reference = ort_session(model)
        props = reference.get_modelmeta().custom_metadata_map
        expected = {"islay.arch": "b8c64-v1", "islay.encoding": "relative-2x8x8-a1-v1",
                    "islay.policy": "a1-h8-pass65-v1", "islay.value": "relative-wdl-v1",
                    "islay.checkpoint_sha256": sha256(checkpoint), "islay.training_steps": str(steps)}
        if any(props.get(k) != v for k, v in expected.items()):
            raise ValueError("ONNX/checkpoint identity or architecture mismatch")
        if ([i.name for i in reference.get_inputs()] != ["board"]
                or [i.name for i in reference.get_outputs()] != ["policy_logits", "value"]):
            raise ValueError("invalid ONNX inference interface")
        self.network = network.to(self.device).eval() if kind != "ort-cpu" else None
        self.session = reference if kind == "ort-cpu" else None
        # Deterministic, model-independent boards including forced pass/terminal and all cells.
        boards = [(2, 1), (1, 2), (0, 0), ((1 << 64) - 1, 0)]
        rng = np.random.default_rng(20260912)
        for _ in range(128):
            cells = rng.integers(0, 3, size=64)
            boards.append((sum(1 << i for i in range(64) if cells[i] == 1),
                           sum(1 << i for i in range(64) if cells[i] == 2)))
        error = 0.0
        for start in range(0, len(boards), batch):
            chosen = boards[start:start + batch]
            chosen += [(0, 0)] * (batch - len(chosen))
            planes = encode(chosen).numpy()
            expected_p, expected_v = reference.run(None, {"board": planes})
            if kind == "ort-cpu":
                # Compare the checkpoint too, not an ORT session against itself.
                with torch.inference_mode():
                    p, v = network(torch.from_numpy(planes))
                    actual = np.concatenate((p.numpy(), v.numpy()), axis=1)
            else:
                actual = self.forward(planes)
            expected_output = np.concatenate((expected_p, expected_v), axis=1)
            np.testing.assert_allclose(actual, expected_output, rtol=1e-4, atol=2e-5)
            error = max(error, float(np.abs(actual - expected_output).max()))
        self.identity = {"backend": kind, "batch": batch, "torch": str(torch.__version__),
                         "ort": ort.__version__, "numpy": np.__version__, "fp32": True,
                         "checkpoint_sha256": expected["islay.checkpoint_sha256"], "training_steps": steps}
        self.startup = {"load_parity_us": (time.perf_counter_ns() - started) / 1000, "max_parity_error": error}

    def forward(self, planes):
        if planes.shape != (self.batch, 2, 8, 8) or planes.dtype != np.float32:
            raise ValueError("inference batch shape/dtype mismatch")
        if self.session is not None:
            p, v = self.session.run(None, {"board": planes})
        else:
            with torch.inference_mode():
                p, v = self.network(torch.from_numpy(planes).to(self.device))
                p, v = p.cpu().numpy(), v.cpu().numpy()
        if p.shape != (self.batch, 65) or v.shape != (self.batch, 1):
            raise ValueError("inference output shape mismatch")
        result = np.concatenate((p, v), axis=1)
        if not np.isfinite(result).all() or (np.abs(v) > 1).any():
            raise ValueError("non-finite/out-of-range inference result")
        return result
