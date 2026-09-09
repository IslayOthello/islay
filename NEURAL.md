# Configuration B: shared policy/value network

P3 implements the model, checkpoint export, optional C++ inference and UCI integration.
P4 now supplies [self-play generation and replay loading](SELFPLAY.md). Neither step supplies
trained playing strength, a replay trainer or an arena/champion loop. The generated random
checkpoint is a test/bootstrap artifact.

## Fixed contract

One shared network, not two independent networks; **605,960 trainable parameters**.

| Component | Shape / operation |
|---|---|
| Input `board` | FP32 `[N,2,8,8]`, N=1..128; mover's discs then opponent's discs |
| Stem | 3x3 convolution, 2 → 64 channels, BatchNorm, ReLU |
| Residual trunk | 8 blocks: Conv3x3/BN/ReLU → Conv3x3/BN → add skip → ReLU |
| Policy head | Conv1x1 64 → 2 / BN / ReLU → flatten 128 → linear 65 logits |
| Value head | Conv1x1 64 → 1 / BN / ReLU → flatten 64 → linear 64 / ReLU → linear 1 / tanh |
| Outputs | `policy_logits` FP32 `[N,65]`, `value` FP32 `[N,1]` |

Convolutions have no bias, stride 1 and same padding; linear layers have bias.
BatchNorm uses epsilon 1e-5, momentum 0.1, learned affine parameters and running statistics.
Export/inference use eval mode. There is no history plane or absolute-color plane.

Row 0 is rank 1, column 0 is file a: `a1=0` through `h8=63`, `PASS=64`.
Value is expected win/draw/loss outcome (+1/0/-1) from the mover's perspective, not disc
margin, centipawns or pure win probability. PUCT masks illegal logits before softmax;
forced pass has probability one and terminal nodes use the exact outcome without inference.
Python includes D4 policy transforms/inverses and a legal-masked policy CE + value MSE loss.
Optimizer regularization is separate. Synthetic gradient checks are not training data.

Metadata fixes `islay.arch=b8c64-v1`, `islay.encoding=relative-2x8x8-a1-v1`,
`islay.policy=a1-h8-pass65-v1`, `islay.value=relative-wdl-v1`, plus checkpoint SHA256 and
training steps. C++ checks these declarations, names, dynamic shapes and finite outputs;
metadata does not authenticate the weights. Load only trusted checkpoints/ONNX files.

## Python setup and export

Tested with Python 3.14.5, PyTorch 2.12.0, ONNX 1.22.0, onnxscript 0.7.1 and ORT 1.29.0.
Direct dependencies are pinned in `training/requirements.txt`; transitive dependencies are
not a lockfile. Install into a project-local environment (no global install needed):

```sh
python3 -m venv _build/nn-venv
_build/nn-venv/bin/python -m pip install -r training/requirements.txt
_build/nn-venv/bin/python training/model_test.py
_build/nn-venv/bin/python training/export.py --init-random --seed 20260909 --output _build/model-b
```

Each output directory must be new. Export produces one `model.onnx`, `manifest.json` and,
for explicit random initialization, `random-init.pt`. No weights are committed.
To export an existing checkpoint:

```sh
_build/nn-venv/bin/python training/export.py --checkpoint /path/to/checkpoint.pt --output _build/model-export
```

Checkpoint schema: a dictionary with `schema=1`, `architecture="b8c64-v1"`, nonnegative
integer `training_steps`, and the model's complete `state_dict`. Loading uses
`weights_only=True` and strict state matching. Future optimizer/RNG/replay fields can be
added without changing the inference contract. A nonzero step count alone proves no strength.
Export uses dynamo, opset 18, dynamic batch and no external tensor files, following the
[PyTorch ONNX exporter](https://docs.pytorch.org/docs/2.12/onnx.html).

## C++ build and use

Default builds keep `ISLAY_ONNX=OFF`; UCI/perft need no Python or ONNX dependency.
For inference, unpack the platform-matching official
[ONNX Runtime C++ distribution](https://onnxruntime.ai/docs/get-started/with-cpp.html),
then point CMake at the directory containing `include/` and `lib/`:

```sh
cmake -S . -B _build/nn -DCMAKE_BUILD_TYPE=Release -DISLAY_ONNX=ON \
  -DONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime-distribution
cmake --build _build/nn -j
_build/nn/islay
```

This run used `onnxruntime-osx-arm64-1.29.0.tgz` from the
[official v1.29.0 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0),
SHA256 `d0706fc34f315d8c88639d0a8c81f2e09e815f282cabed3493c06a054352cf92`.
The extracted library must remain available at runtime. CMake does not download it.
Other platform/runtime combinations require their own parity and runtime validation.

```text
uci
setoption name EvalFile value _build/model-b/model.onnx
isready
position startpos
go nodes 800
```

Wait for `bestmove`, or send `stop` before `quit`. Loading emits the normal option reply;
each search advertises the backend, checkpoint identity and step count, explicitly marking
zero-step initialization. `<empty>` clears the model. Reload is transactional and synchronous:
failure keeps the old evaluator; setting the same filename loads it again. See [UCI.md](UCI.md).

The CPU session is retained across searches; the input buffer is reused, with no neural cache.
Intra/inter-op threads are explicitly one, execution sequential and spinning disabled,
following [ORT threading controls](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).
The search worker owns each call; there is no simultaneous evaluation on that session.
`stop` terminates an active ORT Run through its cancellation API. Deadlines are checked between
calls, so a timed search can overrun by its current evaluation. Perft remains single-threaded.

## Reproduce validation and measurements

```sh
printf 'debug on\ntest\nposition startpos\ngo perft 8\nquit\n' | _build/nn/islay
python3 tools/uci_search_test.py _build/nn/islay
_build/nn-venv/bin/python training/verify.py \
  --checkpoint _build/model-b/random-init.pt --model _build/model-b/model.onnx \
  --probe _build/nn/islay_nn_probe --engine _build/nn/islay --output _build/nn-results
```

The verifier checks 18 opening/middlegame/endgame/pass positions under all eight
D4 transforms plus 128 one-bit encodings (272 inputs), and batch 1/8/32/128.
It also exports a synthetic four-update checkpoint to exercise non-identity BatchNorm;
rejects wrong metadata/shapes, NaN weights, corrupt/missing files; exercises reload,
failed-load retention, clearing models, 20 active-search stops, terminal bypass and perft 8.
Results and invalid fixtures are written only under the fresh output directory.

The initial random-model parity maximum absolute error was **6.11e-7** against PyTorch CPU
(required atol 2e-5, rtol 1e-4). MPS parity uses atol 1e-4, rtol 1e-3.
Native ONNX and default-backend protocol suites passed. Perft 8 remains **390,216**.
The four-update checkpoint also passed export parity (maximum absolute error **2.31e-7**).
Release `ISLAY_NATIVE=OFF` passed aggregate/perft and both protocol suites on the same
arm64 machine (not an x86/Linux runtime test). ASan+UBSan Debug/portable passed aggregate,
perft, the neural multi-position/reload/stop suite and batch-128 inference. Sanitizers cover
Islay's wrapper/core; the downloaded ORT library itself is prebuilt and uninstrumented.
ThreadSanitizer was not rerun for P3. Sanitizer build configuration:

```sh
cmake -S . -B _build/nn-sanitize -DCMAKE_BUILD_TYPE=Debug \
  -DISLAY_NATIVE=OFF -DISLAY_LTO=OFF -DISLAY_ONNX=ON \
  -DONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime-distribution \
  -DCMAKE_CXX_FLAGS=-fsanitize=address,undefined \
  -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined
cmake --build _build/nn-sanitize -j
```

### Initial throughput sample, 2026-09-09

Apple M3 Pro, AppleClang 21, Release/native/LTO, FP32, CPU one compute thread.
Random model SHA256 `cdd2d8cfc9c58316494c3332c6afd13ef67ff5a906789c7448fba8ad53d8c486`.
31 timed calls per backend/batch after warm-up; values below are from the first full run.
Latency is per batch, not per position. P95 uses sorted index floor((n-1)*0.95).

| Backend | Batch | Median µs | P95 µs | Positions/s | Process peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| C++ ORT optimized | 1 | 897 | 954 | 1,115 | 38.1 |
| C++ ORT optimized | 8 | 7,229 | 7,288 | 1,107 | 38.2 |
| C++ ORT optimized | 32 | 30,235 | 30,394 | 1,058 | 39.1 |
| C++ ORT optimized | 128 | 124,422 | 241,578 | 1,029 | 48.7 |
| PyTorch CPU | 1 | 1,041 | 1,078 | 960 | — |
| PyTorch CPU | 8 | 5,737 | 5,868 | 1,394 | — |
| PyTorch CPU | 32 | 31,638 | 31,853 | 1,011 | — |
| PyTorch CPU | 128 | 117,088 | 118,113 | 1,093 | — |
| PyTorch MPS | 1 | 2,059 | 4,254 | 486 | — |
| PyTorch MPS | 8 | 1,767 | 3,156 | 4,527 | — |
| PyTorch MPS | 32 | 1,945 | 2,727 | 16,453 | — |
| PyTorch MPS | 128 | 3,756 | 4,492 | 34,080 | — |

C++ includes board encoding, binding, inference, output checks and copying. PyTorch measures
forward only on preloaded tensors; MPS synchronizes every timed call, but excludes host/device
transfers. RSS is the whole C++ probe's process high-water mark, not engine or GPU memory;
Python/MPS memory was not measured. These are not apples-to-apples end-to-end backend rankings.
Large-batch p95 variation illustrates why this pilot should not be treated as a stable bound.

The verifier also measures ORT optimizations disabled/enabled and five alternating-order
paired rounds at batch 1, the current PUCT workload. Inspect `paired_batch1` in results JSON.
Those five rounds gave a median paired throughput difference of **+1.60%** with graph
optimizations enabled (range -0.87% to +4.87%); this is a small/noisy pilot improvement,
not a general speed guarantee. Enabled remains the reference default. A second full run
measured 1,035 positions/s at C++ batch 1, versus 1,115 initially. Its 20 active-search stop
round trips had median **208 µs**, p95 **598 µs**, including protocol/Python scheduling.
No shared-tree parallelism, GPU C++ backend or search batching is implemented yet.
The useful next step is self-play/replay correctness, then batching independent games to
exploit GPU throughput. Inference positions/s are **not search NPS, perft NPS or Elo**.
No paired arena strength result exists for these untrained weights.
