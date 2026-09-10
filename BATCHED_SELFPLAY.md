# Shared-model batched self-play (P7a)

The opt-in P7 producer keeps **one FP32 model owner**, while independent C++ processes run
single-owner PUCT trees. POSIX shared memory carries encoded boards, network outputs and
completed replay frames. Private socket pairs carry only small control notifications.
Workers do not load weights or link ONNX Runtime; the UCI engine, MCTS core and single-threaded
perft remain unchanged. The serial [P4 producer](SELFPLAY.md) stays available as the reference.

## Run

Build using [NEURAL.md](NEURAL.md), with the pinned Python dependencies. POSIX builds now also
produce `islay-selfplay-worker`. Only the verifier/native reference requires ONNX support;
the new worker itself has no neural runtime dependency.

```sh
_build/nn-venv/bin/python training/batched_selfplay.py \
  --worker _build/nn/islay-selfplay-worker \
  --verifier _build/nn/islay-selfplay \
  --checkpoint _build/train-b/step-000002000/state.pt \
  --model _build/trained-b/model.onnx \
  --backend torch-mps --workers 32 --games 128 \
  --simulations 128 --output _build/batched-replay
```

The checkpoint and its exported ONNX file must match by checkpoint SHA256, training-step
metadata and architecture. Startup loads/validates both and compares fixed-shape inference
against ONNX Runtime on 132 deterministic boards plus padding. Missing MPS, mismatched assets
or a numerical error outside atol 2e-5 / rtol 1e-4 rejects generation; no backend fallback.
Models must be trusted single-file artifacts (checkpoint <=32 MiB, ONNX <=64 MiB).

`ort-cpu` is the default numerical-reference backend; `torch-cpu` and `torch-mps` are explicit
alternatives. All use FP32, contiguous tensors, eval mode, one CPU intra-op thread and no AMP,
compilation, quantization or concurrent model calls. Torch uses
[inference mode](https://docs.pytorch.org/docs/2.12/generated/torch.autograd.grad_mode.inference_mode.html).
The shared model is an in-process runtime object, not a set of independently loaded sessions
whose serialized model files happen to be mapped into shared memory.

Defaults: 16 workers, 128 games, 128 simulations/move, 1 MiB tree and 1 MiB neural cache per
worker, seed 20260909, root noise fraction 0.25 / alpha 0.3, temperature prefix 16 plies.
Worker count is bounded to 1–128. `--worker-mib` defaults to 512 and limits configured tree/cache
capacities plus shm payload; it does **not** include Python/Torch/GPU runtime, process stacks,
allocator overhead or actual total RSS. `--max-mib` defaults to 256 for on-disk data/assets.

## Deterministic cohorts, not a shared tree

```text
one C++ tree per worker ── shm board planes ──► one fixed-shape model batch
                       ◄─ shm logits/value ──
completed whole cohort ──► C++ replay verifier ──► atomic replay shard
```

Each worker has at most one pending leaf. The server waits for every active worker to submit
one request or finish its game, evaluates a batch of the **fixed configured width**, then
acknowledges all pending workers. Finished slots are zero-padded; their outputs are ignored.
No scheduling-dependent microbatch timeout selects batch shape, and there is no virtual loss
or parallel mutation of one tree. Socket waits block rather than busy-spin.

Game IDs map to fixed slots in contiguous cohorts. Per-game RNG is unchanged from P4. Neural
caches are cleared at each game boundary, so a resumed cohort does not depend on cache entries
from earlier games. `--games` must be a multiple of `--workers`; one whole cohort becomes one
shard. There is deliberately no partially filled final cohort or immediate replacement of a
finished game with a new one. This trades tail utilization for reproducible batch shape and
bounded recovery. Metrics distinguish real `neural_positions` from `padded_positions`.

Repeat the same command with a larger whole-cohort `--games` target to resume/append. All
completed shard bytes remain immutable. Model/ONNX hashes, backend, batch width, worker and
verifier binaries, generation config, Python/runtime versions and source fingerprints are
checked before resume. Changing backend or cohort width requires a new dataset. Exact replay
is only promised by tests on the same hardware/runtime/settings; cross-backend or cross-batch
floating-point identity is **not** assumed. Small numerical differences can change visits,
actions and eventual self-play outcomes; these throughput tests do not establish Elo.

## Shared memory and failure boundaries

The owner creates a random private POSIX shm object with mode 0600, unlinks its name immediately,
and passes the open descriptor only to owned workers via `pass_fds`. Both sides map shared pages.
The mapping stays valid after the name is removed and disappears when the last mapping/descriptor
is released; see Apple's [shm_open](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/shm_open.2.html)
and [shm_unlink](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/shm_unlink.2.html).
No named semaphore, public socket listener, reusable global model key or Python resource-tracker
process is needed. Darwin's page-rounded shm size is accepted with a bounded size check.

Private protocol v1 has 72,504 payload bytes per worker: 128 input floats, 66 output floats,
a maximum 128-ply replay frame and two uint64 cache counters. Arrays are separate contiguous
regions, not serialized Python objects. Local peers must be little-endian with FP32 floats.
Release/acquire fences and socket request/acknowledgement establish ownership handoffs; a
worker cannot overwrite an input while the model is consuming it. This is not a network
service and is not designed to accept untrusted external clients.

The controller rejects malformed messages, frames, overlapping/bad board encodings and
cache/request count mismatches. Native workers and the model owner both reject non-finite
outputs. EOF, crash or `--timeout-ms` (default 30,000) aborts the cohort; no partial game/cohort
is published. The timeout bounds protocol waits, not a stuck GPU driver inside a framework
call. External interruption may be needed for a driver-level hang.

On ordinary failure the owner closes channels and terminates/reaps its workers. An external
hard-kill releases its descriptors; workers detect EOF or their request timeout, and anonymous
shared pages disappear after owners exit. A lock prevents two producers writing the same dataset.
Incomplete `.pending-*` shards are ignored on resume but count toward disk capacity; nothing
is automatically evicted. Incomplete initial model/manifest publication requires a fresh directory.
Input files are hashed before/after model load, copying and each cohort publication.

The binary record format, game-level split and legal-masked visit targets stay P4-compatible.
Every published shard is checked by the production C++ replay verifier, including terminal
outcomes and forced passes. Existing `Replay`, D4 decoding and [training](TRAINING.md) consume
these datasets without conversion. The old P4 generator does not resume a P7 manifest.

## Measuring shm model loading versus shared inference

`training/model_load_bench.py` compares warm-cache file reads with reading already-populated
shm. Both feed identical bytes into a **new** one-thread ORT session and perform a batch-1
warm-up. It excludes the initial copy into shm, imports and truly cold physical-disk reads.
Seven alternating paired rounds each retain the median of three repetitions, after a warm round.
This isolates byte-source cost; it does not pretend that sharing serialized bytes also shares
the parsed graph, packed weights or runtime session.

`training/batch_bench.py` compares the serial C++ producer with shm ORT batch 8 and Torch MPS
batches 8/16/32. Each round starts the same game IDs/seed/model and uses equal simulations per
move. It warms one full round, then retains five alternating-order rounds of 32 games each.
Every raw replay is C++-verified; each variant must reproduce its own replay hash across rounds.
Cross-variant hashes need not match. This is a throughput comparison, not an arena trial.

P7's one-time model load/parity is excluded and reported separately; worker startup, shutdown,
IPC and game generation are timed. The native reference includes its one-time model session
initialization and retains cache entries between games; P7 clears per game for recovery.
Disk writes and post-generation verification are outside these benchmark timings. Real dataset
CLI `seconds` includes workers, snapshot checks, replay verification and shard publication,
but excludes initial model/parity and input snapshot setup. For short jobs, include startup too.

2026-09-10, Apple M3 Pro, Release/native/LTO worker, ORT 1.29.0, Torch 2.12.0, architecture B
checkpoint at 80 updates. At **8 simulations/move**, 32 games per variant, five paired rounds:

| Path | Median games/hour | Median paired speedup vs native |
|---|---:|---:|
| Native C++ ORT, serial | 10,773 | 1.00x |
| SHM ORT CPU, 8 workers | 10,054 | 0.926x |
| SHM Torch MPS, 8 workers | 29,813 | 2.694x |
| SHM Torch MPS, 16 workers | 58,624 | 5.485x |
| SHM Torch MPS, 32 workers | 106,630 | 10.040x |

MPS 32 was best **among these tested widths**. CPU batching regressed about 7.4% and remains a
verification path, not a claimed speed improvement. These are low-budget warm throughput
measurements, not sustained production capacity or a strength result. Each variant's records
were stable across all five rounds, but backend/layout arithmetic is not assumed interchangeable.
Detailed raw replays and metrics are local in `_build/p7-benchmark`.

For the 2,517,427-byte ONNX file, a separate seven-round warm-load confirmation measured median
byte acquisition **621.9 us file / 52.25 us shm** and median read+session+warm-up **4.093 ms /
3.583 ms**. Median paired total-load speedup was **1.175x** (an earlier pass was 1.128x).
Load timing had outliers; only fractions of a millisecond are saved in typical initialization.
Consequently, production P7 shares the **model owner** and batches requests, rather than
creating one runtime session per worker backed by shared serialized model bytes.

## Tests

```sh
_build/nn-venv/bin/python training/batched_selfplay_test.py \
  --worker _build/nn/islay-selfplay-worker --verifier _build/nn/islay-selfplay \
  --model _build/trained-b/model.onnx --checkpoint _build/train-b/step-000002000/state.pt \
  --output _build/batch-tests
_build/nn-venv/bin/python training/batch_fault_test.py --worker _build/nn/islay-selfplay-worker
_build/nn-venv/bin/python training/batch_recovery_test.py \
  --worker _build/nn/islay-selfplay-worker --verifier _build/nn/islay-selfplay \
  --model _build/trained-b/model.onnx --checkpoint _build/train-b/step-000002000/state.pt \
  --backend torch-mps --output _build/batch-recovery
_build/nn-venv/bin/python training/model_load_bench.py \
  --model _build/trained-b/model.onnx --output _build/model-load-bench
_build/nn-venv/bin/python training/batch_bench.py \
  --worker _build/nn/islay-selfplay-worker --verifier _build/nn/islay-selfplay \
  --model _build/trained-b/model.onnx --checkpoint _build/train-b/step-000002000/state.pt \
  --rounds 5 --games 32 --simulations 8 --output _build/batch-bench
```

Tests cover native C++ ORT versus shm ORT batch-1 byte parity; fixed-cohort resume for ORT,
Torch CPU and MPS; independent scalar replay oracle; corrupted shards, config mismatch,
partial-cohort requests and storage/worker caps; NaN/model exceptions, worker crash/stall and server disappearance.
Hard-kill recovery verifies immutable committed shards and byte-identical continuation.
At the measured 32-worker width, a separate 32→64-game resumed run also matched the first
64 games of a continuous run byte for byte, at 32 simulations/move.
Native ASan/UBSan worker builds exercise both inference faults and real-network games.
The rebuilt native ONNX engine passed the 272-position export/parity gate (max error 3.49e-6),
updated-checkpoint/BatchNorm parity (3.76e-6), model rejection and UCI stop/lifecycle tests.
macOS/Apple Silicon is runtime-tested; Linux and other devices remain unvalidated.

A final integration dataset used MPS / 32 workers / 32 simulations and completed **128 games,
7,613 records, 74 forced passes**, with **6,824 training / 789 validation positions**.
Its maximum startup MPS/ORT parity error was **5.96e-7**. The unchanged trainer consumed the
dataset directly and completed two finite-gradient updates, carrying the parent counter from
80 to 82. These checks validate the data/learning path, not playing-strength improvement.
Artifacts are local under `_build/p7-replay-final` and `_build/p7-train-smoke`; no generated
checkpoint was promoted or deployed.

Subtree reuse, mixed precision, fused Torch inference and scheduling-based microbatch queues
remain separate experiments. Do not assume the largest possible worker count is best.
