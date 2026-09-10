# Policy/value training (P5 trainer)

The trainer consumes validated P4 replay and updates the fixed **8x64 shared policy/value
network**. It provides bounded replay windows, FP32 CPU/MPS training, full-state resume,
atomic checkpoints and export compatibility. It does not choose a champion or infer Elo;
paired evaluation and offline selection are handled by the [P6 arena](ARENA.md), without automatic deployment.

## Train, resume, export

Use the environment/dependencies from [NEURAL.md](NEURAL.md) and a completed replay dataset
from [SELFPLAY.md](SELFPLAY.md). The CPU path is the reproducibility reference. On the tested
Apple M3 Pro, explicitly select MPS for faster training:

```sh
_build/nn-venv/bin/python training/train.py \
  --replay _build/selfplay-b --verifier _build/nn/islay-selfplay \
  --init _build/model-b/random-init.pt --output _build/train-b \
  --device mps --steps 1000
```

`--steps` is the target number of optimizer updates **in this run**, not an increment.
The output directory must be fresh unless resuming. To continue the same run:

```sh
_build/nn-venv/bin/python training/train.py \
  --replay _build/selfplay-b --verifier _build/nn/islay-selfplay \
  --resume --output _build/train-b --device mps --steps 2000
```

Use identical replay/config/device/layout/code/runtime when resuming. Omitted `--device`
uses the config's CPU default, so keep the MPS override on resume. Unknown config fields,
NaN/out-of-range settings, empty train/validation splits, incompatible state, modified data
or corrupted committed checkpoints are rejected. There is no silent GPU→CPU fallback.
Run directories and committed replay/model files are immutable except for appending new
checkpoints and updating the convenience latest pointer.

Export a committed checkpoint with the existing exporter:

```sh
_build/nn-venv/bin/python training/export.py \
  --checkpoint _build/train-b/step-000002000/state.pt --output _build/trained-b
```

The resulting ONNX file loads through UCI `EvalFile` or `training/selfplay.py --model`.
The numeric `training_steps` metadata is cumulative across parent checkpoints, but is not
proof of playing strength. No engine default model is changed by training/export.

For a **new learning iteration**, generate fresh self-play with an exported checkpoint,
then use `--init` with that checkpoint and a fresh output directory. This carries weights
and cumulative training steps, but intentionally resets optimizer, scheduler and RNG for a
new run/data window. `--resume` instead restores all of those and refuses a changed replay
snapshot. Do not use `--init` as a substitute for interrupted-run resume.

## Versioned configuration

`training/config_b.json` is the baseline; pass an edited copy with `--config`.
These are pilot hyperparameters, not proven optimal Othello settings.

| Setting | Default | Meaning |
|---|---:|---|
| `batch_size` | 128 | Positions/update, sampled uniformly with replacement |
| `learning_rate` | 0.01 | Initial SGD learning rate |
| `momentum` | 0.9 | SGD momentum; no Nesterov/dampening |
| `weight_decay` | 0.0001 | L2 gradient coefficient on all parameters, including BN/bias |
| `milestones` / `lr_gamma` | [1000,3000] / 0.1 | Multiply LR after these completed run updates |
| `grad_clip` | 5.0 | Global L2 norm cap on data-loss gradients before optimizer weight decay |
| `seed` | 20260910 | Python, explicit NumPy sampler, Torch/device RNG |
| `max_games` | 10000 | Keep the latest whole games from the supplied source order |
| `max_replay_mib` | 256 | Cap all input sample payloads before window indexing |
| `cpu_threads` | 1 | PyTorch CPU intra-op threads; does not alter perft |
| `device` / `memory_format` | cpu / contiguous | Explicit execution/layout; CLI overrides available |
| `augment` | true | D4 training augmentation; validation is not augmented |
| `checkpoint_every` | 100 | Persist model/optimizer/RNG and interval metrics |

The objective is legal-masked policy cross-entropy plus mover-relative outcome MSE.
SGD weight decay adds the gradient of `(weight_decay / 2) * sum(parameter**2)` separately;
reported policy/value losses exclude that regularizer. Forced pass remains a single legal
policy action; illegal actions receive no target mass or policy gradient. The optimized
loss path reuses validated CPU labels and avoids repeated label-validation GPU synchronizations;
network outputs and gradient norm still receive finite checks each update, with finite
weights/optimizer checked before checkpoint publication. Updates use `zero_grad(set_to_none=True)`.
Optimizer/scheduler behavior follows the pinned
[PyTorch SGD](https://docs.pytorch.org/docs/2.12/generated/torch.optim.SGD.html) and
[MultiStepLR](https://docs.pytorch.org/docs/2.12/generated/torch.optim.lr_scheduler.MultiStepLR.html) APIs.

Everything stays FP32; no mixed precision, compilation, prefetch workers or tensor-precision
shortcuts are enabled. CPU `channels_last` is exposed for measurement but is not recommended
by the current results. MPS `channels_last` is explicitly rejected because backward failed
with a stride/view error in pinned PyTorch 2.12.0. MPS contiguous is validated. CUDA has an
explicit path with deterministic cuBLAS configuration and TF32 disabled, but has **not** been
runtime-tested on this machine; no CUDA speed or correctness result is claimed.

## Replay snapshot and validation

Repeat `--replay` in **oldest-to-newest** order to train a window spanning generations.
The loader verifies checksums and runs the production C++ replay verifier on complete games
before training. Duplicate dataset identities are rejected. The suffix window selects whole
games; no partial game crosses the boundary. Sampling is uniform over selected positions,
not uniform over games. Payloads remain mmap-backed and only batches are decoded/transferred.

Train/validation assignment remains P4's deterministic game-level split, before D4. All
orientations of a sample remain in the same split. A small suffix may have an empty split;
the trainer then asks for more data rather than silently evaluating on training games.
The snapshot records source manifests, ordered shard hashes/indexes and selected game IDs;
appending/changing sources after a run starts does not change its in-memory snapshot.
Resume rejects an altered snapshot. Create a new iteration to use fresh replay.

Each checkpoint logs policy loss, value loss, legal policy entropy, mean absolute prediction,
gradient norm, LR, update/cumulative step count and full-update microsecond timing.
Validation traverses **all** selected held-out positions without augmentation or replacement,
using eval mode and inference mode. It does not mutate BatchNorm or consume the training
sampler RNG. Global and opening/middle/end validation metrics are recorded (disc-count ranges
<24, 24–43, ≥44). These ranges are reporting bins, never teacher labels or search evaluation.
Run metadata also records mover-relative win/draw/loss sample counts by phase/split.

## Checkpoint guarantees and limits

```text
run/
  run.json
  .lock
  latest.json                 convenience pointer only
  step-000000000/
    state.pt                  initial full state
    metrics.json
    index.json                SHA256 of state and metrics
  step-000000100/
    state.pt
    metrics.json              only steps since the prior commit + validation
    index.json
```

`state.pt` retains export schema 1/`b8c64-v1`, plus trainer schema, weights/BN buffers,
optimizer, scheduler, run updates, cumulative steps, initial checkpoint SHA, config,
replay snapshot, code/runtime versions and Python/NumPy/Torch CPU/accelerator RNG states.
Tensor state is cloned to CPU before serialization. Loading uses `weights_only=True`;
use trusted artifacts regardless, because checksums are not signatures/authentication.

Publication writes a temporary directory, fsyncs its files, checks sizes/hashes and atomically
renames the directory, then fsyncs the parent. The latest pointer is updated separately;
resume scans and verifies committed directories numerically, so losing that pointer is safe.
It never silently falls back past a corrupt committed checkpoint. One producer holds an
OS file lock. An interrupted update is replayed from the last commit with the saved sampler
and optimizer state; half-updated in-memory parameters are never published.

Hard-killed `.pending-*` directories are ignored for resume but counted against `--max-mib`
(default 1024); no automatic eviction/deletion occurs. Each save reserves 32 MiB before writing
the 8x64 checkpoint. Lower checkpoint frequency or increase the cap for long runs. Failed
initialization without any committed state requires a fresh output directory. POSIX/APFS
publication is tested; other filesystems and Windows orchestration are not validated.

Reproducibility is scoped to the same code, config, replay, device/runtime and hardware.
Deterministic-algorithm checks are enabled, but cross-platform/backend floating-point identity
is not promised; see [PyTorch reproducibility notes](https://docs.pytorch.org/docs/2.12/notes/randomness.html).
CPU split/uninterrupted tests compare every weight, optimizer/scheduler field and RNG state.
MPS state capture uses its [RNG APIs](https://docs.pytorch.org/docs/2.12/mps.html), and timings
explicitly synchronize the accelerator. Changing CPU↔MPS or upgrading code is a new run,
not an exact resume.

## Tests and measured performance

```sh
_build/nn-venv/bin/python training/model_test.py
_build/nn-venv/bin/python training/train_test.py \
  --replay _build/selfplay-b --init _build/model-b/random-init.pt \
  --verifier _build/nn/islay-selfplay --output _build/train-tests
_build/nn-venv/bin/python training/train_recovery_test.py \
  --replay _build/selfplay-b --init _build/model-b/random-init.pt \
  --verifier _build/nn/islay-selfplay --output _build/train-recovery
_build/nn-venv/bin/python training/train_bench.py \
  --replay _build/selfplay-b --init _build/model-b/random-init.pt \
  --verifier _build/nn/islay-selfplay --output _build/train-bench
```

2026-09-10, Apple M3 Pro, PyTorch 2.12.0, NumPy 2.4.6, FP32, CPU one intra-op thread.
Each of five alternating-order rounds resets the same initial weights/sampler, warms three
updates and retains seven full updates per variant. Timing includes replay sampling/D4,
transfer, forward/backward, finite checks, gradient clipping, optimizer/scheduler and device
synchronization; excludes startup, replay verification, validation and checkpoint I/O.

| Backend/layout | Batch | Median update across rounds | Approx. samples/s |
|---|---:|---:|---:|
| CPU contiguous | 32 | 75.247 ms | 425 |
| MPS contiguous | 32 | 14.153 ms | 2,261 |
| CPU contiguous | 128 | 282.838 ms | 453 |
| MPS contiguous | 128 | 21.784 ms | 5,876 |

The median **paired** MPS/CPU speedups were **5.41x** at batch 32 and **12.92x** at batch 128.
CPU channels-last median paired throughput changed -0.58% / -0.17%, so it was not selected
as a faster default. One-update weights/BN differences versus CPU contiguous were at most
7.45e-9 (CPU channels-last) and 5.16e-5 (MPS), within declared atol 2e-4 / rtol 1e-3.
Backend arithmetic can diverge over training; numerical checks are not a strength comparison.

CPU resume and the CLI split/uninterrupted path were bit-identical across an LR milestone.
CPU channels-last resume also matched the complete uninterrupted state. Multi-generation
gather/window checks preserve whole games and split isolation; initializing a new iteration
preserves parent weights/counts while resetting optimizer/scheduler.
The same-backend MPS resume test observed zero maximum weight/BN error (required atol 1e-6,
rtol 1e-5), and identical scheduler/RNG. Checksum/config/data mismatch, duplicate replay,
byte-cap and partial-checkpoint tests passed. A hard-killed CPU run resumed to exactly the
same full state as a continuous run, preserving already committed files and lock exclusion.

A fixed eight-position training batch overfit in 120 CPU updates: total data loss
**3.0034 → 0.3431**, eval-mode loss **0.3431**. This checks learning/export-mode mechanics;
it is deliberately not a generalization or Elo result. See [P6](ARENA.md) for paired arena selection.

The integration smoke trained 64 MPS updates, resumed to 80, exported ONNX and checked
272 positions plus batches 1/8/32/128 against C++ ONNX Runtime. Maximum absolute output
error was **3.49e-6**; a further train/BatchNorm/export check stayed below **3.76e-6**.
The exported model then completed **20 fresh self-play games / 1,209 validated samples**
at 16 simulations per move. This exercises self-play → train/resume → export → self-play,
not an arena/champion result. Checkpoint and ONNX SHA256 respectively:

```text
5ae5ffb49d668600c29b2f31a7d2a13c3a19bf7b6dafc18fa266ced0304376bc
c6a553f7c739124369e9e528b58f72132219fd2419b803ebdf8d5385a57d89e5
```
