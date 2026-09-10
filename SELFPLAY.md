# Self-play and replay (P4)

The offline pipeline uses the same C++ Othello rules and PUCT as UCI, with a fixed
8x64 neural checkpoint. Python manages snapshots, persistence, validation and batch loading;
it is not a second production move generator. No teacher labels, random leaf rollouts,
resignation or reward shaping are used. No trained-strength claim follows from throughput.

## Generate and resume

Build with `ISLAY_ONNX=ON` and export a model as described in [NEURAL.md](NEURAL.md).
CMake also builds `islay-selfplay`. The wrapper requires the pinned training dependencies
and a POSIX filesystem (tested on macOS/APFS; Linux runtime not tested).

```sh
_build/nn-venv/bin/python training/selfplay.py \
  --engine _build/nn/islay-selfplay --model _build/model-b/model.onnx \
  --output _build/selfplay-b --games 100 --simulations 128
```

`--games` is a **total target**. Run the same command with `--games 200` to extend to 200,
not add 200 more. Completed shards are verified and never overwritten. One persistent
C++ process generates the missing games in ascending ID order. The model session is
loaded once per run, not once per move/game/shard. Perft remains single-threaded.

| Option | Default | Meaning |
|---|---:|---|
| `--seed` | 20260909 | Master seed; independent deterministic stream per game ID |
| `--simulations` | 128 | Completed PUCT simulations per move, excluding root expansion |
| `--tree-mib` | 8 | Arena cap; an incomplete/memory-limited search aborts the game |
| `--cache-mib` | 16 | Exact neural-output cache budget; 0 disables it |
| `--epsilon` / `--alpha` | 0.25 / 0.3 | Root Dirichlet mixture fraction and concentration |
| `--temperature-plies` | 16 | Sample visits with tau=1 before this ply, then tau=0 |
| `--shard-games` | 16 | Complete games per atomic shard, at most 1024 |
| `--max-mib` | 256 | Dataset byte cap including model snapshot and abandoned partial files |

Every pass consumes a ply and flips the mover perspective. The policy is
`visits / sum(visits)` at tau=1; tau=0 is one-hot on the most-visited action, with lowest
action index breaking ties. There is always at least one completed simulation. Policy is
reconstructed from exact uint32 visits; it is not rounded to a stored float target.
The original legal neural prior is stored separately from root-noise priors.

Noise is applied once to the root only, after legal softmax and outside the evaluation
cache. Terminal nodes bypass inference; forced pass has probability one. UCI search
does not enable root noise. The current self-play loop is deliberately one owner and one
CPU inference thread: there is no concurrent shared tree, tree reuse or GPU batching yet.

The cache is a fixed power-of-two direct-mapped table. Every hit compares both complete
64-bit bitboards, not merely a hash/tag; collisions cause replacement, never wrong hits.
It belongs to one immutable model session, supports Othello only, and stores raw logits/value,
not visits or noise. Requested 16 MiB gives **9 MiB actual entries** on the tested ABI.
Cache size is not part of replay semantics; cache on/off and resume must preserve bytes.

## Data integrity and reproducibility

The manifest fixes schema/encoding/rule, full ONNX SHA256, declared checkpoint identity,
engine binary SHA256, ORT version, platform, seed and semantic search configuration.
The loaded model is a private snapshot under the dataset; its SHA is rechecked before
publishing each shard. Treat committed datasets/model snapshots as immutable.

RNG v1 is SplitMix64 with a documented per-game seed derivation, Box–Muller normals and
Marsaglia–Tsang gamma sampling. Reproducibility means the **same binary, runtime, platform,
model and semantic configuration**, not guaranteed cross-platform libm/ONNX bit identity.
Resume rejects mismatches. Cache size, target game count and shard size can change.
A dataset identity plus game ID is the sample namespace when combining future generations.

One C++ game is buffered until terminal and validated before entering the output pipe.
Python groups complete games in a staging directory, fsyncs the samples, asks the C++
verifier to replay every transition/outcome, writes a SHA256 index, then atomically renames
the directory and fsyncs its parent. Readers see either the old dataset or a complete new
shard. SHA256 detects corruption; it is not an authenticity/signature mechanism.

Only one producer can hold the dataset's OS file lock. `Ctrl-C` terminates the child and
discards the active staging shard; already committed shards remain usable. A hard kill
can leave `.pending-*` directories: they are never replay data and are counted toward
the byte cap. They are not automatically deleted. Resume regenerates their game IDs.
A crash during initial manifest/model installation may require a fresh directory; an
incomplete initialization is rejected rather than guessed. No replay eviction is automatic.

The byte cap reserves the worst-case size of the next shard before consuming it; increase
the cap or lower shard size if that conservative reservation cannot fit. A disk/write/search
failure never turns a truncated game into a draw. Completed games in an unpublished shard
may be regenerated after interruption. Filesystem/power-loss durability outside tested APFS
and ordinary fsync semantics is not claimed.

Layout:

```text
dataset/
  manifest.json
  model.onnx
  .lock
  shard-000000000000/
    index.json       dataset identity, start/count, game lengths, sample SHA256
    samples.bin      fixed-width little-endian records
  shard-000000000016/
    ...
```

Each **560-byte** record has the following exact offsets (no native-struct serialization):

| Offset | Field | Type |
|---:|---|---|
| 0 | game ID | uint64 |
| 8 / 16 | mover / opponent discs | uint64 each |
| 24 | legal placement mask | uint64 |
| 32 | root visits, a1..h8/PASS | uint32[65] |
| 292 | original legal network prior | float32[65] |
| 552 | ply | uint16 |
| 554 / 555 | chosen action / side (Black=0, White=1) | uint8 each |
| 556 | final mover-relative outcome | int8, -1/0/+1 |
| 557 / 558 / 559 | tau-one flag / legal-pass flag / reserved zero | uint8 each |

The internal producer pipe has a 32-byte frame per game (`ISLAGM01`, uint64 game ID,
uint64 derived seed, uint32 sample count, uint32 reserved zero), followed by those records.
This is not a public UCI command. Use the Python wrapper for persistence and resume.

## Fast batch replay

```python
import sys
import numpy as np
sys.path.insert(0, "training")
from replay import Replay

replay = Replay("_build/selfplay-b")
rng = np.random.default_rng(42)
batch = replay.torch_batch(128, rng, split="train", augment=True)
# board [128,2,8,8], legal/policy [128,65], outcome [128,1]
# also priors, action, game_id, ply for diagnostics
```

The loader verifies checksums/index boundaries once, maps records read-only and gathers
only the sampled records. Sampling is uniform over positions with replacement. Input
planes are unpacked as a batch, not via 128 Python per-square loops. D4 augmentation maps
input, legal mask, target policy, prior and chosen action together; PASS is invariant.
[NumPy memmap](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) avoids
loading all sample payloads, while
[unpackbits](https://numpy.org/doc/stable/reference/generated/numpy.unpackbits.html)
provides explicit little-bit ordering matching a1=0.

An independent deterministic hash assigns about 10% of games to validation before
augmentation; every position/orientation from one game stays in one split. Small datasets
can have an empty split and sampling it fails explicitly. Index memory is O(samples),
payload is mmap-backed, and a batch is capped at 65,536 samples. The default directory cap
is intended for a pilot, not an unlimited replay window. `Replay` is a fixed snapshot;
reopen it to see newly committed shards. The [trainer](TRAINING.md) adds a bounded whole-game window.

## Validation and measurement

```sh
printf 'debug on\ntest\nposition startpos\ngo perft 8\nquit\n' | _build/nn/islay
python3 tools/uci_search_test.py _build/nn/islay
_build/nn/islay-selfplay --selftest
_build/nn-venv/bin/python training/selfplay_test.py \
  --engine _build/nn/islay-selfplay --model _build/model-b/model.onnx --output _build/p4-tests
_build/nn-venv/bin/python training/replay_bench.py \
  --engine _build/nn/islay-selfplay --model _build/model-b/model.onnx \
  --replay _build/p4-tests/replay --output _build/p4-bench
_build/nn-venv/bin/python training/selfplay_recovery_test.py \
  --engine _build/nn/islay-selfplay --model _build/model-b/model.onnx \
  --reference _build/p4-tests/replay --output _build/p4-recovery
```

The real-network smoke used **16 simulations/move**, not the default 128: 100 games,
6,042 positions, 45 forced passes, 93 train games / 7 validation games. Independent
square-scanning replay checked every move/outcome. Resume and cache on/off were byte-identical;
wrong config, byte cap, corrupt fields/checksum, partial record and truncated game were rejected.
All eight augmented orientations matched reference transforms; a replay batch produced
finite policy/value loss and gradients. This is pipeline correctness, not a trained model.
Native and portable Release plus ASan/UBSan Debug passed aggregate/perft checks. Portable
and sanitizer builds also passed the four-game neural generation/resume/corruption suite.
The ORT distribution is prebuilt and uninstrumented; sanitizer coverage applies to Islay.
A separate hard-kill test killed the isolated producer group after its first committed shard,
then verified lock exclusion, resume without duplicate IDs, unchanged committed files and exact
sample equality against the reference. UCI lifecycle tests and perft(8)=390,216 still pass.

Benchmark methodology: Apple M3 Pro, Release/native/LTO, ORT 1.29.0 FP32, one compute thread,
untrained model B from P3. Five alternating-order cache-off/on pairs, four complete games per
side per pair, different game IDs across pairs, 32 simulations/move. Full binary output must
match within each pair. Wall time includes process/model startup and pipe transfer; no UCI,
disk publication or extra search workers. Replay timing alternates scalar/vectorized decoding
of identical records, 31 retained samples after two warm-ups, batch 128/1024. The separately
reported sample+augment path also includes RNG, mmap gathering and D4. Warm page-cache numbers
are not cold-storage or GPU-transfer throughput. See generated `results.json` for each round.

Measured 2026-09-09, model SHA256
`cdd2d8cfc9c58316494c3332c6afd13ef67ff5a906789c7448fba8ad53d8c486`:

| Workload | Baseline median | Optimized median | Change |
|---|---:|---:|---:|
| 4 full games, 32 simulations/move, cache off/on | 7.449 s | 5.544 s | median paired throughput **+33.02%** |
| Decode 128 already-gathered records | 9,345 µs | 64.46 µs | 145.0x |
| Decode 1,024 already-gathered records | 75,543 µs | 255.67 µs | 295.5x |

The five self-play paired gains were +37.99%, +33.02%, +41.18%, +27.56%, +27.38%, with
**identical output bytes** in every pair. Cache hits eliminated roughly 23–24% of neural calls.
The cache-on median corresponds to about **2,597 games/hour** for this 32-simulation pilot,
including startup/pipe overhead; it is not a forecast for the default 128 simulations or a
trained checkpoint. The paired median gain differs from the ratio of the two aggregate medians.

The full warm replay path (sample + mmap gather + decode + D4) took **510.6 µs / 128 samples**
(p95 653.9 µs; 250,673 samples/s), and **1,853.5 µs / 1,024 samples** (p95 2,011.0 µs;
552,468 samples/s). Decode-only ratios compare against the straightforward per-record Python
reference, not an existing optimized dataloader. Training, GPU copies, checksum verification
at open, and cold disk reads are excluded. The 100-game corpus occupies about 5.63 MiB including
its ONNX snapshot and indexes; sample payload alone is 3,383,520 bytes.

The [P5 trainer/checkpoint loop](TRAINING.md) now consumes these replay shards. Arena gating
and batching independent games for GPU throughput remain future steps.
These low-budget bootstrap games and throughput measurements do not establish Elo or the
quality/optimality of the chosen exploration settings.
