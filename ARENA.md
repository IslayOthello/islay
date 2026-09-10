# Paired candidate/champion arena (P6)

`training/arena.py` compares two trusted, single-file 8x64 ONNX models using the same
ONNX-enabled Islay executable. Each frozen opening is played twice, exchanging the models'
colors. Two persistent UCI processes retain their loaded models; only one searches at a time.
There is no shared search tree, concurrent inference, root noise or stochastic move selection.
Perft stays single-threaded and unchanged.

This is an offline experiment and **selection artifact**, not automatic deployment or an
autonomous training scheduler. A random initialization may serve as a bootstrap champion,
but beating it does not establish competitive strength or an absolute Elo rating.

## Run a fixed experiment

Use the pinned environment and ONNX build from [NEURAL.md](NEURAL.md), and export a candidate
with the [trainer](TRAINING.md). Freeze both models before choosing an arena seed.

```sh
_build/nn-venv/bin/python training/arena.py \
  --engine _build/nn/islay \
  --candidate _build/trained-b/model.onnx \
  --champion _build/model-b/model.onnx \
  --output _build/arena-nodes \
  --pairs 200 --mode nodes --budget 128
```

The default plan is 200 opening pairs / 400 games, 12 opening plies, seed 20260910,
8 MiB tree per engine, one-sided alpha 0.05 and required lower score bound >0.55.
All are frozen in the manifest before the first game. The executable must load ONNX;
there is no uniform-evaluator fallback. Core self-tests and perft(8)=390,216 must pass in
both processes before play. Every search checks its evaluator identity, full simulation
budget or time stop reason, legal PV and legal best move. Memory exhaustion is an error,
not a valid early completion of a simulation-budget game.

`--mode nodes --budget N` gives both models N completed simulations per move. Its
`--timeout-ms` watchdog defaults to 30,000 ms; this is a fault detector, not a game clock.
Run **a separate experiment** for equal requested time per move:

```sh
_build/nn-venv/bin/python training/arena.py \
  --engine _build/nn/islay \
  --candidate _build/trained-b/model.onnx \
  --champion _build/model-b/model.onnx \
  --output _build/arena-time \
  --pairs 200 --mode movetime --budget 20
```

`--budget` then means milliseconds. The engine checks deadlines between in-flight inference
calls; transport/scheduling also take time. `--grace-ms` defaults to 250 and is frozen too.
A response exceeding budget+grace invalidates the match. This is equal **requested movetime**,
not exact equal consumed wall time and not full match-clock control. Every move records actual
microsecond round-trip time, engine-reported milliseconds, simulations and evaluations; the
report includes per-model totals and median latency. Do not mix node/time-mode games into
one statistical result. Time-limited play is inherently scheduling-sensitive, including on resume.

For bounded sessions, add `--max-new-pairs 20`. This pauses after whole pairs without a final
selection. Resume with the same command, adding `--resume`; omit the batch limit to finish:

```sh
_build/nn-venv/bin/python training/arena.py \
  --engine _build/nn/islay \
  --candidate _build/trained-b/model.onnx \
  --champion _build/model-b/model.onnx \
  --output _build/arena-nodes \
  --pairs 200 --mode nodes --budget 128 --resume
```

Changing total pairs, budgets, seed, openings, models, executable, recorded runtime or arena
code rejects resume. Do not extend a finished favorable/unfavorable trial: plan a new independent
experiment instead. Multiple candidates/seeds/trials require a separately planned multiple-testing
budget; the per-run alpha is not a lifetime family-wise guarantee.

## Openings, rules and statistics

Openings are independent uniform-legal-action playouts from the standard start, sampled with
replacement at the configured length (8–40 plies), rejecting terminal starts. Their entire move
sequences, mover-relative boards and absolute side to move are recorded before any model is
queried. The seed is for reproducibility, not for choosing favorable openings. Sampling is not
uniform over all Othello positions. Duplicate/D4-equivalent boards are counted and retained,
not filtered adaptively; at least 90% D4-distinct openings are required for selection.

The independent scalar rule oracle in `tools/perft_study.py` validates all opening and game
actions, forced passes and terminal disc counts. It never supplies training labels or search
evaluation. Black/White identity is tracked separately from mover-relative boards. The first
game's candidate color alternates between opening pairs to reduce run-order bias.

The report contains candidate W/D/L, score (draw=0.5), five pair-score bins, logistic equivalent
Elo `400*log10(score/(1-score))`, paired bootstrap 95% intervals, and an approximate paired-normal
LOS. Resampling units are **whole opening pairs**, not individual games. Twenty thousand
multinomial draws over the five pair-score bins implement the same bootstrap efficiently;
see [NumPy multinomial](https://numpy.org/doc/2.4/reference/random/generated/numpy.random.Generator.multinomial.html).

Bootstrap/normal summaries are diagnostic, especially for tiny or degenerate samples. Zero
variance is flagged; non-tied degenerate LOS is `null`, and score endpoints produce string
`+inf`/`-inf` Elo, not a finite strength estimate. A separate conservative two-sided Hoeffding
score interval remains wide when all observed pairs have the same result.

Selection uses the mean pair score `p` and the one-sided lower bound
`max(0, p - sqrt(log(1/alpha)/(2*n)))`, derived for independent bounded pair scores from
[Hoeffding's inequality](https://www.stat.cmu.edu/~cshalizi/sml/21/lectures/06/lecture-06.html).
The opening sample and settings must be chosen independently of observed outcomes. This
bound concerns the specified opening distribution and match conditions, not universal strength.

`candidate_eligible` requires all planned pairs, at least 200 pairs, sufficient opening diversity,
the lower bound strictly above the predeclared threshold, distinct model hashes, and all
correctness/publication checks passing. Defaults are deliberately conservative and are not
claimed to reproduce the original AlphaGo Zero promotion rule. Any insufficient result says
`retain_champion`; that is **not** an H0/SPRT acceptance. No early statistical stopping is used.

## Storage, failures and selection

```text
run/
  .lock
  assets/
    manifest.json
    candidate.onnx
    champion.onnx
  pair-000000/
    pair.json                    both complete games and per-move metrics
    index.json                   SHA256 of pair.json
  final/
    report.json                  immutable summary and selected model hash/role
    index.json
```

The model copies and plan publish atomically as `assets/`. Each pair publishes only after
both games complete and independent replay/evaluator checks pass. Files/directories are
fsynced before atomic rename. One producer holds a file lock. Resume verifies checksums,
contiguous pair IDs, exchanged colors and complete game replays; it ignores unfinished
`.pending-*` directories and reruns their whole pair. Completed pair files are never overwritten.
Final reports are rederived and checked on resume, not trusted just because a file exists.

Crash, timeout, illegal move, failed load, memory limit or protocol error stops the match;
`failure.json` retains the error and completed-pair count, with no final selection. Failed
runs cannot silently retry; investigate and use a new experiment directory. An external
hard-kill without a recorded failure can resume from committed pairs. An interrupted initial
publication without `assets/` needs a fresh directory. No failed game is silently discarded
to improve W/D/L, and no failure is counted as a legitimate playing-strength loss.

`--max-mib` defaults to 256. Model snapshots, pair data, indexes and abandoned temporary files
count toward the cap. Each pair reserves 512 KiB including room for final reporting; one pair
payload is limited to 256 KiB. Nothing is evicted automatically. Checksums are integrity checks,
not authentication; use trusted files. POSIX/macOS transport, locks and publication are tested;
Windows orchestration is not supported. Keep external ONNX Runtime libraries and hardware
unchanged on resume: executable hashes and Python package versions do not cryptographically
pin the loaded C++ dynamic libraries. Node-mode exact replay was tested on the same machine/runtime only.

`final/report.json` records `selected_role` and the corresponding snapshot hash. It does not
modify UCI defaults, replace any user's checkpoint, or update an external champion registry.
After inspecting a successful trial, the selected snapshot can explicitly feed the next
[self-play](SELFPLAY.md) generation. Automated training scheduling and GPU self-play batching
remain separate work.

## Validation and throughput

```sh
_build/nn-venv/bin/python training/arena_test.py \
  --engine _build/nn/islay --model _build/trained-b/model.onnx --output _build/arena-tests
_build/nn-venv/bin/python training/arena_fault_test.py \
  --engine _build/nn/islay --model _build/trained-b/model.onnx --output _build/arena-faults
_build/nn-venv/bin/python training/arena_recovery_test.py \
  --engine _build/nn/islay --model _build/trained-b/model.onnx --output _build/arena-recovery
_build/nn-venv/bin/python training/arena_bench.py \
  --engine _build/nn/islay --candidate _build/trained-b/model.onnx \
  --champion _build/model-b/model.onnx --output _build/arena-bench --rounds 5 --budget 4
```

Tests cover forced pass, terminal/tie outcomes, opening replay/D4 diversity, paired statistics
and insufficient-sample gates, equal-model paired score 50%, frozen-plan rejection, checksum/
semantic corruption, whole-pair resume, hard-kill recovery and lock exclusion. Node-mode
split/continuous runs match moves, results and evaluation counts; wall times are not compared
for identity. Movetime mode has a separate completed smoke test. Fault injection checks EOF,
hung engine, oversized output, evaluator/search failure, memory stop, illegal and duplicate moves.

The benchmark alternates persistent sessions and a reference that restarts/reloads both models
each game. Both variants warm once, then play identical paired openings at 12/24/40 plies over
five rounds. Every game trajectory/result must match. Timing includes scalar rules, UCI search,
game reset and per-game startup/shutdown in the reference; excludes one-time persistent-session
startup, preflight, manifest checks and disk commits. It measures steady-state orchestration,
not search NPS or playing strength.

2026-09-10, Apple M3 Pro, Release/native/LTO Islay, ONNX Runtime 1.29.0 sequential CPU
inference, four simulations per move: median six-game workload was **972.06 ms** with
persistent sessions versus **1,583.30 ms** with per-game reload. Median paired speedup was
**1.634x** over five alternating-order rounds. This intentionally low-budget workload exposes
orchestration overhead; the benefit will shrink when searches dominate runtime.

## Completed bootstrap trial

On the same machine/build, a preregistered fixed trial used seed **20260911**, **200 pairs /
400 games**, 12-ply openings, **16 simulations per move**, 8 MiB trees, threshold 0.55 and
alpha 0.05. All 200 openings were D4-distinct. Candidate was the P5 checkpoint after 80
updates; opponent was its random initialization. Candidate W/D/L was **304/14/82**, score
**0.7775**, pair-score bins **[9,2,62,12,115]**. Paired bootstrap score CI95 was
**[0.73625,0.8175]**, equivalent relative Elo **+217.35 [178.33,260.49]**. The approximate
paired-normal LOS rounded to 1.0; this is numerical saturation, not certainty.

The one-sided Hoeffding lower score was **0.69096**, above the required 0.55, so the offline
selection marked the candidate eligible. No failed games were counted or skipped. Cumulative
search nodes were 155,472 candidate / 155,872 champion (16 per decision, different game lengths),
with measured move round-trip totals 146.39 / 148.29 seconds. These wall times are diagnostic;
the separately isolated throughput benchmark above is the performance comparison.

```text
engine:    7c163c14ddb4436bba0b08ce4a8bea9f5ef2c53dc973ef536f23296e1930ead6
candidate: c6a553f7c739124369e9e528b58f72132219fd2419b803ebdf8d5385a57d89e5
champion:  cdd2d8cfc9c58316494c3332c6afd13ef67ff5a906789c7448fba8ad53d8c486
manifest:  e56974d73afc2ae1a1b7ffd165aa6a3436920917b0b0f2eaa3dc8e78d57c8e77
```

Reproduce the command above with `--pairs 200 --seed 20260911 --mode nodes --budget 16`
and these model snapshots. Artifacts remain local under `_build/p6-bootstrap-nodes`.
This closes one bootstrap self-play → train/resume → export → arena cycle. It is evidence
against **this random opponent at this budget/opening distribution**, not an absolute Elo,
high-budget strength claim, or proof that further iterations necessarily improve. No weights
were deployed automatically. An equal-requested-movetime smoke test passed separately;
no adequately sized time-mode strength trial is claimed.
