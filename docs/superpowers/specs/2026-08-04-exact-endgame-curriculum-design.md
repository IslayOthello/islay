# Exact-Endgame NNUE Curriculum Design

## Goal

Train a V23 NNUE candidate from V22 using 50,000 deterministic source trajectories and at most four generation workers. Improve target quality rather than network size: deeper search should supervise opening and midgame positions, exact searches should supervise late positions, and positions where V21 or V22 disagree with the teacher should receive extra training weight. The shipped NNUE format and inference path must remain unchanged.

## Chosen Approach

Use one weighted curriculum and one training pass. This avoids the catastrophic-forgetting risk of sequential midgame/endgame fine-tuning and avoids the runtime cost of an auxiliary WLD head. V22 remains the active search teacher and warm start. A separately loaded V21 network is used only during data generation to measure disagreement.

The existing `ntrain` behavior remains available as a control. Its command accepts two trailing fields, `curriculum` and `reference`; curriculum mode requires a compatible V21 reference file:

```text
ntrain 50000 10 12 0.001 0.00001 weights/v23.nnue 20260804 4 1 1 weights/v21.nnue
```

## Data Generation

Generation remains deterministic for a fixed seed and is hard-capped at four single-threaded `Searcher` instances. The 50,000 trajectories are divided before generation:

- 40,000 main trajectories stop at 21 empties. Every post-opening position receives a depth-12 margin label.
- 5,000 exact-margin trajectories choose one target uniformly from 10–14 empties, solve that root for the exact disc margin, then stop.
- 5,000 exact-WLD trajectories choose one target uniformly from 15–20 empties, run a narrow exact win/draw/loss solve, then stop.

Each main trajectory retains at most one hard position. V21 and V22 static scores are compared with the depth-12 teacher score. A WLD-sign mismatch takes priority; otherwise the position with the largest absolute residual is retained and re-searched at depth 14. This bounds the expensive depth-14 work to 40,000 searches while allowing the trainer to oversample those records.

Each stored target contains the margin when known, WLD class, target kind, and sampling weight. Invalid, aborted, or incomplete exact searches are discarded and counted; they are never silently replaced by a shallow target.

## Sampling and Loss

Training and validation are split by trajectory before sampling. Every epoch draws the following target mix:

- 65% ordinary depth-12 margin records;
- 15% hard depth-14 records;
- 10% exact-margin records;
- 10% exact-WLD records.

The existing scalar NNUE output remains measured in discs. Margin supervision uses Huber loss with a four-disc transition. WLD supervision uses a training-only ternary sign loss: logistic loss for wins and losses, and a bounded quadratic pull toward zero for draws. Records with a margin use `0.75 * margin_loss + 0.25 * wld_loss`; exact-WLD-only records use WLD loss alone. No WLD weights or operations enter the network file or inference hot path.

Early stopping selects the lowest held-out composite loss, with epoch zero remaining eligible. Logs report sample count, RMSE, WLD accuracy, and composite loss for each target kind so a better aggregate cannot hide an endgame regression.

## Implementation Boundaries

- Replace the per-ply teacher integer with a compact target record in the NNUE trainer only.
- Load V21 into an independent `NnueNet`; do not switch the process-global active evaluator while workers are searching.
- Add an explicit fixed-depth WLD solve request for curriculum generation. Normal UCI searches retain their current full-margin behavior.
- Keep legacy `ntrain`, NNUE serialization, quantization, and evaluation code byte-compatible.
- Do not modify unrelated search heuristics or `AGENTS.md`.

## Verification and Promotion

Add deterministic trainer checks for bucket selection, hard-position priority, Huber/WLD gradients, and validation-by-trajectory splitting. Run a small curriculum smoke train twice with the same seed, all built-in self-tests, and start-position perft(11) = `212258216`.

Train the full 50,000-game V23 candidate with four workers. Compare it with V22 using the balanced depth-8 book: first a short fixed-node screen, then a paired, color-reversed 10,000-game equal-time SPRT with `H0=-3`, `H1=+3`, and identical engine options. Report Elo, 95% confidence interval, LOS, and LLR. Commit the candidate network and implementation only when the final Elo estimate is positive; otherwise remove the experiment cleanly.
