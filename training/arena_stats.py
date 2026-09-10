"""Fixed-sample paired statistics; diagnostic bootstrap, conservative selection gate."""
import math
from statistics import NormalDist

import numpy as np


def elo(score):
    if score == 0: return "-inf"
    if score == 1: return "+inf"
    return 400 * math.log10(score / (1 - score))


def summarize(pairs, target, seed, unique_openings, threshold=0.55, alpha=0.05):
    n = len(pairs)
    if not n or n > target: raise ValueError("invalid paired sample size")
    games = [g["score"] for pair in pairs for g in pair["games"]]
    if len(games) != n * 2 or any(s not in (0, 0.5, 1) for s in games):
        raise ValueError("invalid paired scores")
    scores = np.array(games).reshape(n, 2).mean(axis=1)
    counts = np.bincount((scores * 4).astype(int), minlength=5)
    mean = float(scores.mean())
    # Multinomial resampling of five pair-score bins is equivalent to resampling pairs.
    draws = np.random.default_rng(seed).multinomial(n, counts / n, size=20000)
    means = draws @ np.arange(5, dtype=np.float64) / (4 * n)
    ci = np.quantile(means, [0.025, 0.975]).tolist()
    se = float(scores.std(ddof=1) / math.sqrt(n)) if n > 1 else 0
    los = NormalDist().cdf((mean - 0.5) / se) if se else (0.5 if mean == 0.5 else None)
    lower = max(0, mean - math.sqrt(math.log(1 / alpha) / (2 * n)))
    radius = math.sqrt(math.log(40) / (2 * n))
    complete = n == target
    eligible = complete and n >= 200 and unique_openings >= 0.9 * n and lower > threshold
    return {"pairs": n, "games": n * 2, "target_pairs": target, "complete": complete,
            "wdl": {"wins": games.count(1), "draws": games.count(0.5), "losses": games.count(0)},
            "pentanomial": counts.tolist(), "score": mean, "elo": elo(mean),
            "paired_bootstrap_score_ci95": ci, "paired_bootstrap_elo_ci95": [elo(s) for s in ci],
            "bootstrap_degenerate": bool(scores.min() == scores.max()),
            "paired_hoeffding_score_ci95": [max(0, mean - radius), min(1, mean + radius)],
            "paired_normal_los": los, "los_degenerate": se == 0, "unique_openings_d4": unique_openings,
            "hoeffding_lower_score": lower, "gate": {"alpha": alpha, "threshold": threshold, "min_pairs": 200},
            "decision": "candidate_eligible" if eligible else "retain_champion"}
