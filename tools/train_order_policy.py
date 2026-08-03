#!/usr/bin/env python3
"""Fit the compact no-TT move-order policy from `orderdata` CSV on stdin."""

import csv
import math
import random
import sys


SQUARE_ORBIT = (
    0, 1, 2, 3, 3, 2, 1, 0,
    1, 4, 5, 6, 6, 5, 4, 1,
    2, 5, 7, 8, 8, 7, 5, 2,
    3, 6, 8, 9, 9, 8, 6, 3,
    3, 6, 8, 9, 9, 8, 6, 3,
    2, 5, 7, 8, 8, 7, 5, 2,
    1, 4, 5, 6, 6, 5, 4, 1,
    0, 1, 2, 3, 3, 2, 1, 0,
)
SQUARE_VALUE = (18, -4, 4, 2, -8, 1, 1, 2, 1, 1)
FEATURES = 120


def phase(stage):
    return min(stage // 4, 3)


def relation(prev, square):
    if prev >= 64:
        return 0
    dx = abs((prev & 7) - (square & 7))
    dy = abs((prev >> 3) - (square >> 3))
    lo, hi = sorted((dx, dy))
    return hi * (hi + 1) // 2 + lo


def features(row):
    p = phase(row[1])
    sq = row[4]
    return (
        (p * 10 + SQUARE_ORBIT[sq], 1.0),
        (40 + p, float(row[5])),
        (44 + p, float(row[6])),
        (48 + relation(row[2], sq), 1.0),
        (84 + relation(row[3], sq), 1.0),
    )


def score(weights, feat):
    return sum(weights[i] * value for i, value in feat)


def accuracy(samples, weights, baseline=False):
    correct = 0
    reciprocal_rank = 0.0
    for moves in samples:
        ranked = []
        for row in moves:
            if baseline:
                s = 8 * SQUARE_VALUE[SQUARE_ORBIT[row[4]]] - 32 * row[6]
            else:
                s = score(weights, features(row))
            ranked.append((s, row[7]))
        ranked.sort(reverse=True)
        correct += ranked[0][1]
        rank = next(i + 1 for i, (_, label) in enumerate(ranked) if label)
        reciprocal_rank += 1.0 / rank
    n = max(len(samples), 1)
    return 100.0 * correct / n, reciprocal_rank / n


def main():
    grouped = {}
    for raw in csv.reader(line for line in sys.stdin if line and line[0].isdigit()):
        if len(raw) != 8:
            continue
        row = tuple(map(int, raw))
        grouped.setdefault(row[0], []).append(row)
    samples = [moves for _, moves in sorted(grouped.items()) if sum(row[7] for row in moves) == 1]
    train = [moves for i, moves in enumerate(samples) if i % 5]
    valid = [moves for i, moves in enumerate(samples) if i % 5 == 0]

    weights = [0.0] * FEATURES
    for p in range(4):
        for orbit, value in enumerate(SQUARE_VALUE):
            weights[p * 10 + orbit] = 8.0 * value
        weights[44 + p] = -32.0

    rng = random.Random(20260803)
    for epoch in range(10):
        rng.shuffle(train)
        updates = 0
        for moves in train:
            positive = next(row for row in moves if row[7])
            pf = features(positive)
            for negative in moves:
                if negative[7]:
                    continue
                nf = features(negative)
                margin = 64.0 - (score(weights, pf) - score(weights, nf))
                if margin <= 0.0:
                    continue
                diff = {}
                for i, value in pf:
                    diff[i] = diff.get(i, 0.0) + value
                for i, value in nf:
                    diff[i] = diff.get(i, 0.0) - value
                norm = sum(value * value for value in diff.values())
                step = min(0.25, margin / max(norm, 1.0))
                for i, value in diff.items():
                    weights[i] += step * value
                updates += 1
        top1, mrr = accuracy(valid, weights)
        print(f"epoch {epoch + 1}: updates={updates} valid_top1={top1:.2f}% mrr={mrr:.4f}", file=sys.stderr)

    base_top1, base_mrr = accuracy(valid, weights, baseline=True)
    top1, mrr = accuracy(valid, weights)
    scale = max(max(abs(value) for value in weights) / 30000.0, 1.0)
    quantized = [int(round(value / scale)) for value in weights]
    print(f"samples train={len(train)} valid={len(valid)}")
    print(f"baseline top1={base_top1:.2f}% mrr={base_mrr:.4f}")
    print(f"learned  top1={top1:.2f}% mrr={mrr:.4f} scale={scale:.6f}")
    print("weights=" + ",".join(map(str, quantized)))


if __name__ == "__main__":
    main()
