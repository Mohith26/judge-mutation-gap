from __future__ import annotations

import random


def auroc(neg: list[float], pos: list[float]) -> float:
    """P(score_pos > score_neg) with ties counted half (Mann-Whitney)."""
    if not neg or not pos:
        return float("nan")
    allv = sorted([(v, 0) for v in neg] + [(v, 1) for v in pos])
    ranks, i = [0.0] * len(allv), 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    rp = sum(r for r, (_, y) in zip(ranks, allv) if y == 1)
    n1, n0 = len(pos), len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n0)


def threshold_at_fpr(clean_scores: list[float], fpr: float) -> float:
    """Smallest threshold t such that the share of clean scores > t is <= fpr."""
    s = sorted(clean_scores)
    k = int(len(s) * (1 - fpr))
    k = min(max(k, 0), len(s) - 1)
    t = s[k]
    # ensure strictness: flag iff score > t
    return t


def rate(scores: list[float], t: float) -> float:
    return sum(v > t for v in scores) / len(scores) if scores else float("nan")


def bootstrap_ci(values: list[int], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    if not values:
        return (float("nan"), float("nan"))
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]
