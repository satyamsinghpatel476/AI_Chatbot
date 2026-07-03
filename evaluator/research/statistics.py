import math
import statistics

import numpy as np
from scipy.stats import binomtest, wilcoxon


def percentile(values, percentile_value):
    if not values:
        return None
    return float(np.percentile(values, percentile_value))


def bootstrap_ci(values, *, seed=0, samples=4000, confidence=0.95):
    clean = np.asarray(
        [float(value) for value in values if value is not None],
        dtype=float,
    )
    if clean.size == 0:
        return None, None
    if clean.size == 1:
        value = float(clean[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, clean.size, size=(samples, clean.size))
    means = clean[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1 - alpha)),
    )


def describe(values, *, seed=0):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    low, high = bootstrap_ci(clean, seed=seed)
    return {
        "n": len(clean),
        "mean": statistics.mean(clean),
        "median": statistics.median(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "ci95_low": low,
        "ci95_high": high,
    }


def describe_latency(values, *, seed=0):
    result = describe(values, seed=seed)
    clean = [float(value) for value in values if value is not None]
    result["p95"] = percentile(clean, 95)
    return result


def mcnemar_exact(first, second):
    pairs = [
        (int(a), int(b))
        for a, b in zip(first, second)
        if a is not None and b is not None
    ]
    first_only = sum(a == 1 and b == 0 for a, b in pairs)
    second_only = sum(a == 0 and b == 1 for a, b in pairs)
    discordant = first_only + second_only
    p_value = (
        binomtest(
            min(first_only, second_only),
            discordant,
            0.5,
            alternative="two-sided",
        ).pvalue
        if discordant
        else 1.0
    )
    return {
        "n": len(pairs),
        "first_only": first_only,
        "second_only": second_only,
        "discordant": discordant,
        "p_value": float(p_value),
    }


def paired_continuous(first, second):
    pairs = [
        (float(a), float(b))
        for a, b in zip(first, second)
        if a is not None and b is not None
    ]
    if not pairs:
        return {
            "n": 0,
            "mean_difference": None,
            "median_difference": None,
            "wilcoxon_p_value": None,
        }
    differences = [b - a for a, b in pairs]
    if all(abs(value) < 1e-12 for value in differences):
        p_value = 1.0
    else:
        try:
            p_value = float(wilcoxon(
                [a for a, _ in pairs],
                [b for _, b in pairs],
            ).pvalue)
        except ValueError:
            p_value = None
    return {
        "n": len(pairs),
        "mean_difference": statistics.mean(differences),
        "median_difference": statistics.median(differences),
        "wilcoxon_p_value": p_value,
    }
