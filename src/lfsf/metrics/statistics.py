"""Statistical tests and evaluation harness for MFIF benchmarks.

Provides paired bootstrap confidence intervals, Wilcoxon signed-rank tests,
Holm-Bonferroni correction, win/tie/loss counts, and a unified evaluation runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 3407,
) -> Dict[str, float]:
    """Paired bootstrap confidence interval for the mean difference.

    Args:
        values_a: Per-scene metrics for method A.
        values_b: Per-scene metrics for method B.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (0.05 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: delta_mean, ci_lower, ci_upper, p_value (two-sided).
    """
    rng = np.random.RandomState(seed)
    n = len(values_a)
    deltas = values_a - values_b
    obs_delta = np.mean(deltas)

    boot_deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_deltas[i] = np.mean(deltas[idx])

    ci_lower = np.percentile(boot_deltas, 100 * alpha / 2)
    ci_upper = np.percentile(boot_deltas, 100 * (1 - alpha / 2))

    # Two-sided p-value from bootstrap null distribution
    centered = boot_deltas - np.mean(boot_deltas)
    p_value = np.mean(np.abs(centered) >= np.abs(obs_delta))

    return {
        "delta_mean": float(obs_delta),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
        "n_scenes": n,
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
    }


def wilcoxon_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> Dict[str, float]:
    """Two-sided Wilcoxon signed-rank test.

    Args:
        values_a: Per-scene metrics for method A.
        values_b: Per-scene metrics for method B.

    Returns:
        Dict with keys: statistic, p_value.
    """
    result = stats.wilcoxon(values_a, values_b, alternative="two-sided")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def holm_correction(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni correction for multiple comparisons.

    Args:
        p_values: Raw p-values from multiple tests.

    Returns:
        Corrected p-values (same order as input).
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n
    for rank, (idx, p) in enumerate(indexed):
        corrected[idx] = min(1.0, p * (n - rank))
    # Ensure monotonicity
    for i in range(1, n):
        if corrected[indexed[i][0]] < corrected[indexed[i - 1][0]]:
            corrected[indexed[i][0]] = corrected[indexed[i - 1][0]]
    return corrected


def compute_win_tie_loss(
    values_a: np.ndarray,
    values_b: np.ndarray,
    tie_threshold: float = 1e-12,
) -> Dict[str, int]:
    """Count wins, ties, and losses for method A vs. method B.

    Args:
        values_a: Per-scene metrics for method A.
        values_b: Per-scene metrics for method B.
        tie_threshold: Absolute difference below which results are considered tied.

    Returns:
        Dict with keys: wins, ties, losses, total.
    """
    delta = values_a - values_b
    wins = int(np.sum(delta > tie_threshold))
    losses = int(np.sum(delta < -tie_threshold))
    ties = len(delta) - wins - losses
    return {"wins": wins, "ties": ties, "losses": losses, "total": len(delta)}


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def evaluate_all_metrics(
    pred_dir: str,
    gt_dir: str,
    scene_list: Optional[List[str]] = None,
    profile: str = "paper_full",
) -> Dict[str, Dict[str, float]]:
    """Evaluate all metrics for predictions against ground truth.

    Args:
        pred_dir: Directory containing prediction images.
        gt_dir: Directory containing ground truth (AiF) images.
        scene_list: Optional list of scene names to evaluate.
        profile: Metric profile to use ("paper_full", "core", "extended").

    Returns:
        Dict mapping scene name -> dict of metric values.
    """
    from lfsf.metrics.full_reference import evaluate_main_metrics

    pred_path = Path(pred_dir)
    gt_path = Path(gt_dir)
    if scene_list is None:
        scene_list = sorted([p.stem for p in pred_path.glob("*.png") if p.is_file()])

    results = {}
    for scene in scene_list:
        pred_file = pred_path / f"{scene}.png"
        gt_file = gt_path / f"{scene}.png"
        if not pred_file.exists() or not gt_file.exists():
            continue
        metrics = evaluate_main_metrics(str(pred_file), str(gt_file), profile=profile)
        results[scene] = metrics
    return results


def aggregate_dataset_metrics(
    per_scene: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute dataset-level macro averages from per-scene metrics.

    Args:
        per_scene: Dict mapping scene name -> dict of metric values.
        metrics: List of metric names to aggregate (default: all found).

    Returns:
        Dict mapping metric name -> macro-averaged value.
    """
    if metrics is None:
        metrics = set()
        for v in per_scene.values():
            metrics.update(v.keys())
        metrics = sorted(metrics)

    result = {}
    for metric in metrics:
        values = [v[metric] for v in per_scene.values() if metric in v]
        if values:
            result[metric] = float(np.mean(values))
    return result
