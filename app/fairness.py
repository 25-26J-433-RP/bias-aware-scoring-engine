from __future__ import annotations

from typing import Iterable

import numpy as np

from .schemas import FairnessReport

# Shared pass/fail threshold used by API evaluation and batch analysis.
FAIRNESS_PASS_CUTOFF = 45.0


def _to_np(x: Iterable) -> np.ndarray:
    arr = np.asarray(list(x))
    if arr.ndim != 1:
        raise ValueError("Input must be 1-D")
    return arr


def binarize(scores: Iterable[float], cutoff: float = FAIRNESS_PASS_CUTOFF) -> np.ndarray:
    s = _to_np(scores).astype(float)
    return (s >= cutoff).astype(int)


def _rate(mask: np.ndarray, positives: np.ndarray) -> float:
    return float(positives[mask].mean()) if mask.sum() > 0 else 0.0


def spd(y_hat_bin: Iterable[int], groups: Iterable[bool]) -> float:
    yb = _to_np(y_hat_bin).astype(int)
    g = _to_np(groups).astype(bool)
    # AIF360 direction: unprivileged - privileged.
    # Here, dyslexic students are the unprivileged group (g=True).
    return _rate(g, yb) - _rate(~g, yb)


def dir_ratio(y_hat_bin: Iterable[int], groups: Iterable[bool]) -> float:
    yb = _to_np(y_hat_bin).astype(int)
    g = _to_np(groups).astype(bool)
    # AIF360 direction: unprivileged / privileged.
    # Here, dyslexic students are the unprivileged group (g=True).
    rate_unpriv = _rate(g, yb)
    rate_priv = _rate(~g, yb)

    if rate_priv > 0:
        return float(rate_unpriv / rate_priv)
    # If privileged pass rate is zero:
    # - both zero => neutral
    # - unprivileged > 0 => favorable to unprivileged (infinite DIR)
    return 1.0 if rate_unpriv == 0 else float("inf")


def eod(y_hat_bin: Iterable[int], y_true: Iterable[int], groups: Iterable[bool]) -> float:
    yb = _to_np(y_hat_bin).astype(int)
    yt = _to_np(y_true).astype(int)
    g = _to_np(groups).astype(bool)
    # AIF360-aligned direction: unprivileged - privileged.
    return _rate((g) & (yt == 1), yb) - _rate((~g) & (yt == 1), yb)


def safe_float(x: float) -> float:
    if x is None:
        return 0.0
    if x != x:  # NaN
        return 0.0
    if x in [float("inf"), float("-inf")]:
        return 1.0
    return float(x)


def demo_fairness_report(n: int = 50, seed: int = 0) -> FairnessReport:
    rng = np.random.default_rng(seed)
    groups = rng.random(n) < 0.5
    ability = rng.standard_normal(n)
    scores = 70 + 10 * ability - 8 * groups.astype(float)
    scores = np.clip(scores, 0, 100)
    y_true = (ability >= 0).astype(int)
    y_hat_bin = binarize(scores, FAIRNESS_PASS_CUTOFF)

    return FairnessReport(
        spd=safe_float(spd(y_hat_bin, groups)),
        dir=safe_float(dir_ratio(y_hat_bin, groups)),
        eod=safe_float(eod(y_hat_bin, y_true, groups)),
        mitigation_used=None,
    )
