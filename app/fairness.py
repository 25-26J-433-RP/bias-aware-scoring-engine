from __future__ import annotations
import numpy as np
from typing import Iterable, Optional
from .schemas import FairnessReport


def _to_np(x: Iterable) -> np.ndarray:
    arr = np.asarray(list(x))
    if arr.ndim != 1:
        raise ValueError("Input must be 1-D")
    return arr


def binarize(scores: Iterable[float], cutoff: float = 75.0) -> np.ndarray:
    s = _to_np(scores).astype(float)
    return (s >= cutoff).astype(int)


def _rate(mask: np.ndarray, positives: np.ndarray) -> float:
    return float(positives[mask].mean()) if mask.sum() > 0 else 0.0


def spd(y_hat_bin: Iterable[int], groups: Iterable[bool]) -> float:
    yb = _to_np(y_hat_bin).astype(int)
    g = _to_np(groups).astype(bool)
    # Rate(Dyslexic) - Rate(Non-Dyslexic)
    return _rate(g, yb) - _rate(~g, yb)


def dir_ratio(y_hat_bin: Iterable[int], groups: Iterable[bool]) -> float:
    yb = _to_np(y_hat_bin).astype(int)
    g = _to_np(groups).astype(bool)
    rate_privileged = _rate(~g, yb)
    rate_unprivileged = _rate(g, yb)

    # Standard DIR: Rate(Unprivileged) / Rate(Privileged)
    return float(rate_unprivileged / rate_privileged) if rate_privileged > 0 else 1.0


def eod(y_hat_bin: Iterable[int], y_true: Iterable[int], groups: Iterable[bool]) -> float:
    yb = _to_np(y_hat_bin).astype(int)
    yt = _to_np(y_true).astype(int)
    g = _to_np(groups).astype(bool)
    return _rate((~g) & (yt == 1), yb) - _rate((g) & (yt == 1), yb)


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
    y_hat_bin = binarize(scores, 75)

    return FairnessReport(
        spd=safe_float(spd(y_hat_bin, groups)),
        dir=safe_float(dir_ratio(y_hat_bin, groups)),
        eod=safe_float(eod(y_hat_bin, y_true, groups)),
        mitigation_used=None,
    )
