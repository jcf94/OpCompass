"""Raw prediction accuracy and transparent scale calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Iterable


@dataclass(frozen=True)
class AccuracyReport:
    count: int
    median_ape: float
    p90_ape: float
    log_rmse: float
    bias: float
    top_k_recall: float | None
    groups: dict[str, dict[str, float]]


@dataclass(frozen=True)
class CalibrationOverlay:
    overlay_id: str
    scale: float
    training_record_ids: tuple[str, ...]
    model_version: str

    def apply(self, predicted_us: float) -> float:
        return predicted_us * self.scale


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def evaluate_accuracy(rows: Iterable[dict], top_k: int = 1) -> AccuracyReport:
    rows = list(rows)
    if not rows:
        raise ValueError("accuracy report requires at least one row")
    measured = [float(row["measured_us"]) for row in rows]
    predicted = [float(row["predicted_us"]) for row in rows]
    if any(value <= 0 for value in measured + predicted):
        raise ValueError("accuracy runtimes must be positive")
    ape = [abs(p - m) / m for p, m in zip(predicted, measured)]
    log_errors = [math.log(p / m) for p, m in zip(predicted, measured)]
    groups = {}
    for key in ("hardware", "dtype", "shape_regime", "bottleneck"):
        for value in sorted({row.get(key, "unknown") for row in rows}):
            indices = [i for i, row in enumerate(rows) if row.get(key, "unknown") == value]
            values = [ape[i] for i in indices]
            groups[f"{key}:{value}"] = {"count": len(indices), "median_ape": median(values)}
    ranking = [row for row in rows if "measured_rank" in row and "predicted_rank" in row]
    recall = (sum(row["predicted_rank"] <= top_k and row["measured_rank"] <= top_k for row in ranking) / len(ranking)) if ranking else None
    return AccuracyReport(len(rows), median(ape), _percentile(ape, .9), math.sqrt(sum(x*x for x in log_errors) / len(rows)), sum(p-m for p, m in zip(predicted, measured)) / sum(measured), recall, groups)


def fit_scale_overlay(rows: Iterable[dict], overlay_id: str, model_version: str) -> CalibrationOverlay:
    rows = list(rows)
    scale = median(float(row["measured_us"]) / float(row["predicted_us"]) for row in rows)
    return CalibrationOverlay(overlay_id, scale, tuple(str(row["record_id"]) for row in rows), model_version)
