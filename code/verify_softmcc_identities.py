"""Deterministic numerical checks for SoftMCC Propositions 1--3 and 6.

Finite-sample algebraic identities (Propositions 1--3) and the calibrated
population identity (Proposition 6) are checked separately.  The script also
constructs a miscalibrated population for which SoftMCC equals the Brier skill
score, demonstrating that equality is not a converse calibration diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from sklearn.metrics import matthews_corrcoef


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "04_manuscript").is_dir() and (candidate / "03_experiments").is_dir():
            return candidate
    raise RuntimeError("Project root not found")


HERE = Path(__file__).resolve().parent
ROOT = find_project_root(HERE)
SOURCE_SCRIPTS = ROOT / "03_experiments" / "scripts"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SOURCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_SCRIPTS))
import harden_dupsafe_full as H  # noqa: E402


def closed_form(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    prevalence = float(np.mean(y))
    p_mean = float(np.mean(p))
    denominator = np.sqrt(p_mean * (1 - p_mean) * prevalence * (1 - prevalence))
    if denominator == 0:
        return 0.0
    return float(np.mean((p - p_mean) * (y - prevalence)) / denominator)


def rho_kappa(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if np.std(y) == 0 or np.std(p) == 0:
        return 0.0
    rho = float(np.corrcoef(p, y)[0, 1])
    p_mean = float(np.mean(p))
    kappa = float(np.sqrt(np.var(p) / (p_mean * (1 - p_mean))))
    return rho * kappa


def calibrated_population_identity(weights: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    prevalence = float(np.sum(weights * p))
    variance = float(np.sum(weights * (p - prevalence) ** 2))
    softmcc = variance / (prevalence * (1 - prevalence))
    brier = float(np.sum(weights * p * (1 - p)))
    brier_skill = 1.0 - brier / (prevalence * (1 - prevalence))
    return softmcc, brier_skill


def general_population_values(p: np.ndarray, q: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    p_mean = float(np.sum(weights * p))
    prevalence = float(np.sum(weights * q))
    covariance = float(np.sum(weights * p * q) - p_mean * prevalence)
    softmcc = covariance / np.sqrt(p_mean * (1 - p_mean) * prevalence * (1 - prevalence))
    brier = float(np.sum(weights * (p**2 - 2 * p * q + q)))
    brier_skill = 1.0 - brier / (prevalence * (1 - prevalence))
    return softmcc, brier_skill


def run_checks(seed: int = 20260718, trials: int = 500) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    proposition1_errors: list[float] = []
    proposition2_errors: list[float] = []
    proposition3_errors: list[float] = []
    proposition3_bound_violations = 0
    for _ in range(trials):
        n = int(rng.integers(20, 500))
        prevalence = float(rng.uniform(0.03, 0.97))
        y = rng.binomial(1, prevalence, size=n).astype(int)
        if len(np.unique(y)) < 2:
            continue
        p = np.clip(rng.beta(0.7, 0.7, size=n), 1e-9, 1 - 1e-9)
        direct = H.soft_mcc(y, p)
        proposition1_errors.append(abs(direct - closed_form(y, p)))
        factorized = rho_kappa(y, p)
        proposition3_errors.append(abs(direct - factorized))
        rho = float(np.corrcoef(p, y)[0, 1])
        if abs(direct) > abs(rho) + 1e-12 or abs(direct) > 1 + 1e-12:
            proposition3_bound_violations += 1
        hard = rng.binomial(1, rng.uniform(0.05, 0.95), size=n).astype(int)
        if len(np.unique(hard)) == 2:
            proposition2_errors.append(abs(H.soft_mcc(y, hard) - matthews_corrcoef(y, hard)))

    calibrated_cases = [
        (np.asarray([0.2, 0.3, 0.5]), np.asarray([0.05, 0.4, 0.9])),
        (np.asarray([0.1, 0.2, 0.3, 0.4]), np.asarray([0.01, 0.2, 0.7, 0.95])),
        (np.asarray([0.5, 0.5]), np.asarray([0.2, 0.8])),
    ]
    proposition6_errors: list[float] = []
    proposition6_values: list[dict[str, float]] = []
    for weights, p in calibrated_cases:
        weights = weights / weights.sum()
        softmcc, skill = calibrated_population_identity(weights, p)
        proposition6_errors.append(abs(softmcc - skill))
        proposition6_values.append({"softmcc": softmcc, "brier_skill_score": skill})

    p_support = np.asarray([0.2, 0.8])
    weights = np.asarray([0.5, 0.5])
    q_low = 0.01

    def equality_gap(q_high: float) -> float:
        softmcc, skill = general_population_values(
            p_support, np.asarray([q_low, q_high]), weights
        )
        return softmcc - skill

    q_high = float(brentq(equality_gap, 0.6, 0.9, xtol=1e-14))
    counter_softmcc, counter_skill = general_population_values(
        p_support, np.asarray([q_low, q_high]), weights
    )
    counterexample = {
        "p_support": p_support.tolist(),
        "conditional_event_probabilities": [q_low, q_high],
        "maximum_absolute_calibration_error": float(max(abs(q_low - 0.2), abs(q_high - 0.8))),
        "softmcc": counter_softmcc,
        "brier_skill_score": counter_skill,
        "absolute_identity_gap": abs(counter_softmcc - counter_skill),
        "interpretation": "Miscalibrated population with SoftMCC equal to BSS; equality is not a converse diagnostic.",
    }
    return {
        "seed": seed,
        "trials_requested": trials,
        "proposition_1_max_abs_error": float(max(proposition1_errors)),
        "proposition_2_max_abs_error": float(max(proposition2_errors)),
        "proposition_3_factorization_max_abs_error": float(max(proposition3_errors)),
        "proposition_3_bound_violations": proposition3_bound_violations,
        "proposition_6_population_max_abs_error": float(max(proposition6_errors)),
        "proposition_6_population_cases": proposition6_values,
        "nonconverse_counterexample": counterexample,
        "scope_note": "Propositions 1-3 are finite-sample algebraic checks. Proposition 6 is a calibrated population identity, not an exact realized-sample identity.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--trials", type=int, default=500)
    args = parser.parse_args()
    result = run_checks(args.seed, args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    thresholds_pass = (
        result["proposition_1_max_abs_error"] < 1e-12
        and result["proposition_2_max_abs_error"] < 1e-12
        and result["proposition_3_factorization_max_abs_error"] < 1e-12
        and result["proposition_3_bound_violations"] == 0
        and result["proposition_6_population_max_abs_error"] < 1e-12
        and result["nonconverse_counterexample"]["absolute_identity_gap"] < 1e-12
    )
    return 0 if thresholds_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
