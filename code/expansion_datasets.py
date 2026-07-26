"""Prespecified dataset-expansion registry for the SoftMCC selection study.

Methodology change: MCL-20260724-dataset-expansion-01
(`MD/_state/METHODOLOGY_CHANGE_LEDGER.md`). The inclusion rule below was locked
BEFORE any expansion result was computed. The qualifying list is derived
mechanically from the rule applied to the imbalanced-learn benchmark
collection (Zenodo-hosted, `imblearn.datasets.fetch_datasets`); it is not a
hand-picked list. Every candidate dataset in the collection receives an
explicit include/exclude verdict with the failing clause recorded, so the
selection is auditable end to end.

This module does NOT modify `harden_dupsafe_full.py` or
`harden_dupsafe_tieaware.py`; it only adds datasets through the same
`DatasetSpec` interface.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

import harden_dupsafe_full as H

# ---------------------------------------------------------------------------
# Locked inclusion rule (see METHODOLOGY_CHANGE_LEDGER MCL-20260724-...-01)
# ---------------------------------------------------------------------------
RULE = {
    "min_rows": 1500,
    "min_positives": 40,
    "max_prevalence": 0.10,
    "origin_excluded": "raw instances originate from images, audio, or text",
}

# Collection members whose raw instances are image/audio/text renderings.
# The manuscript scope is imbalanced *tabular* binary tasks, so these are
# excluded as a class, independent of their numeric statistics.
ORIGIN_EXCLUDED = {
    "optical_digits": "handwritten digit images",
    "pen_digits": "pen-trajectory digit renderings",
    "letter_img": "letter image renderings",
    "libras_move": "video-derived hand movement curves",
    "isolet": "spoken letter audio features",
    "scene": "natural scene images (multilabel-derived)",
    "satimage": "satellite image pixel neighborhoods",
    "webpage": "web page text term vectors",
}

EXPANSION_DIR = H.DATA / "expansion"
CACHE_HOME = H.DATA / "expansion_zenodo_cache"
MANIFEST_PATH = EXPANSION_DIR / "expansion_rule_manifest.json"


def _binary_target(raw_target: np.ndarray) -> tuple[np.ndarray, bool]:
    """Map collection targets to {0,1} with the minority class as positive."""
    values = set(np.unique(raw_target).tolist())
    if values == {-1, 1}:
        y = (raw_target == 1).astype(np.int8)
    elif values == {0, 1}:
        y = raw_target.astype(np.int8)
    else:
        raise ValueError(f"Unexpected target values: {sorted(values)}")
    flipped = False
    if float(np.mean(y)) > 0.5:
        y = (1 - y).astype(np.int8)
        flipped = True
    return y, flipped


def prepare_expansion_datasets(force: bool = False) -> dict:
    """Download the collection, apply the locked rule, cache qualifiers.

    Returns the manifest dict. Safe to re-run: cached npz files are reused
    unless ``force`` is set. Requires network access on first run.
    """
    from imblearn import __version__ as imblearn_version
    from imblearn.datasets import fetch_datasets

    EXPANSION_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_HOME.mkdir(parents=True, exist_ok=True)

    collection = fetch_datasets(data_home=str(CACHE_HOME))
    entries = []
    for name in sorted(collection):
        bunch = collection[name]
        X = np.asarray(bunch.data, dtype=np.float64)
        y, flipped = _binary_target(np.asarray(bunch.target))
        rows = int(X.shape[0])
        positives = int(np.sum(y))
        prevalence = float(positives / rows)
        finite = bool(np.all(np.isfinite(X)))

        verdict = "include"
        reasons = []
        if name in ORIGIN_EXCLUDED:
            verdict = "exclude"
            reasons.append(f"origin_excluded: {ORIGIN_EXCLUDED[name]}")
        if rows < RULE["min_rows"]:
            verdict = "exclude"
            reasons.append(f"rows {rows} < {RULE['min_rows']}")
        if positives < RULE["min_positives"]:
            verdict = "exclude"
            reasons.append(f"positives {positives} < {RULE['min_positives']}")
        if prevalence > RULE["max_prevalence"]:
            verdict = "exclude"
            reasons.append(
                f"prevalence {prevalence:.4f} > {RULE['max_prevalence']}"
            )
        if not finite:
            verdict = "exclude"
            reasons.append("non-finite feature values (protocol has no imputation)")

        entry = {
            "name": name,
            "rows": rows,
            "features": int(X.shape[1]),
            "positives": positives,
            "prevalence": prevalence,
            "minority_flip_applied": flipped,
            "all_finite": finite,
            "verdict": verdict,
            "exclusion_reasons": reasons,
        }
        if verdict == "include":
            cache_path = EXPANSION_DIR / f"{name}.npz"
            if force or not cache_path.exists():
                # tmp name must end in .npz or np.savez appends the extension
                tmp = EXPANSION_DIR / f"{name}.tmp.npz"
                np.savez_compressed(tmp, X=X, y=y.astype(np.int64))
                os.replace(tmp, cache_path)
            entry["cache_path"] = str(cache_path.relative_to(H.ROOT))
            entry["cache_sha256"] = H.sha256_file(cache_path)
            entry["X_sha256"] = H.sha256_array(X)
            entry["y_sha256"] = H.sha256_array(y.astype(np.int64))
        entries.append(entry)

    manifest = {
        "methodology_change_id": "MCL-20260724-dataset-expansion-01",
        "rule": RULE,
        "origin_excluded_members": ORIGIN_EXCLUDED,
        "source": "imblearn.datasets.fetch_datasets (Zenodo benchmark collection)",
        "imbalanced_learn_version": imblearn_version,
        "prepared_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "collection_size": len(entries),
        "included_count": sum(1 for e in entries if e["verdict"] == "include"),
        "datasets": entries,
    }
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, MANIFEST_PATH)
    return manifest


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise RuntimeError(
            "expansion_rule_manifest.json not found; run "
            "prepare_expansion_datasets() first (requires network)."
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def expansion_specs() -> list[H.DatasetSpec]:
    """Qualifying expansion datasets as DatasetSpec, alphabetical by name."""
    manifest = load_manifest()
    specs: list[H.DatasetSpec] = []
    for entry in manifest["datasets"]:
        if entry["verdict"] != "include":
            continue
        name = entry["name"]
        cache_path = H.ROOT / entry["cache_path"]
        label = f"{name}({entry['prevalence'] * 100:.1f}%)"
        specs.append(
            H.DatasetSpec(
                key=name,
                label=label,
                source="imblearn.fetch_datasets",
                loader=(lambda p=cache_path: H.load_npz(p)),
                cache_path=cache_path,
            )
        )
    return specs


def combined_specs(mode: str = "all") -> list[H.DatasetSpec]:
    """Canonical six specs (unchanged order) followed by expansion specs."""
    canonical = H.dataset_specs()
    if mode == "canonical_only":
        return canonical
    if mode == "new_only":
        return expansion_specs()
    if mode == "all":
        return canonical + expansion_specs()
    raise ValueError(f"Unknown mode: {mode}")


def get_spec(key: str) -> H.DatasetSpec:
    """Resolve one spec by key (used inside unit worker processes).

    Canonical specs are searched first so smoke runs on canonical datasets
    work before the expansion manifest exists (no network required).
    """
    for spec in H.dataset_specs():
        if spec.key == key:
            return spec
    for spec in expansion_specs():
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown dataset key: {key}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = prepare_expansion_datasets(force=args.force)
    included = [e["name"] for e in result["datasets"] if e["verdict"] == "include"]
    print(f"Included ({len(included)}): {', '.join(included)}")
    for entry in result["datasets"]:
        if entry["verdict"] == "exclude":
            print(f"Excluded {entry['name']}: {'; '.join(entry['exclusion_reasons'])}")
