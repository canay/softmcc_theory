"""Durability-compliant unit-level runner for the dataset-expansion study.

Methodology change: MCL-20260724-dataset-expansion-01.
Durability contract: AkademikAkisMerkezi/Akis1_AnaPipeline/EXPERIMENT_DURABILITY_AND_RECOVERY.md.

Scientific computations per (dataset, repeat) unit are a faithful transcription
of the per-block logic in ``harden_dupsafe_tieaware.py`` ``run_raw`` (2026-07-18
canonical run). The canonical scripts are NOT modified; this driver adds:

* atomic unit = (dataset, repeat); each unit writes its own shard atomically;
* validated resume (schema + row counts + npz hash), new attempt IDs, no
  overwrite of completed evidence;
* machine-readable heartbeat (<=300 s cadence; default 60 s) and progress log;
* per-unit timeout via a child process, plus optional whole-run soft watchdog;
* Telegram lifecycle notifications via the ``mesaj`` command (best effort);
* terminal-status handling for normal end, failure, timeout, and cancellation.

Aggregation, statistics, figures, and the prespecified decision artifact are
produced by ``expansion_aggregate.py`` from the unit shards; this runner never
computes manuscript-facing summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

HEARTBEAT_INTERVAL_SECONDS = 60
SCHEMA_VERSION = 1
N_METRICS = 7
N_TEMPS = 4
N_CANDIDATES = 5


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# ---------------------------------------------------------------------------
# Unit worker (runs in a child process)
# ---------------------------------------------------------------------------

def unit_worker(run_root_str: str, spec_key: str, repeat: int, attempt: int) -> None:
    """Compute one (dataset, repeat) unit and write its shard atomically."""
    run_root = Path(run_root_str)
    units_dir = run_root / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    unit_id = f"{spec_key}__r{repeat:02d}"
    phase_path = units_dir / f"{unit_id}.phase.jsonl"

    def phase(name: str) -> None:
        append_jsonl(
            phase_path,
            {"ts": now_iso(), "attempt": attempt, "pid": os.getpid(), "phase": name},
        )

    import numpy as np
    from sklearn.metrics import log_loss, matthews_corrcoef

    import harden_dupsafe_full as H
    import harden_dupsafe_tieaware as T
    import expansion_datasets as X

    phase("resolve_spec")
    spec = X.get_spec(spec_key)

    phase("load_data")
    X_all, y_all = spec.loader()
    X_all = np.asarray(X_all)
    y_all = np.asarray(y_all).astype(int)

    phase("group_ids")
    groups, duplicate_rows = H.row_group_ids(X_all, y_all)
    unique_groups, first_indices = np.unique(groups, return_index=True)
    group_labels = {
        int(group): int(y_all[index])
        for group, index in zip(unique_groups, first_indices)
    }
    positives, negatives, prevalence = H.class_counts(y_all)
    dataset_row = {
        "dataset_key": spec.key,
        "dataset": spec.label,
        "source": spec.source,
        "cache_path": str(spec.cache_path) if spec.cache_path else "",
        "cache_sha256": H.sha256_file(spec.cache_path) if spec.cache_path else "",
        "rows": int(len(y_all)),
        "features": int(X_all.reshape((X_all.shape[0], -1)).shape[1]),
        "positive_rows": positives,
        "negative_rows": negatives,
        "prevalence": prevalence,
        "unique_exact_feature_label_groups": int(len(unique_groups)),
        "exact_duplicate_rows_with_label": duplicate_rows,
        "X_sha256": H.sha256_array(X_all),
        "y_sha256": H.sha256_array(y_all),
    }

    phase("split")
    seed = H.SEED + repeat
    trainval_groups, test_groups = H.split_groups(
        groups, group_labels, unique_groups, 0.25, seed
    )
    train_groups, validation_groups = H.split_groups(
        groups, group_labels, trainval_groups, 0.30, seed + 1
    )
    train_index = H.indices_for_groups(groups, train_groups)
    validation_index = H.indices_for_groups(groups, validation_groups)
    test_index = H.indices_for_groups(groups, test_groups)
    split_row = H.split_manifest_row(
        spec, repeat, seed, y_all, groups,
        train_index, validation_index, test_index, duplicate_rows,
    )

    X_train, y_train = X_all[train_index], y_all[train_index]
    X_validation, y_validation = X_all[validation_index], y_all[validation_index]
    X_test, y_test = X_all[test_index], y_all[test_index]

    validation_scores: dict[str, dict[str, float]] = {}
    validation_probs: dict[str, "np.ndarray"] = {}
    test_cache: dict[str, tuple] = {}
    for name, estimator in H.pool():
        phase(f"fit_{name}")
        estimator.fit(X_train, y_train)
        p_validation = estimator.predict_proba(X_validation)[:, 1]
        p_test = estimator.predict_proba(X_test)[:, 1]
        scores, mcc_threshold = H.selection_scores(y_validation, p_validation)
        validation_scores[name] = scores
        validation_probs[name] = p_validation
        test_cache[name] = (y_test, p_test, mcc_threshold)

    phase("score_rows")
    candidate_order = list(validation_scores)
    candidate_rows: list[dict] = []
    test_values: list[float] = []
    for name in candidate_order:
        y_test_candidate, p_test_candidate, threshold = test_cache[name]
        test_mcc = float(
            matthews_corrcoef(
                y_test_candidate, (p_test_candidate >= threshold).astype(int)
            )
        )
        test_values.append(test_mcc)
        probabilities = validation_probs[name]
        score_row: dict = {
            "dataset_key": spec.key,
            "dataset": spec.label,
            "repeat": repeat,
            "seed": seed,
            "candidate": name,
            "candidate_index": candidate_order.index(name),
            "validation_mccbest_threshold": threshold,
            "test_mcc_at_validation_mccbest_threshold": test_mcc,
            "validation_log_loss": float(
                log_loss(y_validation, np.clip(probabilities, 1e-12, 1 - 1e-12))
            ),
            "validation_ece_10": H.ece_score(y_validation, probabilities, bins=10),
            "validation_prob_mean": float(np.mean(probabilities)),
            "validation_prob_variance": float(np.var(probabilities)),
            "validation_pos_rate": float(np.mean(y_validation)),
        }
        for metric in H.METRICS:
            score_row[f"score_{metric}"] = float(validation_scores[name][metric])
        candidate_rows.append(score_row)

    ranking_rows: list[dict] = []
    utility_rows: list[dict] = []
    for metric in H.METRICS:
        score_map = {
            name: float(validation_scores[name][metric]) for name in candidate_order
        }
        values = [score_map[name] for name in candidate_order]
        ranks = T.midrank_vector(values)
        correction, tied_groups, largest_tie = T.tie_correction(ranks)
        ordered = T.deterministic_order(score_map, candidate_order)
        ranking_rows.append(
            {
                "dataset_key": spec.key,
                "dataset": spec.label,
                "repeat": repeat,
                "seed": seed,
                "metric": metric,
                "candidate_order_json": json.dumps(candidate_order),
                "validation_scores_json": json.dumps(values),
                "midranks_json": json.dumps(ranks.tolist()),
                "tie_correction_T": correction,
                "tied_group_count": tied_groups,
                "largest_tie_size": largest_tie,
                "deterministic_selection_order": ">".join(ordered),
                "selected_candidate": ordered[0],
                "selection_tie_break": "candidate_declaration_order_only_for_exact_score_ties",
            }
        )
        selected_index = candidate_order.index(ordered[0])
        utility_rows.append(
            {
                "dataset_key": spec.key,
                "dataset": spec.label,
                "repeat": repeat,
                "seed": seed,
                "metric": metric,
                "selected_candidate": ordered[0],
                "validation_metric_value": score_map[ordered[0]],
                "test_mcc": test_values[selected_index],
            }
        )

    phase("temperature_shift")
    from scipy.stats import spearmanr

    baseline_ranks = {
        metric: T.midrank_vector(
            [validation_scores[name][metric] for name in candidate_order]
        )
        for metric in H.METRICS
    }
    calibration_rows: list[dict] = []
    for temperature in H.TEMPS:
        shifted_scores = {
            name: H.selection_scores(
                y_validation, H.temperature_scale(validation_probs[name], temperature)
            )[0]
            for name in candidate_order
        }
        for metric in H.METRICS:
            shifted_ranks = T.midrank_vector(
                [shifted_scores[name][metric] for name in candidate_order]
            )
            agreement = spearmanr(baseline_ranks[metric], shifted_ranks).statistic
            calibration_rows.append(
                {
                    "dataset_key": spec.key,
                    "dataset": spec.label,
                    "repeat": repeat,
                    "seed": seed,
                    "T": temperature,
                    "metric": metric,
                    "spearman_vs_T1": 1.0 if np.isnan(agreement) else float(agreement),
                }
            )

    phase("write_shard")
    npz_final = units_dir / f"{unit_id}.npz"
    # NOTE: the tmp name must end in .npz, otherwise np.savez appends the
    # extension and the atomic rename source would not exist.
    npz_tmp = units_dir / f"{unit_id}.attempt{attempt}.tmp.npz"
    np.savez_compressed(
        npz_tmp,
        candidate_order=np.asarray(candidate_order),
        y_validation=np.asarray(y_validation, dtype=np.int8),
        p_validation=np.column_stack(
            [validation_probs[name] for name in candidate_order]
        ).astype(np.float64),
        candidate_test_mcc=np.asarray(test_values, dtype=np.float64),
    )
    os.replace(npz_tmp, npz_final)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": unit_id,
        "dataset_key": spec.key,
        "dataset": spec.label,
        "repeat": repeat,
        "seed": seed,
        "attempt": attempt,
        "worker_pid": os.getpid(),
        "finished_at": now_iso(),
        "npz_sha256": sha256_file(npz_final),
        "validation_rows": int(len(y_validation)),
        "dataset_row": dataset_row,
        "split_row": split_row,
        "candidate_rows": candidate_rows,
        "ranking_rows": ranking_rows,
        "utility_rows": utility_rows,
        "calibration_rows": calibration_rows,
    }
    atomic_write_json(units_dir / f"{unit_id}.json", payload)
    phase("done")


# ---------------------------------------------------------------------------
# Validation and resume
# ---------------------------------------------------------------------------

def validate_unit(run_root: Path, unit_id: str) -> tuple[bool, str]:
    units_dir = run_root / "units"
    json_path = units_dir / f"{unit_id}.json"
    npz_path = units_dir / f"{unit_id}.npz"
    if not json_path.exists():
        return False, "missing_json"
    if not npz_path.exists():
        return False, "missing_npz"
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"json_parse_error:{exc}"
    required = [
        "schema_version", "unit_id", "npz_sha256", "validation_rows",
        "dataset_row", "split_row", "candidate_rows", "ranking_rows",
        "utility_rows", "calibration_rows",
    ]
    for key in required:
        if key not in payload:
            return False, f"missing_key:{key}"
    if payload["unit_id"] != unit_id:
        return False, "unit_id_mismatch"
    if len(payload["candidate_rows"]) != N_CANDIDATES:
        return False, "candidate_row_count"
    if len(payload["ranking_rows"]) != N_METRICS:
        return False, "ranking_row_count"
    if len(payload["utility_rows"]) != N_METRICS:
        return False, "utility_row_count"
    if len(payload["calibration_rows"]) != N_METRICS * N_TEMPS:
        return False, "calibration_row_count"
    split = payload["split_row"]
    for key in (
        "train_validation_group_overlap",
        "train_test_group_overlap",
        "validation_test_group_overlap",
    ):
        if int(split.get(key, -1)) != 0:
            return False, f"nonzero_overlap:{key}"
    if sha256_file(npz_path) != payload["npz_sha256"]:
        return False, "npz_hash_mismatch"
    try:
        import numpy as np

        bundle = np.load(npz_path, allow_pickle=False)
        if bundle["p_validation"].shape != (
            int(payload["validation_rows"]), N_CANDIDATES
        ):
            return False, "npz_shape_mismatch"
        if bundle["candidate_test_mcc"].shape != (N_CANDIDATES,):
            return False, "npz_testmcc_shape"
        if not np.all(np.isfinite(bundle["p_validation"])):
            return False, "npz_nonfinite"
    except Exception as exc:  # noqa: BLE001
        return False, f"npz_load_error:{exc}"
    return True, "ok"


def quarantine_unit(run_root: Path, unit_id: str, reason: str) -> None:
    units_dir = run_root / "units"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for suffix in (".json", ".npz"):
        path = units_dir / f"{unit_id}{suffix}"
        if path.exists():
            os.replace(path, units_dir / f"{unit_id}{suffix}.invalid.{stamp}")
    append_jsonl(
        run_root / "state" / "units_log.jsonl",
        {"ts": now_iso(), "unit_id": unit_id, "event": "quarantined", "reason": reason},
    )


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class Supervisor:
    def __init__(self, args: argparse.Namespace, planned_units: list[tuple[str, int]]):
        self.args = args
        self.run_root = Path(args.run_root).resolve()
        self.state_dir = self.run_root / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.planned = planned_units
        self.completed: set[str] = set()
        self.failed: dict[str, str] = {}
        self.active_unit: str | None = None
        self.active_attempt: int = 0
        self.active_child: mp.process.BaseProcess | None = None
        self.active_started: float = 0.0
        self.stop_requested = False
        self.run_started = time.time()
        self.durations: list[float] = []
        self._hb_stop = threading.Event()
        self._mesaj = shutil.which(os.environ.get("MESAJ_CMD", "mesaj"))

    # -- notifications ------------------------------------------------------
    def notify(self, text: str) -> None:
        stamp = f"[{self.args.run_id}] {text}"
        append_jsonl(self.state_dir / "notifications.jsonl", {"ts": now_iso(), "text": stamp})
        if not self._mesaj:
            return
        try:
            subprocess.run([self._mesaj, stamp], timeout=15, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:  # noqa: BLE001
            append_jsonl(
                self.state_dir / "notifications.jsonl",
                {"ts": now_iso(), "text": "NOTIFY_FAILED", "original": stamp},
            )

    # -- heartbeat ----------------------------------------------------------
    def _child_cpu_seconds(self) -> float | None:
        child = self.active_child
        if child is None or child.pid is None:
            return None
        stat = Path(f"/proc/{child.pid}/stat")
        if not stat.exists():
            return None
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
            utime, stime = int(fields[11]), int(fields[12])
            return (utime + stime) / os.sysconf("SC_CLK_TCK")
        except Exception:  # noqa: BLE001
            return None

    def _active_phase(self) -> dict:
        if not self.active_unit:
            return {"phase": "idle", "phase_started_at": None}
        phase_path = self.run_root / "units" / f"{self.active_unit}.phase.jsonl"
        if not phase_path.exists():
            return {"phase": "starting", "phase_started_at": None}
        try:
            last = phase_path.read_text(encoding="utf-8").strip().splitlines()[-1]
            record = json.loads(last)
            return {"phase": record.get("phase"), "phase_started_at": record.get("ts")}
        except Exception:  # noqa: BLE001
            return {"phase": "unknown", "phase_started_at": None}

    def write_heartbeat(self) -> None:
        phase_info = self._active_phase()
        last_checkpoint = None
        log_path = self.state_dir / "units_log.jsonl"
        if log_path.exists():
            for line in reversed(log_path.read_text(encoding="utf-8").strip().splitlines()):
                try:
                    record = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if record.get("event") in {"completed", "skipped_validated"}:
                    last_checkpoint = record.get("ts")
                    break
        atomic_write_json(
            self.state_dir / "heartbeat.json",
            {
                "timestamp": now_iso(),
                "run_id": self.args.run_id,
                "unit_id": self.active_unit,
                "attempt_id": self.active_attempt,
                "pid": self.active_child.pid if self.active_child else None,
                "supervisor_pid": os.getpid(),
                "phase": phase_info["phase"],
                "phase_started_at": phase_info["phase_started_at"],
                "unit_elapsed_seconds": (
                    round(time.time() - self.active_started, 1) if self.active_unit else None
                ),
                "completed_atomic_units": len(self.completed),
                "planned_atomic_units": len(self.planned),
                "last_durable_checkpoint_at": last_checkpoint,
                "process_cpu_seconds": self._child_cpu_seconds(),
            },
        )

    def _heartbeat_loop(self) -> None:
        while not self._hb_stop.wait(self.args.heartbeat_seconds):
            try:
                self.write_heartbeat()
            except Exception:  # noqa: BLE001
                append_jsonl(
                    self.state_dir / "notifications.jsonl",
                    {"ts": now_iso(), "text": "HEARTBEAT_WRITE_FAILED",
                     "trace": traceback.format_exc(limit=3)},
                )

    # -- unit orchestration -------------------------------------------------
    def _attempt_number(self, unit_id: str) -> int:
        log_path = self.state_dir / "units_log.jsonl"
        if not log_path.exists():
            return 1
        count = 0
        for line in log_path.read_text(encoding="utf-8").strip().splitlines():
            try:
                record = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if record.get("unit_id") == unit_id and record.get("event") in {
                "failed", "timed_out", "cancelled", "quarantined",
            }:
                count += 1
        return count + 1

    def _log_unit(self, unit_id: str, event: str, **extra) -> None:
        append_jsonl(
            self.state_dir / "units_log.jsonl",
            {"ts": now_iso(), "unit_id": unit_id, "event": event,
             "attempt": self.active_attempt, **extra},
        )

    def _progress(self, unit_id: str, status: str, duration: float | None) -> None:
        remaining = len(self.planned) - len(self.completed)
        rate = (sum(self.durations) / len(self.durations)) if self.durations else None
        append_jsonl(
            self.state_dir / "progress.jsonl",
            {
                "ts": now_iso(),
                "unit_id": unit_id,
                "status": status,
                "duration_seconds": round(duration, 1) if duration else None,
                "completed": len(self.completed),
                "planned": len(self.planned),
                "mean_unit_seconds": round(rate, 1) if rate else None,
                "naive_eta_seconds": round(rate * remaining, 0) if rate else None,
            },
        )

    def run_units(self) -> None:
        ctx = mp.get_context("spawn")
        planned_ids = [f"{key}__r{repeat:02d}" for key, repeat in self.planned]

        # Resume pass: validate any existing shards before doing work.
        for unit_id in planned_ids:
            json_path = self.run_root / "units" / f"{unit_id}.json"
            if json_path.exists():
                ok, reason = validate_unit(self.run_root, unit_id)
                if ok:
                    self.completed.add(unit_id)
                    self._log_unit(unit_id, "skipped_validated")
                else:
                    quarantine_unit(self.run_root, unit_id, reason)

        milestones = {max(1, len(self.planned) // 4) * k for k in (1, 2, 3)}
        for index, (key, repeat) in enumerate(self.planned):
            unit_id = f"{key}__r{repeat:02d}"
            if unit_id in self.completed:
                continue
            if self.stop_requested:
                break
            if (
                self.args.max_run_seconds
                and time.time() - self.run_started > self.args.max_run_seconds
            ):
                self.notify("TIMED_OUT whole-run soft watchdog; checkpoints preserved")
                self._log_unit(unit_id, "run_watchdog_stop")
                break

            self.active_unit = unit_id
            self.active_attempt = self._attempt_number(unit_id)
            self.active_started = time.time()
            self._log_unit(unit_id, "running", pid=None)
            child = ctx.Process(
                target=unit_worker,
                args=(str(self.run_root), key, repeat, self.active_attempt),
                name=f"unit-{unit_id}",
            )
            child.start()
            self.active_child = child
            self.write_heartbeat()

            smoke_kill = (
                self.args.smoke_kill_unit_index is not None
                and index == self.args.smoke_kill_unit_index
            )
            if smoke_kill:
                deadline = self.active_started + self.args.smoke_kill_after_seconds
                while child.is_alive() and time.time() < deadline:
                    time.sleep(0.5)
                if child.is_alive():
                    child.terminate()
                    child.join(30)
                    self._log_unit(unit_id, "cancelled", reason="smoke_controlled_kill")
                    self.notify(f"CANCELLED smoke controlled kill at {unit_id}")
                    self.stop_requested = True
                    self.active_child = None
                    self.active_unit = None
                    break

            child.join(self.args.max_unit_seconds)
            duration = time.time() - self.active_started
            if child.is_alive():
                child.terminate()
                child.join(30)
                self.failed[unit_id] = "timed_out"
                self._log_unit(unit_id, "timed_out", duration=round(duration, 1))
                self._progress(unit_id, "timed_out", duration)
                self.notify(f"TIMED_OUT unit {unit_id} after {duration:.0f}s")
            elif child.exitcode == 0:
                ok, reason = validate_unit(self.run_root, unit_id)
                if ok:
                    self.completed.add(unit_id)
                    self.durations.append(duration)
                    self._log_unit(unit_id, "completed", duration=round(duration, 1),
                                   exitcode=0)
                    self._progress(unit_id, "completed", duration)
                else:
                    quarantine_unit(self.run_root, unit_id, reason)
                    self.failed[unit_id] = f"invalid_output:{reason}"
                    self._log_unit(unit_id, "failed", reason=reason)
                    self._progress(unit_id, "failed", duration)
                    self.notify(f"FAILED unit {unit_id}: {reason}")
            else:
                self.failed[unit_id] = f"exitcode_{child.exitcode}"
                self._log_unit(unit_id, "failed", exitcode=child.exitcode,
                               duration=round(duration, 1))
                self._progress(unit_id, "failed", duration)
                self.notify(f"FAILED unit {unit_id} exitcode={child.exitcode}")
            self.active_child = None
            self.active_unit = None
            self.write_heartbeat()

            if len(self.completed) in milestones:
                self.notify(
                    f"MILESTONE {len(self.completed)}/{len(self.planned)} units complete"
                )

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> int:
        hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        hb_thread.start()

        def handle_signal(signum, frame):  # noqa: ANN001, ARG001
            self.stop_requested = True
            self.notify(f"CANCELLED signal {signum}; resume with the same command")

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handle_signal)
            except (ValueError, OSError):
                pass

        exit_code = 1
        final_status = "failed"
        try:
            self.notify(
                f"STARTED host={os.uname().nodename if hasattr(os, 'uname') else 'win'} "
                f"pid={os.getpid()} planned_units={len(self.planned)} "
                f"per_unit_timeout={self.args.max_unit_seconds}s"
            )
            self.run_units()
            pending = len(self.planned) - len(self.completed)
            if self.stop_requested:
                final_status, exit_code = "cancelled", 130
                self.notify(
                    f"CANCELLED {len(self.completed)}/{len(self.planned)} complete; "
                    "resume with the same command"
                )
            elif pending == 0 and not self.failed:
                final_status, exit_code = "completed", 0
                self.notify(
                    f"COMPLETED {len(self.completed)}/{len(self.planned)} units in "
                    f"{(time.time() - self.run_started) / 60:.1f} min; "
                    f"results under {self.run_root}/units"
                )
            else:
                final_status, exit_code = "partial", 3
                self.notify(
                    f"PARTIAL {len(self.completed)}/{len(self.planned)} complete, "
                    f"{len(self.failed)} failed/timed out; resume with the same command"
                )
        except Exception:  # noqa: BLE001
            final_status, exit_code = "failed", 1
            append_jsonl(
                self.state_dir / "notifications.jsonl",
                {"ts": now_iso(), "text": "SUPERVISOR_EXCEPTION",
                 "trace": traceback.format_exc()},
            )
            self.notify("FAILED supervisor exception; see state/notifications.jsonl")
            raise
        finally:
            self._hb_stop.set()
            hb_thread.join(5)
            if self.active_child is not None and self.active_child.is_alive():
                self.active_child.terminate()
                self.active_child.join(30)
            try:
                self.write_heartbeat()
            except Exception:  # noqa: BLE001
                pass
            atomic_write_json(
                self.state_dir / "final_status.json",
                {
                    "run_id": self.args.run_id,
                    "status": final_status,
                    "exit_code": exit_code,
                    "ended_at": now_iso(),
                    "elapsed_seconds": round(time.time() - self.run_started, 1),
                    "completed_units": sorted(self.completed),
                    "failed_units": self.failed,
                    "planned_unit_count": len(self.planned),
                },
            )
        return exit_code


# ---------------------------------------------------------------------------
# Planning and manifest
# ---------------------------------------------------------------------------

def build_plan(dataset_keys: list[str], repeats: int) -> list[tuple[str, int]]:
    seen = set()
    plan: list[tuple[str, int]] = []
    for key in dataset_keys:
        if key in seen:
            raise SystemExit(f"Duplicate dataset key in plan: {key}")
        seen.add(key)
        for repeat in range(repeats):
            plan.append((key, repeat))
    return plan


def write_run_manifest(args: argparse.Namespace, plan: list[tuple[str, int]]) -> None:
    run_root = Path(args.run_root).resolve()
    manifest = {
        "run_id": args.run_id,
        "run_family": "expansion_dupsafe_tieaware",
        "methodology_change_id": "MCL-20260724-dataset-expansion-01",
        "created_at": now_iso(),
        "command": " ".join(sys.argv),
        "atomic_unit": "dataset x repeat (grouped duplicate-safe split, full candidate pool)",
        "planned_unit_count": len(plan),
        "planned_datasets": list(dict.fromkeys(key for key, _ in plan)),
        "repeats": args.repeats,
        "checkpoint_path_and_schema": "units/<dataset>__r<NN>.json + .npz; schema_version=1; row counts 5/7/7/28; npz sha256 embedded",
        "atomic_write_strategy": "tmp file + fsync + os.replace for every shard, heartbeat, and status write",
        "resume_command": f"python expansion_runner.py --run-root {run_root} --run-id {args.run_id} --datasets {args.datasets} --repeats {args.repeats}",
        "resume_validation_rule": "json parse + required keys + row counts + zero split overlap + npz sha256 + shape/finite checks; invalid shards quarantined, never overwritten",
        "per_unit_timeout_seconds": args.max_unit_seconds,
        "whole_run_watchdog": (
            f"soft internal limit {args.max_run_seconds}s" if args.max_run_seconds
            else "external (launcher timeout / systemd-run); internal soft limit disabled"
        ),
        "eta_basis_and_margin": "observed mean unit duration this session; naive_eta_seconds in state/progress.jsonl; no fabricated percentages",
        "max_workers_and_thread_limits": "1 unit child process at a time; BLAS threads inherited from environment",
        "progress_heartbeat_path_and_stall_threshold": "state/heartbeat.json every 60s (<=300s contract); stall requires >=2 stale intervals plus flat CPU and no new checkpoint",
        "heartbeat_cadence_schema_writer_and_atomicity": "supervisor thread, 60s cadence, atomic replace; fields per durability contract 8.1",
        "opaque_phase_supervisor_sampling_rule": "child appends units/<id>.phase.jsonl at load/group/split/fit_*/score/temperature/write; supervisor samples last phase + /proc CPU",
        "raw_output_contract": "unit shards are immutable raw evidence; aggregation reads shards only",
        "aggregate_script_and_inputs": "expansion_aggregate.py over units/*.json + units/*.npz",
        "plot_script_and_inputs": "harden_dupsafe_tieaware.make_figures_and_table via expansion_aggregate.py (reads merged CSVs only)",
        "decision_artifact_path_and_criterion_schema": "analysis/decision_criteria.csv + analysis/decision_summary.json (criterion_id,value,threshold,pass,contributes_to)",
        "decision_discriminator_statistics_persisted": True,
        "notification_lifecycle": "mesaj CLI best-effort: STARTED/MILESTONE/COMPLETED/FAILED/TIMED_OUT/CANCELLED; mirrored to state/notifications.jsonl",
        "terminal_status_paths": "state/final_status.json + state/units_log.jsonl",
        "partial_result_promotion_policy": "prohibited",
    }
    atomic_write_json(run_root / "RUN_MANIFEST.json", manifest)


def write_status_md(args: argparse.Namespace, plan: list[tuple[str, int]], phase: str) -> None:
    run_root = Path(args.run_root).resolve()
    lines = [
        f"# STATUS - {args.run_id}",
        "",
        f"- Phase: {phase}",
        f"- Updated: {now_iso()}",
        f"- Methodology change: MCL-20260724-dataset-expansion-01",
        f"- Planned units: {len(plan)} ({args.repeats} repeats x {len({k for k, _ in plan})} datasets)",
        f"- Resume: `python expansion_runner.py --run-root {run_root} --run-id {args.run_id} --datasets {args.datasets} --repeats {args.repeats}`",
        "- Live progress: `state/heartbeat.json`, `state/progress.jsonl`",
        "- Terminal status: `state/final_status.json`",
    ]
    (run_root / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--datasets", required=True,
        help="'all', 'canonical_only', 'new_only', or comma-separated dataset keys",
    )
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--max-unit-seconds", type=int, default=3600)
    parser.add_argument("--max-run-seconds", type=int, default=0)
    parser.add_argument("--heartbeat-seconds", type=int, default=HEARTBEAT_INTERVAL_SECONDS,
                        help="Heartbeat cadence; contract maximum is 300")
    parser.add_argument("--prepare-expansion", action="store_true",
                        help="Download/cache expansion datasets first (network)")
    parser.add_argument("--smoke-kill-unit-index", type=int, default=None,
                        help="Durability smoke: kill the child at this plan index")
    parser.add_argument("--smoke-kill-after-seconds", type=float, default=2.0)
    args = parser.parse_args()

    import harden_dupsafe_full as H
    import expansion_datasets as X

    if args.repeats is None:
        args.repeats = H.N_REPEATS

    if args.prepare_expansion:
        X.prepare_expansion_datasets()

    if args.datasets in {"all", "canonical_only", "new_only"}:
        specs = X.combined_specs(args.datasets)
        keys = [spec.key for spec in specs]
    else:
        keys = [item.strip() for item in args.datasets.split(",") if item.strip()]
        for key in keys:
            X.get_spec(key)  # fail fast on unknown keys

    plan = build_plan(keys, args.repeats)
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "units").mkdir(exist_ok=True)
    write_run_manifest(args, plan)
    write_status_md(args, plan, phase="running")

    supervisor = Supervisor(args, plan)
    exit_code = supervisor.start()
    write_status_md(args, plan, phase=f"ended:{exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
