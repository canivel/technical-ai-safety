"""
I/O utility functions for saving, loading, and logging experiment artefacts.

All path arguments accept :class:`pathlib.Path` or ``str``.  Format detection
is automatic based on file extension when *format* is ``"auto"``.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from research.config import OUTPUT_DIR


# ── Save / Load ──────────────────────────────────────────────────────────

def save_results(data: Any, path: Path, format: str = "auto") -> None:
    """Persist *data* to *path*, auto-detecting the serialisation format.

    Parameters
    ----------
    data : Any
        The object to save.
    path : Path
        Destination file path.
    format : str, optional
        One of ``"auto"``, ``"pt"``, ``"json"``, ``"csv"``, ``"pkl"``.
        When ``"auto"`` the format is inferred from the file extension.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "auto":
        format = path.suffix.lstrip(".")

    if format == "pt":
        torch.save(data, path)
    elif format == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    elif format == "csv":
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=False)
        else:
            pd.DataFrame(data).to_csv(path, index=False)
    elif format == "pkl":
        with open(path, "wb") as f:
            pickle.dump(data, f)
    else:
        # Fallback: pickle for unknown extensions
        with open(path, "wb") as f:
            pickle.dump(data, f)


def load_results(path: Path) -> Any:
    """Load a previously saved artefact, auto-detecting format by extension.

    Parameters
    ----------
    path : Path
        File to load.

    Returns
    -------
    Any
        The deserialised object.
    """
    path = Path(path)

    ext = path.suffix.lstrip(".")
    if ext == "pt":
        return torch.load(path, weights_only=False)
    elif ext == "json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif ext == "csv":
        return pd.read_csv(path)
    elif ext in ("pkl", "pickle"):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        # Attempt pickle as fallback
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Experiment Logging ───────────────────────────────────────────────────

def _default_log_path() -> Path:
    return OUTPUT_DIR / "experiment_log.jsonl"


def save_experiment_log(log_entry: dict, log_file: Path | None = None) -> None:
    """Append a timestamped entry to the experiment log (JSONL).

    Parameters
    ----------
    log_entry : dict
        Arbitrary key-value data describing the experiment event.
    log_file : Path, optional
        Path to the JSONL log file.  Defaults to
        ``OUTPUT_DIR / "experiment_log.jsonl"``.
    """
    log_file = Path(log_file) if log_file is not None else _default_log_path()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **log_entry,
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def load_experiment_log(log_file: Path | None = None) -> list[dict]:
    """Load all entries from the experiment log.

    Parameters
    ----------
    log_file : Path, optional
        Path to the JSONL log file.  Defaults to
        ``OUTPUT_DIR / "experiment_log.jsonl"``.

    Returns
    -------
    list[dict]
        List of log entries in chronological order.
    """
    log_file = Path(log_file) if log_file is not None else _default_log_path()

    if not log_file.exists():
        return []

    entries: list[dict] = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ── Experiment Snapshots ─────────────────────────────────────────────────

def create_experiment_snapshot(
    experiment_name: str,
    results: dict,
    config: dict,
) -> Path:
    """Create a timestamped directory containing experiment results and config.

    The snapshot directory is created under ``OUTPUT_DIR / "snapshots"`` with
    the naming pattern ``<experiment_name>_<YYYYMMDD_HHMMSS>``.

    Parameters
    ----------
    experiment_name : str
        Human-readable experiment identifier.
    results : dict
        Experiment results to persist as JSON.
    config : dict
        Configuration used for the experiment, saved as JSON.

    Returns
    -------
    Path
        Absolute path to the created snapshot directory.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot_dir = OUTPUT_DIR / "snapshots" / f"{experiment_name}_{timestamp}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    save_results(results, snapshot_dir / "results.json", format="json")
    save_results(config, snapshot_dir / "config.json", format="json")

    # Also log this snapshot event
    save_experiment_log(
        {
            "event": "snapshot_created",
            "experiment_name": experiment_name,
            "snapshot_path": str(snapshot_dir),
        }
    )

    return snapshot_dir


# ── Pretty Printing ──────────────────────────────────────────────────────

def format_results_table(results: dict) -> str:
    """Format a flat or single-nested results dict as an ASCII table.

    Parameters
    ----------
    results : dict
        Keys become row labels; values are displayed in the second column.
        If a value is itself a dict, it is expanded into sub-rows.

    Returns
    -------
    str
        An ASCII-formatted table suitable for terminal output.
    """
    rows: list[tuple[str, str]] = []

    for key, value in results.items():
        if isinstance(value, dict):
            rows.append((str(key), ""))
            for sub_key, sub_val in value.items():
                formatted = _format_value(sub_val)
                rows.append((f"  {sub_key}", formatted))
        else:
            rows.append((str(key), _format_value(value)))

    if not rows:
        return "(empty results)"

    # Compute column widths
    key_width = max(len(r[0]) for r in rows)
    val_width = max(len(r[1]) for r in rows)

    sep = "+" + "-" * (key_width + 2) + "+" + "-" * (val_width + 2) + "+"
    lines = [sep]

    for key, val in rows:
        lines.append(f"| {key:<{key_width}} | {val:<{val_width}} |")

    lines.append(sep)
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    """Format a single value for table display."""
    if isinstance(value, float):
        if abs(value) < 0.01 and value != 0:
            return f"{value:.4e}"
        return f"{value:.4f}"
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    if isinstance(value, (list, tuple)):
        return str(value)
    return str(value)
