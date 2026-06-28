from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


TEST_RECORD_FIELDS = [
    "timestamp",
    "evaluation",
    "model",
    "checkpoint",
    "split",
    "condition",
    "mask_type",
    "mask_ratio",
    "excluded_scenes",
    "IoU",
    "F1",
    "Precision",
    "Recall",
    "config",
    "notes",
]


def append_test_records(
    rows: dict[str, Any] | Iterable[dict[str, Any]],
    output_csv: str | Path = "outputs/s1s2_water/test_records/test_metrics_log.csv",
) -> None:
    if isinstance(rows, dict):
        rows = [rows]
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    timestamp = datetime.now().isoformat(timespec="seconds")
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TEST_RECORD_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            record = {key: "" for key in TEST_RECORD_FIELDS}
            record.update({key: row.get(key, "") for key in TEST_RECORD_FIELDS})
            record["timestamp"] = row.get("timestamp") or timestamp
            writer.writerow(record)
