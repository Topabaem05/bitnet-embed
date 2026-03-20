from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from bitnet_embed.utils.io import ensure_dir


@dataclass(slots=True)
class RunLedgerEntry:
    run_id: str
    experiment_name: str
    status: str
    summary_path: str
    checkpoint_dir: str | None
    metrics: dict[str, float | int] = field(default_factory=dict)
    config_path: str | None = None
    parent_run_id: str | None = None
    plan_name: str | None = None
    resume_from: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def append_run_ledger_entry(ledger_path: Path | str, entry: RunLedgerEntry) -> None:
    destination = Path(ledger_path)
    ensure_dir(destination.parent)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
