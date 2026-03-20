from __future__ import annotations

import json
from pathlib import Path

from bitnet_embed.ledger import RunLedgerEntry, append_run_ledger_entry


def test_append_run_ledger_entry_writes_jsonl_lines(tmp_path: Path) -> None:
    ledger_path = tmp_path / "runs" / "ledger.jsonl"
    append_run_ledger_entry(
        ledger_path,
        RunLedgerEntry(
            run_id="run-a",
            experiment_name="exp-a",
            status="completed",
            config_path="configs/train/smoke.yaml",
            summary_path="runs/exp-a/artifacts/summary.json",
            checkpoint_dir="runs/exp-a/checkpoints/step-00001",
            metrics={"global_step": 1, "avg_loss": 0.1},
        ),
    )
    append_run_ledger_entry(
        ledger_path,
        RunLedgerEntry(
            run_id="run-b",
            experiment_name="exp-b",
            status="failed",
            summary_path="runs/exp-b/artifacts/summary.json",
            checkpoint_dir=None,
            metrics={},
        ),
    )

    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["run_id"] == "run-a"
    assert first["status"] == "completed"
    assert first["metrics"]["global_step"] == 1
    assert second["run_id"] == "run-b"
    assert second["status"] == "failed"
