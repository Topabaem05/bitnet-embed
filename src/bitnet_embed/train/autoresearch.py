from __future__ import annotations

from pathlib import Path
from typing import Any

from bitnet_embed.train.search import run_search


def run_autoresearch_search(config_path: str) -> dict[str, Any]:
    summary = run_search(config_path)
    best_trial = summary.get("best_trial")
    if not isinstance(best_trial, dict):
        raise RuntimeError("Autoresearch search requires a best_trial in the search summary")

    output_root = Path(str(summary["output_root"]))
    result = {
        "config_path": config_path,
        "search_name": summary["search_name"],
        "search_run_id": summary["search_run_id"],
        "primary_metric": summary["primary_metric"],
        "maximize": summary["maximize"],
        "best_trial_name": best_trial["trial_name"],
        "best_metric": best_trial["metric_value"],
        "best_checkpoint_dir": best_trial.get("checkpoint_dir"),
        "summary_json": str(output_root / "search_summary.json"),
        "summary_md": str(output_root / "search_summary.md"),
        "experiment_surface": config_path,
    }
    return result
