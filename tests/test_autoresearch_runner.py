from __future__ import annotations

from typing import Any

import pytest

from bitnet_embed.train.autoresearch import run_autoresearch_search


def test_run_autoresearch_search_returns_best_trial_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_search(config_path: str) -> dict[str, Any]:
        assert config_path == "configs/search/autoresearch_msmarco_retrieval.yaml"
        return {
            "search_name": "autoresearch_msmarco_retrieval",
            "search_run_id": "auto-123",
            "output_root": "/tmp/search",
            "primary_metric": "avg_loss",
            "maximize": False,
            "best_trial": {
                "trial_name": "alpha",
                "metric_value": 0.123,
                "checkpoint_dir": "/tmp/checkpoints/alpha",
            },
        }

    monkeypatch.setattr("bitnet_embed.train.autoresearch.run_search", fake_run_search)

    result = run_autoresearch_search("configs/search/autoresearch_msmarco_retrieval.yaml")

    assert result["search_name"] == "autoresearch_msmarco_retrieval"
    assert result["best_trial_name"] == "alpha"
    assert result["best_metric"] == 0.123
    assert result["best_checkpoint_dir"] == "/tmp/checkpoints/alpha"
    assert result["summary_json"] == "/tmp/search/search_summary.json"
    assert result["summary_md"] == "/tmp/search/search_summary.md"
    assert result["experiment_surface"] == "configs/search/autoresearch_msmarco_retrieval.yaml"


def test_run_autoresearch_search_reports_passed_config_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_search(config_path: str) -> dict[str, Any]:
        return {
            "search_name": "smoke_budget_search",
            "search_run_id": "smoke-123",
            "output_root": "/tmp/smoke-search",
            "primary_metric": "avg_loss",
            "maximize": False,
            "best_trial": {
                "trial_name": "baseline_lr",
                "metric_value": 0.5,
                "checkpoint_dir": "/tmp/checkpoints/baseline",
            },
        }

    monkeypatch.setattr("bitnet_embed.train.autoresearch.run_search", fake_run_search)

    result = run_autoresearch_search("configs/search/smoke_budget.yaml")

    assert result["experiment_surface"] == "configs/search/smoke_budget.yaml"
