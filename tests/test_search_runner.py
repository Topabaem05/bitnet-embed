from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bitnet_embed.train.search import run_search
from bitnet_embed.train.trainer import TrainingSummary
from bitnet_embed.utils.io import load_yaml


def test_run_search_executes_rungs_and_promotes_with_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_config_path = tmp_path / "base.yaml"
    base_config_path.write_text(
        "\n".join(
            [
                "experiment_name: smoke_base",
                "model:",
                "  backend: toy",
                "training:",
                f"  run_root: {tmp_path / 'runs'}",
                "  max_update_steps: 99",
            ]
        ),
        encoding="utf-8",
    )

    search_config_path = tmp_path / "search.yaml"
    search_config_path.write_text(
        "\n".join(
            [
                "search_name: queue_promotion",
                f"base_config: {base_config_path}",
                f"output_root: {tmp_path / 'search-out'}",
                "primary_metric: score",
                "maximize: true",
                "trials:",
                "  - name: alpha",
                "    overrides:",
                "      training:",
                "        lr: 0.001",
                "  - name: beta",
                "    overrides:",
                "      training:",
                "        lr: 0.0005",
                "  - name: gamma",
                "    overrides:",
                "      training:",
                "        lr: 0.0001",
                "rungs:",
                "  - name: short",
                "    max_update_steps: 2",
                "    promote_top_k: 2",
                "  - name: long",
                "    max_update_steps: 4",
            ]
        ),
        encoding="utf-8",
    )

    calls: list[dict[str, Any]] = []
    scores_by_rung = {
        1: {"alpha": 0.30, "beta": 0.95, "gamma": 0.60},
        2: {"beta": 0.97, "gamma": 0.65},
    }

    def fake_run_training(
        config_path: str,
        *,
        mode_override: str | None = None,
        plan_name: str | None = None,
        parent_run_id: str | None = None,
    ) -> TrainingSummary:
        _ = mode_override
        payload = load_yaml(config_path)
        training = payload.get("training", {})
        assert isinstance(training, dict)
        run_id = str(training["run_id"])
        parts = run_id.split(":")
        assert len(parts) == 3
        trial_name = parts[1]
        rung_index = int(parts[2].replace("r", ""))
        calls.append(
            {
                "trial_name": trial_name,
                "rung_index": rung_index,
                "max_update_steps": int(training["max_update_steps"]),
                "resume_from_checkpoint": training.get("resume_from_checkpoint"),
                "plan_name": plan_name,
                "parent_run_id": parent_run_id,
            }
        )
        return TrainingSummary(
            run_id=run_id,
            global_step=int(training["max_update_steps"]),
            avg_loss=1.0,
            throughput=10.0,
            checkpoint_dir=f"/checkpoints/{trial_name}/r{rung_index}",
            metrics={"score": scores_by_rung[rung_index][trial_name]},
        )

    monkeypatch.setattr("bitnet_embed.train.search.run_training", fake_run_training)

    summary = run_search(str(search_config_path))
    assert summary["trial_count"] == 3
    assert summary["rung_count"] == 2
    assert summary["best_trial"]["trial_name"] == "beta"

    assert [call["trial_name"] for call in calls[:3]] == ["alpha", "beta", "gamma"]
    assert len(calls) == 5
    assert all(call["max_update_steps"] == 2 for call in calls[:3])
    assert all(call["max_update_steps"] == 4 for call in calls[3:])
    assert all(call["resume_from_checkpoint"] is None for call in calls[:3])
    assert calls[3]["resume_from_checkpoint"] in {"/checkpoints/beta/r1", "/checkpoints/gamma/r1"}
    assert calls[4]["resume_from_checkpoint"] in {"/checkpoints/beta/r1", "/checkpoints/gamma/r1"}

    output_dir = tmp_path / "search-out"
    assert (output_dir / "search_summary.json").exists()
    assert (output_dir / "search_summary.md").exists()
    payload = json.loads((output_dir / "search_summary.json").read_text(encoding="utf-8"))
    assert payload["rungs"][0]["promoted_trials"] == ["beta", "gamma"]
    assert payload["rungs"][1]["promoted_trials"] == []
    assert payload["rungs"][1]["discarded_trials"] == []
    assert payload["rungs"][1]["finalists"] == ["beta", "gamma"]


def test_run_search_raises_when_primary_metric_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_config_path = tmp_path / "base.yaml"
    base_config_path.write_text("experiment_name: smoke\ntraining: {}\n", encoding="utf-8")
    search_config_path = tmp_path / "search.yaml"
    search_config_path.write_text(
        "\n".join(
            [
                "search_name: missing_metric",
                f"base_config: {base_config_path}",
                "primary_metric: metric_not_present",
                "trials:",
                "  - name: only_trial",
                "rungs:",
                "  - name: only_rung",
                "    max_update_steps: 1",
            ]
        ),
        encoding="utf-8",
    )

    def fake_run_training(
        config_path: str,
        *,
        mode_override: str | None = None,
        plan_name: str | None = None,
        parent_run_id: str | None = None,
    ) -> TrainingSummary:
        _ = (config_path, mode_override, plan_name, parent_run_id)
        return TrainingSummary(
            run_id="trial",
            global_step=1,
            avg_loss=0.2,
            throughput=1.0,
            checkpoint_dir="/checkpoints/one",
            metrics={"other": 0.5},
        )

    monkeypatch.setattr("bitnet_embed.train.search.run_training", fake_run_training)
    with pytest.raises(RuntimeError, match="Primary metric"):
        run_search(str(search_config_path))


def test_run_search_dispatches_external_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_config_path = tmp_path / "base.yaml"
    base_config_path.write_text("experiment_name: smoke\ntraining: {}\n", encoding="utf-8")
    search_config_path = tmp_path / "search.yaml"
    search_config_path.write_text(
        "\n".join(
            [
                "search_name: external_search",
                f"base_config: {base_config_path}",
                "primary_metric: score",
                "executor: external",
                "trials:",
                "  - name: alpha",
                "rungs:",
                "  - name: only_rung",
                "    max_update_steps: 1",
            ]
        ),
        encoding="utf-8",
    )

    in_process_calls: list[str] = []
    external_calls: list[dict[str, Any]] = []

    def fake_run_training(
        config_path: str,
        *,
        mode_override: str | None = None,
        plan_name: str | None = None,
        parent_run_id: str | None = None,
    ) -> TrainingSummary:
        _ = (mode_override, plan_name, parent_run_id)
        in_process_calls.append(config_path)
        return TrainingSummary(
            run_id="in-process",
            global_step=1,
            avg_loss=0.1,
            throughput=1.0,
            checkpoint_dir="/checkpoints/in-process",
            metrics={"score": 0.1},
        )

    def fake_run_training_external(
        config_path: str,
        *,
        mode_override: str | None = None,
        plan_name: str | None = None,
        parent_run_id: str | None = None,
        resume_from_checkpoint: str | None | object = object(),
    ) -> TrainingSummary:
        _ = (mode_override, resume_from_checkpoint)
        external_calls.append(
            {
                "config_path": config_path,
                "plan_name": plan_name,
                "parent_run_id": parent_run_id,
            }
        )
        return TrainingSummary(
            run_id="external",
            global_step=1,
            avg_loss=0.1,
            throughput=1.0,
            checkpoint_dir="/checkpoints/external",
            metrics={"score": 0.9},
        )

    monkeypatch.setattr("bitnet_embed.train.search.run_training", fake_run_training)
    monkeypatch.setattr(
        "bitnet_embed.train.search.run_training_external", fake_run_training_external
    )

    summary = run_search(str(search_config_path))
    assert summary["executor"] == "external"
    assert len(external_calls) == 1
    assert len(in_process_calls) == 0
