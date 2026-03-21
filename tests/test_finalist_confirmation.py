from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bitnet_embed.eval.finalist_confirmation import run_finalist_confirmation
from bitnet_embed.utils.io import dump_json


def _write_config(
    path: Path,
    *,
    search_summary: Path,
    output_dir: Path,
    export_finalist_packages: bool,
    package_output_root: Path,
) -> None:
    path.write_text(
        "\n".join(
            [
                "confirmation_name: test-finalists",
                f"search_summary: {search_summary}",
                f"output_dir: {output_dir}",
                f"export_finalist_packages: {'true' if export_finalist_packages else 'false'}",
                f"package_output_root: {package_output_root}",
            ]
        ),
        encoding="utf-8",
    )


def test_finalists_resolve_from_final_rung(tmp_path: Path) -> None:
    summary_path = tmp_path / "search_summary.json"
    dump_json(
        summary_path,
        {
            "search_name": "demo_search",
            "search_run_id": "search-123",
            "primary_metric": "score",
            "maximize": True,
            "rungs": [
                {
                    "rung_name": "final",
                    "ranked_trials": [
                        {
                            "trial_name": "beta",
                            "run_id": "run-beta",
                            "checkpoint_dir": "/ckpt/beta",
                            "metric_value": 0.91,
                            "rank": 1,
                            "config_path": "/cfg/beta.yaml",
                            "resume_from_checkpoint": "/prev/beta",
                        },
                        {
                            "trial_name": "alpha",
                            "run_id": "run-alpha",
                            "checkpoint_dir": "/ckpt/alpha",
                            "metric_value": 0.88,
                            "rank": 2,
                            "config_path": "/cfg/alpha.yaml",
                            "resume_from_checkpoint": "/prev/alpha",
                        },
                    ],
                    "finalists": ["alpha", "beta"],
                }
            ],
            "best_trial": {
                "trial_name": "beta",
                "run_id": "run-beta",
                "checkpoint_dir": "/ckpt/beta",
                "metric_value": 0.91,
                "rank": 1,
                "config_path": "/cfg/beta.yaml",
                "resume_from_checkpoint": "/prev/beta",
            },
        },
    )
    config_path = tmp_path / "finalists.yaml"
    _write_config(
        config_path,
        search_summary=summary_path,
        output_dir=tmp_path / "out",
        export_finalist_packages=False,
        package_output_root=tmp_path / "packages",
    )

    payload = run_finalist_confirmation(str(config_path))

    assert [item["trial_name"] for item in payload["finalists"]] == ["beta", "alpha"]
    assert payload["finalist_count"] == 2
    assert payload["candidates"][0]["rank"] == 1
    assert payload["candidates"][1]["rank"] == 2
    assert (tmp_path / "out" / "finalist_confirmation.json").exists()
    assert (tmp_path / "out" / "finalist_confirmation.md").exists()


def test_finalists_fallback_to_best_trial_when_finalists_absent(tmp_path: Path) -> None:
    summary_path = tmp_path / "search_summary.json"
    dump_json(
        summary_path,
        {
            "search_name": "fallback_search",
            "search_run_id": "fallback-123",
            "primary_metric": "score",
            "maximize": True,
            "rungs": [
                {
                    "rung_name": "final",
                    "ranked_trials": [
                        {
                            "trial_name": "beta",
                            "run_id": "run-beta",
                            "checkpoint_dir": "/ckpt/beta",
                            "metric_value": 0.95,
                            "rank": 1,
                            "config_path": "/cfg/beta.yaml",
                            "resume_from_checkpoint": "/prev/beta",
                        }
                    ],
                }
            ],
            "best_trial": {
                "trial_name": "beta",
                "run_id": "run-beta",
                "checkpoint_dir": "/ckpt/beta",
                "metric_value": 0.95,
                "rank": 1,
                "config_path": "/cfg/beta.yaml",
                "resume_from_checkpoint": "/prev/beta",
            },
        },
    )
    config_path = tmp_path / "finalists.yaml"
    _write_config(
        config_path,
        search_summary=summary_path,
        output_dir=tmp_path / "out",
        export_finalist_packages=False,
        package_output_root=tmp_path / "packages",
    )

    payload = run_finalist_confirmation(str(config_path))

    assert payload["finalist_count"] == 1
    finalist = payload["finalists"][0]
    assert finalist["trial_name"] == "beta"
    assert finalist["run_id"] == "run-beta"
    assert finalist["checkpoint_dir"] == "/ckpt/beta"
    assert finalist["metric_value"] == 0.95
    assert finalist["rank"] == 1
    assert finalist["config_path"] == "/cfg/beta.yaml"
    assert finalist["resume_from_checkpoint"] == "/prev/beta"
    assert finalist["package_manifest_path"] is None


def test_package_export_runs_only_for_finalists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path = tmp_path / "search_summary.json"
    dump_json(
        summary_path,
        {
            "search_name": "export_search",
            "search_run_id": "export-123",
            "primary_metric": "score",
            "maximize": True,
            "rungs": [
                {
                    "rung_name": "final",
                    "ranked_trials": [
                        {
                            "trial_name": "alpha",
                            "run_id": "run-alpha",
                            "checkpoint_dir": "/ckpt/alpha",
                            "metric_value": 0.92,
                            "rank": 1,
                            "config_path": "/cfg/alpha.yaml",
                            "resume_from_checkpoint": "/prev/alpha",
                        },
                        {
                            "trial_name": "beta",
                            "run_id": "run-beta",
                            "checkpoint_dir": "/ckpt/beta",
                            "metric_value": 0.90,
                            "rank": 2,
                            "config_path": "/cfg/beta.yaml",
                            "resume_from_checkpoint": "/prev/beta",
                        },
                        {
                            "trial_name": "gamma",
                            "run_id": "run-gamma",
                            "checkpoint_dir": "/ckpt/gamma",
                            "metric_value": 0.88,
                            "rank": 3,
                            "config_path": "/cfg/gamma.yaml",
                            "resume_from_checkpoint": "/prev/gamma",
                        },
                    ],
                    "finalists": ["alpha", "gamma"],
                }
            ],
            "best_trial": {
                "trial_name": "alpha",
                "run_id": "run-alpha",
                "checkpoint_dir": "/ckpt/alpha",
                "metric_value": 0.92,
                "rank": 1,
                "config_path": "/cfg/alpha.yaml",
                "resume_from_checkpoint": "/prev/alpha",
            },
        },
    )
    config_path = tmp_path / "finalists.yaml"
    _write_config(
        config_path,
        search_summary=summary_path,
        output_dir=tmp_path / "out",
        export_finalist_packages=True,
        package_output_root=tmp_path / "packages",
    )

    calls: list[dict[str, Any]] = []

    def fake_export_hf_package(
        checkpoint_dir: str | Path,
        output_dir: str | Path,
        *,
        package_name: str | None = None,
    ) -> dict[str, Any]:
        calls.append(
            {
                "checkpoint_dir": str(checkpoint_dir),
                "output_dir": str(output_dir),
                "package_name": package_name,
            }
        )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "config.json").write_text("{}", encoding="utf-8")
        return {"package_name": package_name or ""}

    monkeypatch.setattr(
        "bitnet_embed.eval.finalist_confirmation.export_hf_package",
        fake_export_hf_package,
    )

    payload = run_finalist_confirmation(str(config_path))

    assert [call["checkpoint_dir"] for call in calls] == ["/ckpt/alpha", "/ckpt/gamma"]
    assert all(item["trial_name"] != "beta" for item in payload["finalists"])
    assert payload["finalists"][0]["package_manifest_path"]
    assert payload["finalists"][1]["package_manifest_path"]
    saved_payload = json.loads(
        (tmp_path / "out" / "finalist_confirmation.json").read_text(encoding="utf-8")
    )
    assert saved_payload["finalist_count"] == 2
