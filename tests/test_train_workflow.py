from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import IterableDataset

from bitnet_embed.data.loaders import IterableExampleDataset
from bitnet_embed.train.workflow import (
    build_train_dataset,
    build_training_config,
    configure_trainable_parameters,
    run_training,
)


def test_build_training_config_reads_max_update_steps() -> None:
    config = {
        "training": {
            "epochs": 2,
            "micro_batch_size": 2,
            "grad_accum_steps": 4,
            "max_update_steps": 123,
        }
    }
    training = build_training_config(config)
    assert training.max_update_steps == 123


def test_build_train_dataset_supports_lazy_materialization() -> None:
    dataset, dataset_format = build_train_dataset(
        {
            "train_sets": [
                {
                    "local_path": "data/smoke/pairs.jsonl",
                    "format": "pair",
                    "materialization": "lazy",
                }
            ]
        }
    )
    assert dataset_format == "pair"
    assert isinstance(dataset, IterableDataset)
    assert isinstance(dataset, IterableExampleDataset)


def test_build_training_config_parses_optional_resume_metadata() -> None:
    config = {
        "training": {
            "run_id": "run-123",
            "parent_run_id": "parent-456",
            "plan_name": "plan-a",
            "resume_from_checkpoint": "runs/example/checkpoints/step-00001",
        }
    }
    training = build_training_config(config)
    assert training.run_id == "run-123"
    assert training.parent_run_id == "parent-456"
    assert training.plan_name == "plan-a"
    assert training.resume_from_checkpoint == "runs/example/checkpoints/step-00001"


def test_run_training_writes_ledger_entry(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: workflow_ledger_smoke",
                "seed: 42",
                "model:",
                "  backend: toy",
                "  projection_dim: 8",
                "  normalize: true",
                "tokenization:",
                "  max_length: 32",
                "training:",
                "  mode: head_only",
                "  epochs: 1",
                "  micro_batch_size: 2",
                "  grad_accum_steps: 1",
                "  lr: 0.001",
                "  weight_decay: 0.0",
                "  warmup_ratio: 0.1",
                "  eval_every_steps: 100",
                "  save_every_steps: 1",
                f"  run_root: {tmp_path}",
                "data:",
                "  train_sets:",
                "    - local_path: data/smoke/pairs.jsonl",
                "      format: pair",
                "      split: train",
                "loss:",
                "  temperature: 0.05",
            ]
        ),
        encoding="utf-8",
    )
    summary = run_training(str(config_path))
    assert summary.global_step > 0

    ledger_path = tmp_path / "ledger.jsonl"
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["run_id"] == summary.run_id
    assert payload["experiment_name"] == "workflow_ledger_smoke"
    assert payload["status"] == "completed"
    assert payload["config_path"] == str(config_path)
    assert Path(payload["summary_path"]).exists()
    assert payload["checkpoint_dir"] == summary.checkpoint_dir
    assert payload["metrics"]["global_step"] == summary.global_step


def test_configure_trainable_parameters_lora_freezes_backbone_and_keeps_lora() -> None:
    from unittest.mock import MagicMock

    model = MagicMock()
    model.parameters.return_value = [MagicMock(), MagicMock()]
    model.named_parameters.return_value = [("lora_A", MagicMock()), ("base", MagicMock())]
    model.projection.parameters.return_value = [MagicMock()]

    configure_trainable_parameters(model, "lora")

    model.parameters.assert_called()
    model.named_parameters.assert_called()
    model.projection.parameters.assert_called()


def test_configure_trainable_parameters_rejects_unknown_mode() -> None:
    class _DummyModel:
        pass

    try:
        configure_trainable_parameters(_DummyModel(), "unknown")
    except ValueError as exc:
        assert "Unsupported training mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown mode")
