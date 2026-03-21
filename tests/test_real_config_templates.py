from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest
import torch

from bitnet_embed.data.loaders import build_dataset_spec
from bitnet_embed.data.schemas import PairExample, TripletExample
from bitnet_embed.train.plan import load_stage_specs
from bitnet_embed.train.search import load_search_spec
from bitnet_embed.train.workflow import (
    build_eval_fn,
    build_train_dataset,
    build_training_config,
    load_config,
    load_data_config,
)
from bitnet_embed.utils.io import load_yaml


class _FakeDataset(list[dict[str, object]]):
    pass


class _FakeModel:
    def encode(self, texts: Sequence[str], config: object) -> torch.Tensor:
        _ = config
        rows = []
        for text in texts:
            score = sum(ord(character) for character in text)
            rows.append([float(len(text)), float(score % 997), float(score % 37)])
        return torch.tensor(rows, dtype=torch.float32)


def _fake_dataset_rows(name: str, subset: str | None) -> list[dict[str, object]]:
    if name == "sentence-transformers/natural-questions" and subset == "pair":
        return [
            {
                "query": "who wrote hamlet",
                "answer": "William Shakespeare wrote Hamlet.",
                "source": "nq",
            },
            {
                "query": "capital of france",
                "answer": "Paris is the capital of France.",
                "source": "nq",
            },
        ]
    if name == "tomaarsen/natural-questions-hard-negatives" and subset == "triplet-5":
        return [
            {
                "query": "who wrote hamlet",
                "document": "William Shakespeare wrote Hamlet.",
                "hard_negative": "The Pacific Ocean is the largest ocean.",
                "source": "nq",
            },
            {
                "query": "capital of france",
                "document": "Paris is the capital of France.",
                "hard_negative": "Bananas are rich in potassium.",
                "source": "nq",
            },
        ]
    if name == "sentence-transformers/msmarco-bm25" and subset == "triplet":
        return [
            {
                "query": "symptoms of lupus",
                "document": "Common lupus symptoms include fatigue and joint pain.",
                "hard_negative": "Stock indexes closed higher today.",
                "source": "msmarco",
            },
            {
                "query": "how to boil eggs",
                "document": "Boil eggs for about ten minutes for a firm yolk.",
                "hard_negative": "Wind turbines convert wind into electricity.",
                "source": "msmarco",
            },
        ]
    if name.startswith("irds/beir_"):
        return [
            {
                "query": "symptoms of lupus",
                "document": "Common lupus symptoms include fatigue and joint pain.",
                "label": 1,
                "source": name,
            },
            {
                "query": "symptoms of lupus",
                "document": "Stock indexes closed higher today.",
                "label": 0,
                "source": name,
            },
        ]
    raise AssertionError(f"Unexpected dataset request: name={name!r}, subset={subset!r}")


def _install_fake_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = ModuleType("datasets")

    def load_dataset(
        name: str,
        subset: str | None = None,
        *,
        split: str = "train",
        streaming: bool = False,
    ) -> _FakeDataset:
        _ = (split, streaming)
        return _FakeDataset(_fake_dataset_rows(name, subset))

    fake_module.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_module)


@pytest.mark.parametrize(
    ("config_path", "expected_mode", "expected_format", "expected_example_type"),
    [
        (
            "configs/train/nq_head_only.yaml",
            "head_only",
            "pair",
            PairExample,
        ),
        (
            "configs/train/nq_lora_retrieval.yaml",
            "lora",
            "triplet",
            TripletExample,
        ),
        (
            "configs/train/nq_full_ft_retrieval.yaml",
            "full_ft",
            "triplet",
            TripletExample,
        ),
        (
            "configs/train/msmarco_lora_retrieval.yaml",
            "lora",
            "triplet",
            TripletExample,
        ),
        (
            "configs/train/msmarco_full_ft_retrieval.yaml",
            "full_ft",
            "triplet",
            TripletExample,
        ),
    ],
)
def test_real_train_configs_parse_with_existing_workflow_builders(
    monkeypatch: pytest.MonkeyPatch,
    config_path: str,
    expected_mode: str,
    expected_format: str,
    expected_example_type: type[PairExample] | type[TripletExample],
) -> None:
    _install_fake_datasets(monkeypatch)

    config = load_config(config_path)
    training = build_training_config(config)
    assert training.mode == expected_mode

    data_config = load_data_config(config)
    train_specs = [build_dataset_spec(payload) for payload in data_config["train_sets"]]
    eval_specs = [build_dataset_spec(payload) for payload in data_config["eval_sets"]]
    assert all(spec.format == expected_format for spec in train_specs)
    assert all(spec.format == "query_document" for spec in eval_specs)

    dataset, dataset_format = build_train_dataset(data_config)
    assert dataset_format == expected_format
    first_example = next(iter(dataset))
    assert isinstance(first_example, expected_example_type)

    eval_fn = build_eval_fn(data_config)
    assert eval_fn is not None
    metrics = eval_fn(_FakeModel())
    assert "mrr@10" in metrics
    assert "recall@10" in metrics
    assert "ndcg@10" in metrics


def test_beir_eval_config_parses_as_query_document_sets() -> None:
    payload = load_yaml("configs/data/beir_eval.yaml")
    eval_specs = [build_dataset_spec(entry) for entry in payload["eval_sets"]]
    assert len(eval_specs) == 3
    assert all(spec.format == "query_document" for spec in eval_specs)


def test_real_search_configs_parse_with_current_loader() -> None:
    search_specs = {
        "configs/search/nq_retrieval_budget.yaml": "configs/train/nq_lora_retrieval.yaml",
        "configs/search/msmarco_retrieval_budget.yaml": "configs/train/msmarco_lora_retrieval.yaml",
    }

    for config_path, expected_base_config in search_specs.items():
        spec = load_search_spec(config_path)
        assert spec.base_config == expected_base_config
        assert spec.primary_metric == "mrr@10"
        assert spec.maximize is True
        assert len(spec.trials) == 3
        assert len(spec.rungs) == 2
        assert spec.rungs[0].max_update_steps < spec.rungs[1].max_update_steps
        assert Path(spec.base_config).exists()


def test_real_stage_plans_reference_real_train_progressions() -> None:
    expected_stage_configs = {
        "configs/plan/nq_stages.yaml": [
            "configs/train/nq_head_only.yaml",
            "configs/train/nq_lora_retrieval.yaml",
            "configs/train/nq_full_ft_retrieval.yaml",
        ],
        "configs/plan/msmarco_stages.yaml": [
            "configs/train/msmarco_lora_retrieval.yaml",
            "configs/train/msmarco_full_ft_retrieval.yaml",
        ],
    }

    for config_path, expected_configs in expected_stage_configs.items():
        plan_name, stages, output_root = load_stage_specs(load_yaml(config_path))
        assert plan_name
        assert output_root.startswith("runs/plans/")
        assert [stage.train_config for stage in stages] == expected_configs
        assert stages[0].resume_policy == "none"
        assert all(Path(stage.train_config).exists() for stage in stages)
        for stage in stages[1:]:
            assert stage.resume_policy == "previous_stage_checkpoint"
