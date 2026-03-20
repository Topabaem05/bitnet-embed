from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from bitnet_embed.data.collators import PairCollator, TripletCollator
from bitnet_embed.data.loaders import ExampleDataset, build_dataset_spec, load_examples
from bitnet_embed.data.schemas import (
    PairExample,
    QueryDocumentExample,
    ScoredPairExample,
    TripletExample,
)
from bitnet_embed.eval.harness import evaluate_query_documents, evaluate_scored_pairs
from bitnet_embed.losses.contrastive import SymmetricInfoNCELoss
from bitnet_embed.losses.triplet import CosineTripletLoss
from bitnet_embed.modeling.prompts import PromptConfig
from bitnet_embed.train.factory import build_model, freeze_backbone, unfreeze_all
from bitnet_embed.train.trainer import EmbeddingTrainer, TrainingConfig, TrainingSummary
from bitnet_embed.utils.io import load_yaml
from bitnet_embed.utils.seed import set_seed


def load_config(path: str) -> dict[str, Any]:
    return load_yaml(path)


def build_training_config(config: dict[str, Any]) -> TrainingConfig:
    training = config.get("training", {})
    return TrainingConfig(
        experiment_name=str(config.get("experiment_name", "bitnet-smoke")),
        mode=str(training.get("mode", "head_only")),
        epochs=int(training.get("epochs", 1)),
        micro_batch_size=int(training.get("micro_batch_size", 2)),
        grad_accum_steps=int(training.get("grad_accum_steps", 1)),
        lr=float(training.get("lr", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        warmup_ratio=float(training.get("warmup_ratio", 0.1)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", False)),
        bf16=bool(training.get("bf16", False)),
        log_every_steps=int(training.get("log_every_steps", 10)),
        eval_every_steps=int(training.get("eval_every_steps", 50)),
        save_every_steps=int(training.get("save_every_steps", 50)),
        max_length=int(config.get("tokenization", {}).get("max_length", 64)),
        run_root=str(training.get("run_root", "runs")),
        seed=int(config.get("seed", 42)),
    )


def load_data_config(config: dict[str, Any]) -> dict[str, Any]:
    data_config_path = config.get("data_config")
    if data_config_path is None:
        data_payload = config.get("data", {})
        return data_payload if isinstance(data_payload, dict) else {}
    return load_yaml(str(data_config_path))


def build_train_dataset(
    data_config: dict[str, Any],
) -> tuple[ExampleDataset[PairExample] | ExampleDataset[TripletExample], str]:
    train_sets = data_config.get("train_sets", [])
    if not train_sets:
        raise RuntimeError("At least one training dataset must be configured")
    all_examples: list[PairExample] | list[TripletExample]
    formats = {str(payload.get("format", "pair")) for payload in train_sets}
    if len(formats) != 1:
        raise RuntimeError(f"Mixed training formats are not supported yet: {sorted(formats)}")
    dataset_format = next(iter(formats))
    loaded_examples = [load_examples(build_dataset_spec(payload)) for payload in train_sets]
    if dataset_format == "pair":
        pairs: list[PairExample] = []
        for batch in loaded_examples:
            pairs.extend(item for item in batch if isinstance(item, PairExample))
        all_examples = pairs
    elif dataset_format == "triplet":
        triplets: list[TripletExample] = []
        for batch in loaded_examples:
            triplets.extend(item for item in batch if isinstance(item, TripletExample))
        return ExampleDataset(triplets), dataset_format
    else:
        raise RuntimeError(f"Unsupported training format: {dataset_format}")
    return ExampleDataset(all_examples), dataset_format


def build_eval_fn(data_config: dict[str, Any]) -> Any:
    eval_sets = data_config.get("eval_sets", [])
    if not eval_sets:
        return None
    loaded_sets = [
        (str(payload.get("format", "pair")), load_examples(build_dataset_spec(payload)))
        for payload in eval_sets
    ]

    def evaluate(current_model: Any) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for dataset_format, examples in loaded_sets:
            if dataset_format == "scored_pair":
                scored_pairs = [item for item in examples if isinstance(item, ScoredPairExample)]
                metrics.update(evaluate_scored_pairs(current_model, scored_pairs))
            elif dataset_format == "query_document":
                query_documents = [
                    item for item in examples if isinstance(item, QueryDocumentExample)
                ]
                metrics.update(evaluate_query_documents(current_model, query_documents))
        return metrics

    return evaluate


def run_training(config_path: str, *, mode_override: str | None = None) -> TrainingSummary:
    config = load_config(config_path)
    data_config = load_data_config(config)
    training_config = build_training_config(config)
    if mode_override is not None:
        training_config.mode = mode_override
    set_seed(training_config.seed)
    model = build_model(config.get("model", {}), config.get("lora"))
    if training_config.mode == "head_only":
        freeze_backbone(model)
    else:
        unfreeze_all(model)
    if model.tokenizer is None:
        raise RuntimeError("Model tokenizer is required for training")

    prompt_config = PromptConfig()
    dataset, dataset_format = build_train_dataset(data_config)
    training_config.batch_format = dataset_format
    collator: Any
    if dataset_format == "pair":
        collator = PairCollator(
            model.tokenizer,
            max_length=training_config.max_length,
            prompt_config=prompt_config,
        )
        loss_fn: Any = SymmetricInfoNCELoss(float(config.get("loss", {}).get("temperature", 0.05)))
    else:
        collator = TripletCollator(
            model.tokenizer,
            max_length=training_config.max_length,
            prompt_config=prompt_config,
        )
        loss_fn = CosineTripletLoss(float(config.get("loss", {}).get("margin", 0.2)))
    dataloader: DataLoader[PairExample | TripletExample] = DataLoader(
        dataset,
        batch_size=training_config.micro_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    trainer = EmbeddingTrainer(model, loss_fn, training_config)
    eval_fn = build_eval_fn(data_config)
    summary = trainer.train(
        dataloader,
        eval_fn=eval_fn,
        config_snapshot={**config, "resolved_data": data_config},
    )
    output_path = (
        Path(training_config.run_root)
        / training_config.experiment_name
        / "artifacts"
        / "summary.json"
    )
    output_path.write_text(
        json.dumps(
            summary.metrics
            | {
                "avg_loss": summary.avg_loss,
                "global_step": summary.global_step,
                "throughput": summary.throughput,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return summary
