from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from torch.utils.data import DataLoader, IterableDataset

from bitnet_embed.data.collators import PairCollator, TripletCollator
from bitnet_embed.data.loaders import (
    ExampleDataset,
    IterableExampleDataset,
    build_dataset_spec,
    iter_examples_from_specs,
    load_examples,
)
from bitnet_embed.data.schemas import (
    PairExample,
    QueryDocumentExample,
    ScoredPairExample,
    TripletExample,
)
from bitnet_embed.eval.harness import evaluate_query_documents, evaluate_scored_pairs
from bitnet_embed.ledger import RunLedgerEntry, append_run_ledger_entry
from bitnet_embed.losses.contrastive import SymmetricInfoNCELoss
from bitnet_embed.losses.triplet import CosineTripletLoss
from bitnet_embed.modeling.prompts import PromptConfig
from bitnet_embed.train.factory import build_model, freeze_backbone, unfreeze_all, freeze_for_lora
from bitnet_embed.train.trainer import EmbeddingTrainer, TrainingConfig, TrainingSummary
from bitnet_embed.utils.io import load_yaml
from bitnet_embed.utils.seed import set_seed

_UNSET_RESUME_FROM_CHECKPOINT: Final = object()


def load_config(path: str) -> dict[str, Any]:
    return load_yaml(path)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip()
    return candidate if candidate else None


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
        bf16=bool(
            training.get("bf16", False)
            or str(training.get("mixed_precision", "")).lower() == "bf16"
        ),
        fp16=bool(
            training.get("fp16", False)
            or str(training.get("mixed_precision", "")).lower() == "fp16"
        ),
        log_every_steps=int(training.get("log_every_steps", 10)),
        eval_every_steps=int(training.get("eval_every_steps", 50)),
        save_every_steps=int(training.get("save_every_steps", 50)),
        max_update_steps=(
            int(training["max_update_steps"])
            if training.get("max_update_steps") is not None
            else None
        ),
        max_grad_norm=(
            float(training["max_grad_norm"]) if training.get("max_grad_norm") is not None else 1.0
        ),
        fail_on_non_finite_loss=bool(training.get("fail_on_non_finite_loss", True)),
        fail_on_non_finite_grads=bool(training.get("fail_on_non_finite_grads", True)),
        check_parameters_every_steps=int(training.get("check_parameters_every_steps", 50)),
        check_eval_outputs_finite=bool(training.get("check_eval_outputs_finite", True)),
        log_grad_norm=bool(training.get("log_grad_norm", True)),
        max_length=int(config.get("tokenization", {}).get("max_length", 64)),
        run_root=str(training.get("run_root", "runs")),
        seed=int(config.get("seed", 42)),
        run_id=_optional_string(training.get("run_id")),
        parent_run_id=_optional_string(training.get("parent_run_id")),
        plan_name=_optional_string(training.get("plan_name")),
        resume_from_checkpoint=_optional_string(training.get("resume_from_checkpoint")),
        resume_weights_only=bool(training.get("resume_weights_only", False)),
    )


def load_data_config(config: dict[str, Any]) -> dict[str, Any]:
    data_config_path = config.get("data_config")
    if data_config_path is None:
        data_payload = config.get("data", {})
        return data_payload if isinstance(data_payload, dict) else {}
    return load_yaml(str(data_config_path))


def build_train_dataset(
    data_config: dict[str, Any],
) -> tuple[
    ExampleDataset[PairExample]
    | ExampleDataset[TripletExample]
    | IterableExampleDataset[PairExample]
    | IterableExampleDataset[TripletExample],
    str,
]:
    train_sets = data_config.get("train_sets", [])
    if not train_sets:
        raise RuntimeError("At least one training dataset must be configured")
    formats = {str(payload.get("format", "pair")) for payload in train_sets}
    if len(formats) != 1:
        raise RuntimeError(f"Mixed training formats are not supported yet: {sorted(formats)}")
    dataset_format = next(iter(formats))
    specs = [build_dataset_spec(payload) for payload in train_sets]
    use_lazy_loading = any(spec.materialization == "lazy" for spec in specs)

    if use_lazy_loading:
        if dataset_format == "pair":

            def pair_iterator() -> Any:
                for item in iter_examples_from_specs(specs):
                    if isinstance(item, PairExample):
                        yield item

            return IterableExampleDataset(pair_iterator), dataset_format

        if dataset_format == "triplet":

            def triplet_iterator() -> Any:
                for item in iter_examples_from_specs(specs):
                    if isinstance(item, TripletExample):
                        yield item

            return IterableExampleDataset(triplet_iterator), dataset_format
        raise RuntimeError(f"Unsupported training format: {dataset_format}")

    loaded_examples = [load_examples(spec) for spec in specs]
    if dataset_format == "pair":
        pairs: list[PairExample] = []
        for batch in loaded_examples:
            pairs.extend(item for item in batch if isinstance(item, PairExample))
        return ExampleDataset(pairs), dataset_format
    if dataset_format == "triplet":
        triplets: list[TripletExample] = []
        for batch in loaded_examples:
            triplets.extend(item for item in batch if isinstance(item, TripletExample))
        return ExampleDataset(triplets), dataset_format
    raise RuntimeError(f"Unsupported training format: {dataset_format}")


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


def configure_trainable_parameters(model: Any, mode: str) -> None:
    if mode == "head_only":
        freeze_backbone(model)
        return
    if mode == "lora":
        freeze_for_lora(model)
        return
    if mode == "full_ft":
        unfreeze_all(model)
        return
    raise ValueError(f"Unsupported training mode: {mode}")


def run_training(
    config_path: str,
    *,
    mode_override: str | None = None,
    plan_name: str | None = None,
    parent_run_id: str | None = None,
    resume_from_checkpoint: str | None | object = _UNSET_RESUME_FROM_CHECKPOINT,
) -> TrainingSummary:
    config = load_config(config_path)
    data_config = load_data_config(config)
    training_config = build_training_config(config)
    if mode_override is not None:
        training_config.mode = mode_override
    if plan_name is not None:
        training_config.plan_name = plan_name
    if parent_run_id is not None:
        training_config.parent_run_id = parent_run_id
    if resume_from_checkpoint is not _UNSET_RESUME_FROM_CHECKPOINT:
        training_config.resume_from_checkpoint = _optional_string(resume_from_checkpoint)
    set_seed(training_config.seed)

    model_config = config.get("model", {})
    if "gradient_checkpointing" not in model_config:
        model_config["gradient_checkpointing"] = training_config.gradient_checkpointing

    model = build_model(model_config, config.get("lora"))
    configure_trainable_parameters(model, training_config.mode)
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
        shuffle=not isinstance(dataset, IterableDataset),
        collate_fn=collator,
    )
    trainer = EmbeddingTrainer(model, loss_fn, training_config)
    eval_fn = build_eval_fn(data_config)
    output_path = (
        Path(training_config.run_root)
        / training_config.experiment_name
        / "artifacts"
        / "summary.json"
    )
    ledger_path = Path(training_config.run_root) / "ledger.jsonl"
    try:
        summary = trainer.train(
            dataloader,
            eval_fn=eval_fn,
            config_snapshot={**config, "resolved_data": data_config},
        )

        import math

        if not math.isfinite(summary.avg_loss):
            raise RuntimeError(f"Training failed: avg_loss is not finite ({summary.avg_loss})")
        if not math.isfinite(summary.throughput):
            raise RuntimeError(f"Training failed: throughput is not finite ({summary.throughput})")
        for k, v in summary.metrics.items():
            if isinstance(v, (int, float)) and not math.isfinite(v):
                raise RuntimeError(f"Training failed: metric {k} is not finite ({v})")

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
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        append_run_ledger_entry(
            ledger_path,
            RunLedgerEntry(
                run_id=summary.run_id,
                experiment_name=training_config.experiment_name,
                status="completed",
                config_path=config_path,
                summary_path=str(output_path),
                checkpoint_dir=summary.checkpoint_dir,
                metrics={
                    "global_step": summary.global_step,
                    "avg_loss": summary.avg_loss,
                    "throughput": summary.throughput,
                    **summary.metrics,
                },
                parent_run_id=training_config.parent_run_id,
                plan_name=training_config.plan_name,
                resume_from=training_config.resume_from_checkpoint,
            ),
        )
        return summary
    except Exception:
        append_run_ledger_entry(
            ledger_path,
            RunLedgerEntry(
                run_id=trainer.metadata.run_id,
                experiment_name=training_config.experiment_name,
                status="failed",
                config_path=config_path,
                summary_path=str(output_path),
                checkpoint_dir=None,
                metrics={},
                parent_run_id=training_config.parent_run_id,
                plan_name=training_config.plan_name,
                resume_from=training_config.resume_from_checkpoint,
            ),
        )
        raise
