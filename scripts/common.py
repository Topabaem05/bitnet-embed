from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from bitnet_embed.data.collators import PairCollator
from bitnet_embed.data.loaders import ExampleDataset, build_smoke_pairs
from bitnet_embed.data.schemas import PairExample
from bitnet_embed.losses.contrastive import SymmetricInfoNCELoss
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


def run_training(config_path: str, *, mode_override: str | None = None) -> TrainingSummary:
    config = load_config(config_path)
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
    collator = PairCollator(
        model.tokenizer, max_length=training_config.max_length, prompt_config=prompt_config
    )
    dataset: ExampleDataset[PairExample] = ExampleDataset(build_smoke_pairs())
    dataloader: DataLoader[PairExample] = DataLoader(
        dataset,
        batch_size=training_config.micro_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    temperature = float(config.get("loss", {}).get("temperature", 0.05))
    trainer = EmbeddingTrainer(model, SymmetricInfoNCELoss(temperature), training_config)
    summary = trainer.train(dataloader, config_snapshot=config)
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
