from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from bitnet_embed.train.callbacks import RunMetadata
from bitnet_embed.train.loops import encode_pair_batch, encode_triplet_batch, move_batch_to_device
from bitnet_embed.train.optim import OptimizerConfig, build_optimizer, build_scheduler
from bitnet_embed.utils.io import dump_json, ensure_dir, get_git_revision
from bitnet_embed.utils.metrics import RunningAverage, ThroughputMeter


@dataclass(slots=True)
class TrainingConfig:
    experiment_name: str = "bitnet_smoke"
    mode: str = "head_only"
    epochs: int = 1
    micro_batch_size: int = 4
    grad_accum_steps: int = 1
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_checkpointing: bool = False
    bf16: bool = False
    log_every_steps: int = 10
    eval_every_steps: int = 50
    save_every_steps: int = 100
    max_length: int = 256
    run_root: str = "runs"
    seed: int = 42
    batch_format: str = "pair"


@dataclass(slots=True)
class TrainingSummary:
    global_step: int
    avg_loss: float
    throughput: float
    checkpoint_dir: str | None
    metrics: dict[str, float] = field(default_factory=dict)


def build_accelerator(config: TrainingConfig) -> Any:
    try:
        from accelerate import Accelerator
    except ImportError as exc:
        raise RuntimeError("accelerate is required for training") from exc
    mixed_precision = "bf16" if config.bf16 and torch.cuda.is_available() else "no"
    return Accelerator(
        gradient_accumulation_steps=config.grad_accum_steps,
        mixed_precision=mixed_precision,
    )


class EmbeddingTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        config: TrainingConfig,
        *,
        accelerator: Any | None = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.accelerator = accelerator or build_accelerator(config)
        self.optimizer = build_optimizer(
            model,
            OptimizerConfig(
                lr=config.lr,
                weight_decay=config.weight_decay,
                warmup_ratio=config.warmup_ratio,
                total_steps=max(1, config.epochs),
            ),
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            OptimizerConfig(
                lr=config.lr,
                weight_decay=config.weight_decay,
                warmup_ratio=config.warmup_ratio,
                total_steps=max(1, config.epochs),
            ),
        )
        self.run_dir = self._build_run_dir()
        self.metadata = RunMetadata.create(
            experiment_name=config.experiment_name,
            seed=config.seed,
            mode=config.mode,
            git_revision=get_git_revision(Path.cwd()),
        )

    def _build_run_dir(self) -> Path:
        root = Path(self.config.run_root)
        run_dir = ensure_dir(root / self.config.experiment_name)
        ensure_dir(run_dir / "checkpoints")
        ensure_dir(run_dir / "metrics")
        ensure_dir(run_dir / "configs")
        ensure_dir(run_dir / "artifacts")
        return run_dir

    def train(
        self,
        train_dataloader: DataLoader[Any],
        *,
        eval_fn: Callable[[torch.nn.Module], dict[str, float]] | None = None,
        config_snapshot: dict[str, Any] | None = None,
    ) -> TrainingSummary:
        self.model, self.optimizer, train_dataloader, self.scheduler = self.accelerator.prepare(
            self.model,
            self.optimizer,
            train_dataloader,
            self.scheduler,
        )
        global_step = 0
        running_loss = RunningAverage()
        throughput = ThroughputMeter()
        latest_metrics: dict[str, float] = {}
        checkpoint_dir: str | None = None

        for _ in range(self.config.epochs):
            self.model.train()
            for batch in train_dataloader:
                start_time = time.perf_counter()
                batch = move_batch_to_device(batch, self.accelerator.device)
                with self.accelerator.accumulate(self.model):
                    if self.config.batch_format == "pair":
                        anchor_embeddings, positive_embeddings = encode_pair_batch(
                            self.model, batch
                        )
                        loss = self.loss_fn(anchor_embeddings, positive_embeddings)
                    elif self.config.batch_format == "triplet":
                        anchor_embeddings, positive_embeddings, negative_embeddings = (
                            encode_triplet_batch(self.model, batch)
                        )
                        loss = self.loss_fn(
                            anchor_embeddings,
                            positive_embeddings,
                            negative_embeddings,
                        )
                    else:
                        raise ValueError(f"Unsupported batch format: {self.config.batch_format}")
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                global_step += 1
                running_loss.update(float(loss.detach().item()))
                throughput.update(anchor_embeddings.size(0), time.perf_counter() - start_time)

                if eval_fn is not None and global_step % self.config.eval_every_steps == 0:
                    latest_metrics = eval_fn(self.accelerator.unwrap_model(self.model))
                    dump_json(
                        self.run_dir / "metrics" / f"step-{global_step:05d}.json", latest_metrics
                    )

                if global_step % self.config.save_every_steps == 0:
                    checkpoint_path = self.save_checkpoint(
                        global_step=global_step,
                        metrics=latest_metrics,
                        config_snapshot=config_snapshot,
                    )
                    checkpoint_dir = str(checkpoint_path)

        if checkpoint_dir is None:
            checkpoint_path = self.save_checkpoint(
                global_step=global_step,
                metrics=latest_metrics,
                config_snapshot=config_snapshot,
            )
            checkpoint_dir = str(checkpoint_path)
        if eval_fn is not None and not latest_metrics:
            latest_metrics = eval_fn(self.accelerator.unwrap_model(self.model))
            dump_json(self.run_dir / "metrics" / "final.json", latest_metrics)

        return TrainingSummary(
            global_step=global_step,
            avg_loss=running_loss.value,
            throughput=throughput.per_second,
            checkpoint_dir=checkpoint_dir,
            metrics=latest_metrics,
        )

    def save_checkpoint(
        self,
        *,
        global_step: int,
        metrics: dict[str, float],
        config_snapshot: dict[str, Any] | None,
    ) -> Path:
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        checkpoint_dir = ensure_dir(self.run_dir / "checkpoints" / f"step-{global_step:05d}")
        model_path = checkpoint_dir / "model.pt"
        torch.save(unwrapped_model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        dump_json(checkpoint_dir / "metadata.json", self.metadata.to_dict())
        dump_json(checkpoint_dir / "metrics.json", metrics)
        if config_snapshot is not None:
            dump_json(checkpoint_dir / "config.json", config_snapshot)
        tokenizer = getattr(unwrapped_model, "tokenizer", None)
        if tokenizer is not None:
            tokenizer_dir = checkpoint_dir / "tokenizer"
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(tokenizer_dir)
            else:
                dump_json(
                    tokenizer_dir.with_suffix(".json"), {"type": tokenizer.__class__.__name__}
                )
        return checkpoint_dir
