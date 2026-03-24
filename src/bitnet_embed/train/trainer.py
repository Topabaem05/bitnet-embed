from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from bitnet_embed.train.callbacks import RunMetadata
from bitnet_embed.train.factory import load_checkpoint_weights, load_training_checkpoint_state
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
    fp16: bool = False
    log_every_steps: int = 10
    eval_every_steps: int = 50
    save_every_steps: int = 100
    max_update_steps: int | None = None
    max_grad_norm: float | None = 1.0
    max_length: int = 256
    run_root: str = "runs"
    seed: int = 42
    batch_format: str = "pair"
    run_id: str | None = None
    parent_run_id: str | None = None
    plan_name: str | None = None
    resume_from_checkpoint: str | None = None
    resume_weights_only: bool = False


@dataclass(slots=True)
class TrainingSummary:
    run_id: str
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
    mixed_precision = "no"
    if torch.cuda.is_available():
        if config.fp16:
            mixed_precision = "fp16"
        elif config.bf16:
            mixed_precision = "bf16"
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
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self.run_dir = self._build_run_dir()
        self.metadata = RunMetadata.create(
            experiment_name=config.experiment_name,
            seed=config.seed,
            mode=config.mode,
            git_revision=get_git_revision(Path.cwd()),
            run_id=config.run_id,
            parent_run_id=config.parent_run_id,
            plan_name=config.plan_name,
            resume_from=config.resume_from_checkpoint,
        )

    def _log_progress(self, message: str) -> None:
        if bool(getattr(self.accelerator, "is_local_main_process", True)):
            print(message, flush=True)

    def _cuda_device(self) -> torch.device | None:
        device = getattr(self.accelerator, "device", None)
        if isinstance(device, torch.device) and device.type == "cuda":
            return device
        return None

    def _log_cuda_memory(self) -> None:
        device = self._cuda_device()
        if device is None:
            return
        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        self._log_progress(f"allocated: {allocated:.2f} GB")
        self._log_progress(f"reserved: {reserved:.2f} GB")
        self._log_progress(f"peak allocated: {peak_allocated:.2f} GB")
        self._log_progress(f"peak reserved: {peak_reserved:.2f} GB")

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
        total_steps = self._resolve_total_steps(train_dataloader)
        self.optimizer = build_optimizer(
            self.model,
            OptimizerConfig(
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                warmup_ratio=self.config.warmup_ratio,
                total_steps=total_steps,
            ),
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            OptimizerConfig(
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                warmup_ratio=self.config.warmup_ratio,
                total_steps=total_steps,
            ),
        )
        self.model, self.optimizer, train_dataloader, self.scheduler = self.accelerator.prepare(
            self.model,
            self.optimizer,
            train_dataloader,
            self.scheduler,
        )
        device = self._cuda_device()
        if device is not None:
            torch.cuda.reset_peak_memory_stats(device)
        if self.optimizer is None or self.scheduler is None:
            raise RuntimeError("Optimizer and scheduler must be initialized before training")
        global_step = 0
        if self.config.resume_from_checkpoint:
            global_step = self._resume_from_checkpoint(Path(self.config.resume_from_checkpoint))
        running_loss = RunningAverage()
        throughput = ThroughputMeter()
        latest_metrics: dict[str, float] = {}
        checkpoint_dir: str | None = None

        self._log_progress(
            f"[train] start experiment={self.config.experiment_name} mode={self.config.mode} "
            f"target_steps={total_steps}"
        )

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
                    if self.accelerator.sync_gradients:
                        if self.config.max_grad_norm is not None:
                            self.accelerator.clip_grad_norm_(
                                self.model.parameters(), self.config.max_grad_norm
                            )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

                running_loss.update(float(loss.detach().item()))
                throughput.update(anchor_embeddings.size(0), time.perf_counter() - start_time)

                if not self.accelerator.sync_gradients:
                    continue

                global_step += 1

                should_log_progress = (
                    global_step == 1
                    or global_step <= 3
                    or global_step == total_steps
                    or global_step % self.config.log_every_steps == 0
                )
                if should_log_progress:
                    percent = (global_step / max(1, total_steps)) * 100.0
                    self._log_progress(
                        "[train] progress "
                        f"step={global_step}/{total_steps} "
                        f"percent={percent:.1f} "
                        f"avg_loss={running_loss.value:.6f} "
                        f"throughput={throughput.per_second:.2f}"
                    )
                    self._log_cuda_memory()

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

                if (
                    self.config.max_update_steps is not None
                    and global_step >= self.config.max_update_steps
                ):
                    break
            if (
                self.config.max_update_steps is not None
                and global_step >= self.config.max_update_steps
            ):
                break

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

        self._log_progress(
            f"[train] complete experiment={self.config.experiment_name} "
            f"steps={global_step} checkpoint={checkpoint_dir}"
        )

        return TrainingSummary(
            run_id=self.metadata.run_id,
            global_step=global_step,
            avg_loss=running_loss.value,
            throughput=throughput.per_second,
            checkpoint_dir=checkpoint_dir,
            metrics=latest_metrics,
        )

    def _resume_from_checkpoint(self, checkpoint_dir: Path) -> int:
        if self.optimizer is None or self.scheduler is None:
            raise RuntimeError("Optimizer and scheduler must be initialized before resume")
        checkpoint_state = load_training_checkpoint_state(checkpoint_dir)
        if self.config.resume_weights_only:
            load_checkpoint_weights(self.accelerator.unwrap_model(self.model), checkpoint_state)
            self.optimizer.zero_grad(set_to_none=True)
            return 0
        self.model.load_state_dict(checkpoint_state.model_state)
        self.optimizer.load_state_dict(checkpoint_state.optimizer_state)
        self.scheduler.load_state_dict(checkpoint_state.scheduler_state)
        return checkpoint_state.global_step

    def _resolve_total_steps(self, train_dataloader: DataLoader[Any]) -> int:
        if self.config.max_update_steps is not None:
            return max(1, self.config.max_update_steps)

        inferred_steps = self._infer_total_update_steps(train_dataloader)
        if inferred_steps is not None:
            return inferred_steps

        raise RuntimeError(
            "Unable to infer optimizer update steps from training data. "
            "Set training.max_update_steps when using unknown-length datasets."
        )

    def _infer_total_update_steps(self, train_dataloader: DataLoader[Any]) -> int | None:
        try:
            micro_steps_per_epoch = len(train_dataloader)
        except TypeError:
            return None
        updates_per_epoch = ceil(micro_steps_per_epoch / max(1, self.config.grad_accum_steps))
        return max(1, updates_per_epoch * self.config.epochs)

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
        if self.optimizer is None or self.scheduler is None:
            raise RuntimeError("Optimizer and scheduler must be initialized before checkpointing")
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        dump_json(checkpoint_dir / "metadata.json", self.metadata.to_dict())
        dump_json(checkpoint_dir / "training_state.json", {"global_step": global_step})
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
