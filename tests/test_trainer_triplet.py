from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from bitnet_embed.data.collators import TripletCollator
from bitnet_embed.data.loaders import ExampleDataset, build_smoke_triplets
from bitnet_embed.data.schemas import TripletExample
from bitnet_embed.losses.triplet import CosineTripletLoss
from bitnet_embed.modeling.prompts import PromptConfig
from bitnet_embed.modeling.smoke import build_toy_embedding_model
from bitnet_embed.train.trainer import EmbeddingTrainer, TrainingConfig


def test_trainer_runs_triplet_epoch_and_writes_checkpoint(tmp_path: Path) -> None:
    model = build_toy_embedding_model(projection_dim=8)
    assert model.tokenizer is not None
    collator = TripletCollator(model.tokenizer, max_length=16, prompt_config=PromptConfig())
    dataset: ExampleDataset[TripletExample] = ExampleDataset(build_smoke_triplets())
    dataloader: DataLoader[TripletExample] = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collator,
    )
    trainer = EmbeddingTrainer(
        model,
        CosineTripletLoss(),
        TrainingConfig(
            experiment_name="trainer_triplet_smoke",
            epochs=1,
            micro_batch_size=2,
            batch_format="triplet",
            save_every_steps=1,
            eval_every_steps=10,
            run_root=str(tmp_path),
        ),
    )
    summary = trainer.train(
        dataloader,
        config_snapshot={"experiment_name": "trainer_triplet_smoke"},
    )
    assert summary.global_step == 1
    assert summary.checkpoint_dir is not None
