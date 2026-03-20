from __future__ import annotations

from torch.utils.data import IterableDataset

from bitnet_embed.data.loaders import IterableExampleDataset
from bitnet_embed.train.workflow import build_train_dataset, build_training_config


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
