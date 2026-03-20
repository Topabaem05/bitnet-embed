from __future__ import annotations

import argparse
import json

from bitnet_embed.data.loaders import build_dataset_spec, load_examples
from bitnet_embed.data.schemas import QueryDocumentExample
from bitnet_embed.eval.harness import evaluate_query_documents
from bitnet_embed.modeling.smoke import build_toy_embedding_model
from bitnet_embed.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/smoke_retrieval.yaml")
    args = parser.parse_args()

    data_config = load_yaml(args.config)
    eval_payload = data_config["eval_sets"][0]
    examples = [
        item
        for item in load_examples(build_dataset_spec(eval_payload))
        if isinstance(item, QueryDocumentExample)
    ]
    model = build_toy_embedding_model()
    metrics = evaluate_query_documents(model, examples)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
