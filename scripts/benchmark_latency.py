from __future__ import annotations

import json

from bitnet_embed.eval.benchmark import measure_latency
from bitnet_embed.modeling.model import EncodeConfig
from bitnet_embed.modeling.smoke import build_toy_embedding_model


def main() -> None:
    model = build_toy_embedding_model()
    batches = [["a quick fox"], ["bright ocean", "quiet forest"], ["lorem ipsum"]]
    encode_config = EncodeConfig(batch_size=4)
    metrics = measure_latency(lambda texts: model.encode(texts, encode_config), batches)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
