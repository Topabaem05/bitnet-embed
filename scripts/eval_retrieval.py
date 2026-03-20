from __future__ import annotations

import json

from bitnet_embed.eval.retrieval import evaluate_retrieval
from bitnet_embed.modeling.model import EncodeConfig
from bitnet_embed.modeling.smoke import build_toy_embedding_model


def main() -> None:
    model = build_toy_embedding_model()
    queries = ["query: fast car", "query: ocean waves"]
    documents = ["document: quick automobile", "document: sea surf", "document: stock report"]
    query_embeddings = model.encode(queries, EncodeConfig(task="query", batch_size=2))
    doc_embeddings = model.encode(documents, EncodeConfig(task="document", batch_size=3))
    metrics = evaluate_retrieval(query_embeddings, doc_embeddings, {0: {0}, 1: {1}})
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
