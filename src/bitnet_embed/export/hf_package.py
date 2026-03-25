from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch

from bitnet_embed.modeling.backbone import BitNetBackbone
from bitnet_embed.train.factory import build_model, load_model_checkpoint
from bitnet_embed.utils.io import dump_json, ensure_dir, load_json


def build_package_manifest(
    checkpoint_dir: Path,
    package_name: str,
    config_snapshot: dict[str, Any],
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "package_name": package_name,
        "format": "bitnet-embed-hf-style",
        "checkpoint_dir": str(checkpoint_dir),
        "model": config_snapshot.get("model", {}),
        "lora": config_snapshot.get("lora"),
        "metrics": metrics,
        "metadata": metadata,
    }


def export_hf_package(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    package_name: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_dir)
    output_path = ensure_dir(output_dir)
    config_snapshot = load_json(checkpoint_path / "config.json")
    metrics = load_json(checkpoint_path / "metrics.json")
    metadata = load_json(checkpoint_path / "metadata.json")
    run_root = checkpoint_path.parent.parent
    final_metrics_path = run_root / "metrics" / "final.json"
    summary_metrics_path = run_root / "artifacts" / "summary.json"
    if not metrics and final_metrics_path.exists():
        metrics = load_json(final_metrics_path)
    if not metrics and summary_metrics_path.exists():
        metrics = load_json(summary_metrics_path)
    resolved_package_name = package_name or str(
        config_snapshot.get("experiment_name", output_path.name)
    )
    model = load_model_checkpoint(checkpoint_path)
    torch.save(model.state_dict(), output_path / "pytorch_model.bin")
    manifest = build_package_manifest(
        checkpoint_path,
        resolved_package_name,
        config_snapshot,
        metrics,
        metadata,
    )
    dump_json(output_path / "config.json", manifest)
    dump_json(output_path / "training_config.json", config_snapshot)
    dump_json(output_path / "metrics.json", metrics)
    dump_json(output_path / "metadata.json", metadata)
    tokenizer_dir = checkpoint_path / "tokenizer"
    if tokenizer_dir.exists():
        shutil.rmtree(output_path / "tokenizer", ignore_errors=True)
        shutil.copytree(tokenizer_dir, output_path / "tokenizer")
    else:
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(output_path / "tokenizer")
        else:
            dump_json(
                output_path / "tokenizer.json",
                {"type": tokenizer.__class__.__name__ if tokenizer is not None else "unknown"},
            )
    readme_lines = [
        "---",
        "license: mit",
        "base_model: microsoft/bitnet-b1.58-2B-4T-bf16",
        "library_name: transformers",
        "pipeline_tag: feature-extraction",
        "tags:",
        "- bitnet-embed",
        "- embeddings",
        "- feature-extraction",
        "- pytorch",
        "language:",
        "- en",
        "- ko",
        "---",
        "",
        f"# {resolved_package_name}",
        "",
        "## Overview",
        "",
        "This is a `bitnet-embed-hf-style` export produced by the `bitnet-embed` training stack.",
        "It is intended to be loaded with `bitnet_embed.export.load_hf_package()`.",
        "",
        "## Export Summary",
        "",
        f"- checkpoint: `{checkpoint_path}`",
        f"- model artifact: `{output_path / 'pytorch_model.bin'}`",
    ]
    if metrics:
        readme_lines.append("- metrics:")
        for key, value in sorted(metrics.items()):
            readme_lines.append(f"  - `{key}`: `{value}`")
    readme_lines.extend(
        [
            "",
            "## Usage",
            "",
            "Load this package with `bitnet_embed.export.load_hf_package()` rather than",
            "`transformers.AutoModel.from_pretrained()`.",
            "",
            "## 한국어 안내",
            "",
            "- 이 모델은 일반 `transformers` 모델처럼 바로 여는 형식이 아닙니다.",
            "- `bitnet_embed.export.load_hf_package()`로 패키지를 먼저 읽어야 합니다.",
            "- 검색 용도면 `task='query'`와 `task='document'`를 나눠서 임베딩하세요.",
            "",
            "### Python: local package path",
            "",
            "```python",
            "from bitnet_embed.export import load_hf_package",
            "from bitnet_embed.modeling.model import EncodeConfig",
            "",
            f"model = load_hf_package('{output_path}')",
            "embeddings = model.encode(",
            "    ['What is BitNet?', 'How do I train embeddings?'],",
            "    EncodeConfig(batch_size=2, task='query', normalize=True, truncate_dim=768),",
            ")",
            "print(embeddings.shape)",
            "```",
            "",
            "### Python: download from Hugging Face Hub first",
            "",
            "```python",
            "from huggingface_hub import snapshot_download",
            "from bitnet_embed.export import load_hf_package",
            "from bitnet_embed.modeling.model import EncodeConfig",
            "",
            "package_dir = snapshot_download('topabaem/bitnet-embed')",
            "model = load_hf_package(package_dir)",
            "embeddings = model.encode(",
            "    ['retrieve this passage', 'another query'],",
            "    EncodeConfig(batch_size=2, task='query', normalize=True, truncate_dim=768),",
            ")",
            "print(embeddings.shape)",
            "```",
            "",
            "### Query / document example",
            "",
            "```python",
            "query_vectors = model.encode(",
            "    ['neural search with bitnet'],",
            "    EncodeConfig(task='query', normalize=True, truncate_dim=768),",
            ")",
            "doc_vectors = model.encode(",
            "    ['BitNet can be adapted into an embedding encoder with staged training.'],",
            "    EncodeConfig(task='document', normalize=True, truncate_dim=768),",
            ")",
            "score = (query_vectors @ doc_vectors.T).item()",
            "print(score)",
            "```",
            "",
            "## API Serving",
            "",
            "Use the original checkpoint directory for serving inside this repository.",
            "",
            "### Service config example",
            "",
            "```yaml",
            "model_name: bitnet-embed",
            "host: 0.0.0.0",
            "port: 8000",
            "normalize_default: true",
            "truncate_dim_default: 768",
            "backend: checkpoint",
            f"checkpoint_dir: {checkpoint_path}",
            "```",
            "",
            "### Start the API",
            "",
            "```bash",
            "uv run python scripts/run_api.py --config configs/service/api.yaml",
            "```",
            "",
            "### curl example",
            "",
            "```bash",
            "curl -X POST http://127.0.0.1:8000/v1/embeddings \\",
            "  -H 'Content-Type: application/json' \\",
            "  -d '{",
            '    "input": ["neural search with bitnet", "dense retrieval"],',
            '    "task": "query",',
            '    "normalize": true,',
            '    "truncate_dim": 768',
            "  }'",
            "```",
            "",
            "The response shape is `{ model, data, usage }`, where each `data` item has an",
            "`index` and an `embedding` vector.",
            "",
            "### Notes",
            "",
            "- The package contains `training_config.json`, `metadata.json`,",
            "  tokenizer files, and `pytorch_model.bin`.",
            "- For API serving in this repo, use the original checkpoint",
            "  directory rather than this exported package.",
        ]
    )
    (output_path / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    return manifest


def load_hf_package(package_dir: str | Path) -> Any:
    package_path = Path(package_dir)
    config_snapshot = load_json(package_path / "training_config.json")
    model = build_model(config_snapshot.get("model", {}), config_snapshot.get("lora"))
    state_dict = torch.load(
        package_path / "pytorch_model.bin", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict)
    tokenizer_path = package_path / "tokenizer"
    if tokenizer_path.exists() and isinstance(model.backbone, BitNetBackbone):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required to load a packaged tokenizer") from exc
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model.backbone.tokenizer = tokenizer
        model.tokenizer = tokenizer
    model.eval()
    return model
