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
        f"# {resolved_package_name}",
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
