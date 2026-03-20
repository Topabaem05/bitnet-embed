from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bitnet_embed.utils.io import ensure_dir, load_yaml


@dataclass(slots=True)
class BitNetCppConfig:
    binary_path: str = "bitnet_cpp/bitnet.cpp"
    model_path: str = "runs/packages/bitnet_smoke_hf"
    output_path: str = "bitnet_cpp/feasibility.json"
    prompt_mode: str = "embedding"


def load_bitnet_cpp_config(path: str) -> BitNetCppConfig:
    payload = load_yaml(path)
    return BitNetCppConfig(
        binary_path=str(payload.get("binary_path", "bitnet_cpp/bitnet.cpp")),
        model_path=str(payload.get("model_path", "runs/packages/bitnet_smoke_hf")),
        output_path=str(payload.get("output_path", "bitnet_cpp/feasibility.json")),
        prompt_mode=str(payload.get("prompt_mode", "embedding")),
    )


def build_feasibility_report(config: BitNetCppConfig) -> dict[str, Any]:
    binary_path = Path(config.binary_path)
    model_path = Path(config.model_path)
    binary_found = binary_path.exists()
    model_found = model_path.exists()
    ready_for_integration = binary_found and model_found
    next_steps = [
        "Provide a compiled bitnet.cpp binary or bridge entrypoint",
        "Map exported embedding package weights into the low-bit runtime path",
        "Validate embedding extraction parity against the Python runtime",
        "Benchmark latency, memory, and throughput on the specialized runtime",
    ]
    return {
        "status": "ready" if ready_for_integration else "scaffold",
        "binary_path": str(binary_path),
        "binary_found": binary_found,
        "model_path": str(model_path),
        "model_found": model_found,
        "prompt_mode": config.prompt_mode,
        "ready_for_integration": ready_for_integration,
        "next_steps": next_steps,
    }


def run_bitnet_cpp_feasibility(config_path: str) -> dict[str, Any]:
    config = load_bitnet_cpp_config(config_path)
    report = build_feasibility_report(config)
    output_path = Path(config.output_path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
