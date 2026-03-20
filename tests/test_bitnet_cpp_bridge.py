from __future__ import annotations

from pathlib import Path

from bitnet_embed.bitnet_cpp_bridge import run_bitnet_cpp_feasibility


def test_run_bitnet_cpp_feasibility_writes_report(tmp_path: Path) -> None:
    binary_path = tmp_path / "bitnet.cpp"
    model_path = tmp_path / "package"
    model_path.mkdir()
    config_path = tmp_path / "bitnet_cpp.yaml"
    output_path = tmp_path / "feasibility.json"
    config_path.write_text(
        "\n".join(
            [
                f"binary_path: {binary_path}",
                f"model_path: {model_path}",
                f"output_path: {output_path}",
                "prompt_mode: embedding",
            ]
        ),
        encoding="utf-8",
    )
    report = run_bitnet_cpp_feasibility(str(config_path))
    assert output_path.exists()
    assert report["binary_found"] is False
    assert report["model_found"] is True
