from __future__ import annotations

import sys
from types import SimpleNamespace

from _pytest.monkeypatch import MonkeyPatch

from bitnet_embed.eval.mteb_runner import run_mteb


def test_run_mteb_uses_get_tasks_with_current_api(monkeypatch: MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeEvaluation:
        def __init__(self, *, tasks: object) -> None:
            captured["evaluation_tasks"] = tasks

        def run(self, model: object, output_folder: str | None = None) -> dict[str, object]:
            captured["model"] = model
            captured["output_folder"] = output_folder
            return {"ok": True}

    def _get_tasks(*, tasks: list[str]) -> list[str]:
        captured["requested_task_names"] = tasks
        return [f"resolved::{name}" for name in tasks]

    fake_mteb = SimpleNamespace(MTEB=_FakeEvaluation, get_tasks=_get_tasks)
    monkeypatch.setitem(sys.modules, "mteb", fake_mteb)

    wrapped = SimpleNamespace(mteb_model_meta=True)
    result = run_mteb(wrapped, ["STSBenchmark"], output_folder="/tmp/results")

    assert result == {"ok": True}
    assert captured["requested_task_names"] == ["STSBenchmark"]
    assert captured["evaluation_tasks"] == ["resolved::STSBenchmark"]
    assert captured["model"] is wrapped
    assert captured["output_folder"] == "/tmp/results"
