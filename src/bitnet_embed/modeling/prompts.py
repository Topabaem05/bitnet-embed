from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

TaskType = Literal["query", "document"]


@dataclass(slots=True)
class PromptConfig:
    query_prefix: str = "query: "
    document_prefix: str = "document: "
    instruction_prefix: str = "instruction: "

    def prefix_for(self, task: TaskType) -> str:
        if task == "query":
            return self.query_prefix
        return self.document_prefix


def format_text(
    text: str,
    *,
    task: TaskType,
    prompt_config: PromptConfig | None = None,
    instruction: str | None = None,
) -> str:
    config = prompt_config or PromptConfig()
    normalized_text = text.strip()
    chunks: list[str] = []
    if instruction:
        chunks.append(f"{config.instruction_prefix}{instruction.strip()}")
    chunks.append(f"{config.prefix_for(task)}{normalized_text}")
    return "\n".join(chunks)


def format_batch(
    texts: Sequence[str],
    *,
    task: TaskType,
    prompt_config: PromptConfig | None = None,
    instruction: str | None = None,
) -> list[str]:
    return [
        format_text(text, task=task, prompt_config=prompt_config, instruction=instruction)
        for text in texts
    ]
