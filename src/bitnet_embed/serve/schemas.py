from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    task: Literal["query", "document"] = "document"
    normalize: bool | None = None
    truncate_dim: int | None = Field(default=None, ge=1)

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, list) and len(value) == 0:
            raise ValueError("input must not be empty")
        if isinstance(value, str) and not value.strip():
            raise ValueError("input must not be blank")
        return value

    def to_inputs(self) -> list[str]:
        if isinstance(self.input, str):
            return [self.input]
        return self.input


class EmbeddingData(BaseModel):
    index: int
    embedding: list[float]


class UsageInfo(BaseModel):
    input_texts: int
    tokens: int


class EmbeddingResponse(BaseModel):
    model: str
    data: list[EmbeddingData]
    usage: UsageInfo
