from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone


@dataclass(slots=True)
class RunMetadata:
    experiment_name: str
    seed: int
    mode: str
    git_revision: str | None
    created_at: str

    @classmethod
    def create(
        cls, experiment_name: str, seed: int, mode: str, git_revision: str | None
    ) -> RunMetadata:
        return cls(
            experiment_name=experiment_name,
            seed=seed,
            mode=mode,
            git_revision=git_revision,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, str | int | None]:
        return asdict(self)
