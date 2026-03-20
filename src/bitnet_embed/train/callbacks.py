from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from uuid import uuid4


@dataclass(slots=True)
class RunMetadata:
    run_id: str
    experiment_name: str
    seed: int
    mode: str
    git_revision: str | None
    created_at: str
    parent_run_id: str | None = None
    plan_name: str | None = None
    resume_from: str | None = None

    @classmethod
    def create(
        cls,
        experiment_name: str,
        seed: int,
        mode: str,
        git_revision: str | None,
        *,
        run_id: str | None = None,
        parent_run_id: str | None = None,
        plan_name: str | None = None,
        resume_from: str | None = None,
    ) -> RunMetadata:
        return cls(
            run_id=run_id or uuid4().hex,
            experiment_name=experiment_name,
            seed=seed,
            mode=mode,
            git_revision=git_revision,
            created_at=datetime.now(timezone.utc).isoformat(),
            parent_run_id=parent_run_id,
            plan_name=plan_name,
            resume_from=resume_from,
        )

    def to_dict(self) -> dict[str, str | int | None]:
        return asdict(self)
