from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any, Final

from bitnet_embed.train.workflow import run_training

_UNSET_RESUME: Final = "__BITNET_EMBED_UNSET_RESUME__"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode-override")
    parser.add_argument("--plan-name")
    parser.add_argument("--parent-run-id")
    parser.add_argument("--resume-from-checkpoint", default=_UNSET_RESUME)
    args = parser.parse_args()

    kwargs: dict[str, Any] = {}
    if args.mode_override is not None:
        kwargs["mode_override"] = args.mode_override
    if args.plan_name is not None:
        kwargs["plan_name"] = args.plan_name
    if args.parent_run_id is not None:
        kwargs["parent_run_id"] = args.parent_run_id
    if args.resume_from_checkpoint != _UNSET_RESUME:
        kwargs["resume_from_checkpoint"] = args.resume_from_checkpoint or None

    summary = run_training(args.config, **kwargs)
    print(json.dumps(asdict(summary), sort_keys=True))


if __name__ == "__main__":
    main()
