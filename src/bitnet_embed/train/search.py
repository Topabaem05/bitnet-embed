from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from bitnet_embed.train.external_runner import run_training_external
from bitnet_embed.train.trainer import TrainingSummary
from bitnet_embed.train.workflow import run_training
from bitnet_embed.utils.io import dump_json, ensure_dir, load_yaml


@dataclass(slots=True)
class TrialSpec:
    name: str
    overrides: dict[str, Any]


@dataclass(slots=True)
class RungSpec:
    name: str
    max_update_steps: int
    promote_top_k: int | None = None
    promote_fraction: float | None = None


@dataclass(slots=True)
class SearchSpec:
    search_name: str
    base_config: str
    output_root: str
    primary_metric: str
    maximize: bool
    executor: str
    trials: list[TrialSpec]
    rungs: list[RungSpec]


@dataclass(slots=True)
class ActiveTrial:
    trial: TrialSpec
    previous_checkpoint: str | None = None


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _sanitize_slug(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "-" for character in value
    )


def _load_trials(payload: dict[str, Any]) -> list[TrialSpec]:
    raw_trials = payload.get("trials", [])
    if not isinstance(raw_trials, list) or not raw_trials:
        raise RuntimeError("Search config must include at least one trial")
    trials: list[TrialSpec] = []
    for index, raw_trial in enumerate(raw_trials, start=1):
        if not isinstance(raw_trial, dict):
            raise RuntimeError("Each trial entry must be a mapping")
        name = str(raw_trial.get("name", f"trial_{index:02d}"))
        overrides = raw_trial.get("overrides", {})
        if not isinstance(overrides, dict):
            raise RuntimeError(f"Trial '{name}' overrides must be a mapping")
        trials.append(TrialSpec(name=name, overrides=overrides))
    return trials


def _load_rungs(payload: dict[str, Any]) -> list[RungSpec]:
    raw_rungs = payload.get("rungs", [])
    if not isinstance(raw_rungs, list) or not raw_rungs:
        raise RuntimeError("Search config must include at least one rung")

    rungs: list[RungSpec] = []
    previous_budget = 0
    for index, raw_rung in enumerate(raw_rungs, start=1):
        if not isinstance(raw_rung, dict):
            raise RuntimeError("Each rung entry must be a mapping")
        budget = int(raw_rung["max_update_steps"])
        if budget <= previous_budget:
            raise RuntimeError("Rung budgets must be strictly increasing")
        previous_budget = budget

        promote_top_k = raw_rung.get("promote_top_k")
        promote_fraction = raw_rung.get("promote_fraction")
        if promote_top_k is not None and promote_fraction is not None:
            raise RuntimeError("A rung can set promote_top_k or promote_fraction, not both")

        parsed_top_k: int | None = int(promote_top_k) if promote_top_k is not None else None
        parsed_fraction: float | None = (
            float(promote_fraction) if promote_fraction is not None else None
        )
        if parsed_top_k is not None and parsed_top_k <= 0:
            raise RuntimeError("promote_top_k must be greater than zero")
        if parsed_fraction is not None and not (0.0 < parsed_fraction <= 1.0):
            raise RuntimeError("promote_fraction must be in (0, 1]")

        rungs.append(
            RungSpec(
                name=str(raw_rung.get("name", f"rung_{index}")),
                max_update_steps=budget,
                promote_top_k=parsed_top_k,
                promote_fraction=parsed_fraction,
            )
        )
    return rungs


def load_search_spec(config_path: str) -> SearchSpec:
    payload = load_yaml(config_path)
    base_config = payload.get("base_config")
    primary_metric = payload.get("primary_metric")
    if base_config is None:
        raise RuntimeError("Search config requires base_config")
    if primary_metric is None:
        raise RuntimeError("Search config requires primary_metric")

    search_name = str(payload.get("search_name", Path(config_path).stem))
    output_root = str(payload.get("output_root", f"runs/search/{search_name}"))
    executor = str(payload.get("executor", "in_process"))
    if executor not in {"in_process", "external"}:
        raise RuntimeError("Search executor must be one of: in_process, external")
    return SearchSpec(
        search_name=search_name,
        base_config=str(base_config),
        output_root=output_root,
        primary_metric=str(primary_metric),
        maximize=bool(payload.get("maximize", False)),
        executor=executor,
        trials=_load_trials(payload),
        rungs=_load_rungs(payload),
    )


def _resolve_promotions(rung: RungSpec, trial_count: int) -> int:
    if trial_count <= 1:
        return trial_count
    if rung.promote_top_k is not None:
        return max(1, min(trial_count, rung.promote_top_k))
    if rung.promote_fraction is not None:
        return max(1, min(trial_count, int(trial_count * rung.promote_fraction)))
    return max(1, trial_count // 2)


def _metric_value(summary: TrainingSummary, metric_name: str) -> float:
    if metric_name == "avg_loss":
        return summary.avg_loss
    if metric_name == "throughput":
        return summary.throughput
    if metric_name == "global_step":
        return float(summary.global_step)
    value = summary.metrics.get(metric_name)
    if isinstance(value, (int, float)):
        return float(value)
    raise RuntimeError(f"Primary metric '{metric_name}' was not found in run metrics")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def build_search_markdown(summary: dict[str, Any]) -> str:
    lines = [f"# {summary['search_name']}", "", "## Overview", ""]
    lines.append(f"- `search_run_id`: `{summary['search_run_id']}`")
    lines.append(f"- `base_config`: `{summary['base_config']}`")
    lines.append(f"- `primary_metric`: `{summary['primary_metric']}`")
    lines.append(f"- `maximize`: `{summary['maximize']}`")
    lines.append(f"- `trial_count`: `{summary['trial_count']}`")
    lines.append(f"- `rung_count`: `{summary['rung_count']}`")
    best_trial = summary.get("best_trial")
    if isinstance(best_trial, dict):
        lines.append(f"- `best_trial`: `{best_trial.get('trial_name')}`")
        lines.append(f"- `best_metric`: `{best_trial.get('metric_value')}`")
    lines.append("")

    lines.append("## Rungs")
    lines.append("")
    for rung in summary.get("rungs", []):
        if not isinstance(rung, dict):
            continue
        lines.append(f"### {rung['rung_name']}")
        lines.append(f"- budget: `{rung['max_update_steps']}`")
        lines.append(f"- promoted: `{', '.join(rung.get('promoted_trials', [])) or 'none'}`")
        lines.append(f"- finalists: `{', '.join(rung.get('finalists', [])) or 'none'}`")
        lines.append(f"- discarded: `{', '.join(rung.get('discarded_trials', [])) or 'none'}`")
        lines.append("")
        for ranked in rung.get("ranked_trials", []):
            if not isinstance(ranked, dict):
                continue
            lines.append(
                "- "
                f"rank `{ranked['rank']}`: `{ranked['trial_name']}` "
                f"metric=`{ranked['metric_value']}` run_id=`{ranked['run_id']}`"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def run_search(config_path: str) -> dict[str, Any]:
    spec = load_search_spec(config_path)
    base_config = load_yaml(spec.base_config)
    search_run_id = f"{_sanitize_slug(spec.search_name)}-{uuid4().hex[:12]}"

    output_dir = ensure_dir(spec.output_root)
    configs_dir = ensure_dir(output_dir / "configs")

    active_trials = [ActiveTrial(trial=trial) for trial in spec.trials]
    rung_summaries: list[dict[str, Any]] = []
    final_ranked_trials: list[dict[str, Any]] = []

    for rung_index, rung in enumerate(spec.rungs, start=1):
        rung_results: list[dict[str, Any]] = []
        for trial_state in active_trials:
            merged_config = _deep_merge(base_config, trial_state.trial.overrides)
            training_payload = merged_config.setdefault("training", {})
            if not isinstance(training_payload, dict):
                raise RuntimeError("training config must be a mapping")

            base_experiment_name = str(merged_config.get("experiment_name", "bitnet_search"))
            trial_slug = _sanitize_slug(trial_state.trial.name)
            merged_config["experiment_name"] = (
                f"{base_experiment_name}_{_sanitize_slug(spec.search_name)}_{trial_slug}_r{rung_index}"
            )
            training_payload["max_update_steps"] = rung.max_update_steps
            training_payload["run_id"] = f"{search_run_id}:{trial_slug}:r{rung_index}"
            if trial_state.previous_checkpoint is not None:
                training_payload["resume_from_checkpoint"] = trial_state.previous_checkpoint

            trial_config_path = configs_dir / f"r{rung_index:02d}_{trial_slug}.yaml"
            _write_yaml(trial_config_path, merged_config)

            if spec.executor == "external":
                training_summary = run_training_external(
                    str(trial_config_path),
                    plan_name=spec.search_name,
                    parent_run_id=search_run_id,
                )
            else:
                training_summary = run_training(
                    str(trial_config_path),
                    plan_name=spec.search_name,
                    parent_run_id=search_run_id,
                )
            metric_value = _metric_value(training_summary, spec.primary_metric)
            rung_results.append(
                {
                    "trial_name": trial_state.trial.name,
                    "config_path": str(trial_config_path),
                    "resume_from_checkpoint": trial_state.previous_checkpoint,
                    "run_id": training_summary.run_id,
                    "checkpoint_dir": training_summary.checkpoint_dir,
                    "global_step": training_summary.global_step,
                    "avg_loss": training_summary.avg_loss,
                    "throughput": training_summary.throughput,
                    "metrics": training_summary.metrics,
                    "metric_value": metric_value,
                }
            )

        ranked_trials = sorted(
            rung_results,
            key=lambda item: (
                -float(item["metric_value"]) if spec.maximize else float(item["metric_value"]),
                str(item["trial_name"]),
            ),
        )
        for rank, ranked in enumerate(ranked_trials, start=1):
            ranked["rank"] = rank

        has_next_rung = rung_index < len(spec.rungs)
        promote_count = _resolve_promotions(rung, len(ranked_trials)) if has_next_rung else 0
        promoted = ranked_trials[:promote_count]
        promoted_names = [str(item["trial_name"]) for item in promoted]
        finalist_names = (
            [str(item["trial_name"]) for item in ranked_trials] if not has_next_rung else []
        )
        discarded_names = (
            [
                str(item["trial_name"])
                for item in ranked_trials
                if item["trial_name"] not in promoted_names
            ]
            if has_next_rung
            else []
        )

        rung_summaries.append(
            {
                "rung_name": rung.name,
                "rung_index": rung_index,
                "max_update_steps": rung.max_update_steps,
                "promote_top_k": rung.promote_top_k,
                "promote_fraction": rung.promote_fraction,
                "ranked_trials": ranked_trials,
                "promoted_trials": promoted_names,
                "finalists": finalist_names,
                "discarded_trials": discarded_names,
            }
        )

        if not has_next_rung:
            final_ranked_trials = ranked_trials
            break

        checkpoint_by_name = {
            str(item["trial_name"]): (
                str(item["checkpoint_dir"]) if item["checkpoint_dir"] else None
            )
            for item in ranked_trials
        }
        trial_by_name = {trial_state.trial.name: trial_state.trial for trial_state in active_trials}
        active_trials = [
            ActiveTrial(
                trial=trial_by_name[trial_name], previous_checkpoint=checkpoint_by_name[trial_name]
            )
            for trial_name in promoted_names
            if trial_name in trial_by_name
        ]

    best_trial = final_ranked_trials[0] if final_ranked_trials else None
    summary: dict[str, Any] = {
        "search_name": spec.search_name,
        "search_run_id": search_run_id,
        "config_path": config_path,
        "output_root": str(output_dir),
        "base_config": spec.base_config,
        "primary_metric": spec.primary_metric,
        "maximize": spec.maximize,
        "executor": spec.executor,
        "trial_count": len(spec.trials),
        "rung_count": len(spec.rungs),
        "rungs": rung_summaries,
        "best_trial": best_trial,
    }

    dump_json(output_dir / "search_summary.json", summary)
    (output_dir / "search_summary.md").write_text(build_search_markdown(summary), encoding="utf-8")
    return summary
