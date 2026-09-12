from __future__ import annotations

import csv
import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from safety_gap.api_evaluation.client import AbliterationClient, CompletionResult, Message
from safety_gap.api_evaluation.config import (
    ApiExperimentConfig,
    GenerationConfig,
)
from safety_gap.api_evaluation.datasets import (
    CapabilitySample,
    PropensitySample,
    choice_letters,
    load_capability_samples,
    load_propensity_samples,
)
from safety_gap.api_evaluation.metrics import compute_summary
from safety_gap.api_evaluation.scoring import (
    capability_messages,
    parse_choice_letter,
    parse_strongreject,
    strongreject_messages,
)
from safety_gap.api_evaluation.transcript import TranscriptWriter, json_value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _generated_run_id(run_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{run_name}_{timestamp}"


def _completion_columns(result: CompletionResult, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_request_payload_json": json_value(result.request_payload),
        f"{prefix}_answer": result.content,
        f"{prefix}_reasoning": result.reasoning_content,
        f"{prefix}_annotations_json": json_value(result.annotations),
        f"{prefix}_raw_response_json": result.raw_response_json(),
        f"{prefix}_success": result.success,
        f"{prefix}_error_type": result.error_type or "",
        f"{prefix}_error_message": result.error_message or "",
        f"{prefix}_status_code": result.status_code,
        f"{prefix}_response_id": result.response_id or "",
        f"{prefix}_returned_model": result.returned_model or "",
        f"{prefix}_finish_reason": result.finish_reason or "",
        f"{prefix}_latency_seconds": result.latency_seconds,
        f"{prefix}_attempts": result.attempts,
        f"{prefix}_retry_delays_seconds_json": json_value(
            result.retry_delays_seconds
        ),
        f"{prefix}_prompt_tokens": result.prompt_tokens,
        f"{prefix}_cached_input_tokens": result.cached_input_tokens,
        f"{prefix}_completion_tokens": result.completion_tokens,
        f"{prefix}_reasoning_tokens": result.reasoning_tokens,
        f"{prefix}_total_tokens": result.total_tokens,
        f"{prefix}_estimated_cost_usd": result.estimated_cost_usd,
        f"{prefix}_recalculated_cost_usd": result.recalculated_cost_usd,
        f"{prefix}_estimated_credits_used": result.estimated_credits_used,
        f"{prefix}_remaining_credits": result.remaining_credits,
    }


def _messages_json(messages: Sequence[Message]) -> str:
    return json_value(list(messages))


def _preferred_cost(result: CompletionResult) -> float:
    if result.estimated_cost_usd is not None:
        return result.estimated_cost_usd
    return result.recalculated_cost_usd or 0.0


class CostLimitReached(RuntimeError):
    pass


def _base_row(
    run_id: str,
    task_type: str,
    dataset_name: str,
    sample_id: str,
    question: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp_utc": _utc_now(),
        "task_type": task_type,
        "dataset_name": dataset_name,
        "sample_id": sample_id,
        "question": question,
    }


def _capability_row(
    run_id: str,
    dataset_name: str,
    sample: CapabilitySample,
    messages: Sequence[Message],
    result: CompletionResult,
) -> dict[str, Any]:
    letters = choice_letters(sample.choices)
    correct_letter = letters[sample.answer_index]
    parsed_letter = parse_choice_letter(result.content, letters) if result.success else None
    row = _base_row(
        run_id,
        "capability",
        dataset_name,
        sample.sample_id,
        sample.question,
    )
    row.update(
        {
            "choices_json": json_value(sample.choices),
            "correct_answer_letter": correct_letter,
            "target_messages_json": _messages_json(messages),
            "parsed_answer_letter": parsed_letter or "",
            "capability_correct": parsed_letter == correct_letter,
        }
    )
    row.update(_completion_columns(result, "target"))
    return row


def _propensity_target_row(
    config: ApiExperimentConfig,
    run_id: str,
    sample: PropensitySample,
    target_messages: Sequence[Message],
    target_result: CompletionResult,
) -> dict[str, Any]:
    row = _base_row(
        run_id,
        "propensity",
        config.propensity_dataset.name,
        sample.sample_id,
        sample.question,
    )
    row.update(
        {
            "unsafe_probability": sample.unsafe_probability,
            "target_messages_json": _messages_json(target_messages),
        }
    )
    row.update(_completion_columns(target_result, "target"))
    return row


def _add_judgment(
    config: ApiExperimentConfig,
    row: dict[str, Any],
    judge_messages: Sequence[Message],
    judge_result: CompletionResult,
) -> None:
    row["judge_messages_json"] = _messages_json(judge_messages)
    row.update(_completion_columns(judge_result, "judge"))
    if not judge_result.success:
        return

    try:
        judgment = parse_strongreject(
            judge_result.content,
            config.scoring.useful_score_threshold,
        )
    except ValueError as exc:
        row["judge_parse_error"] = str(exc)
        return

    row.update(
        {
            "strongreject_refusal": judgment.refusal,
            "strongreject_convincingness": judgment.convincingness,
            "strongreject_specificity": judgment.specificity,
            "strongreject_score": judgment.score,
            "strongreject_useful_at_threshold": judgment.useful,
        }
    )


class EventWriter:
    """Thread-safe, append-only record of every API result, including failures."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()

    def write(
        self,
        *,
        task_type: str,
        sample_id: str,
        phase: str,
        result: CompletionResult,
    ) -> None:
        event = {
            "timestamp_utc": _utc_now(),
            "task_type": task_type,
            "sample_id": sample_id,
            "phase": phase,
            "success": result.success,
            "status_code": result.status_code,
            "error_type": result.error_type,
            "error_message": result.error_message,
            "latency_seconds": result.latency_seconds,
            "attempts": result.attempts,
            "retry_delays_seconds": result.retry_delays_seconds,
            "estimated_cost_usd": result.estimated_cost_usd,
            "recalculated_cost_usd": result.recalculated_cost_usd,
            "request_payload": result.request_payload,
            "raw_response": result.raw_response,
        }
        serialized = json.dumps(event, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._handle.write(serialized + "\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()
                os.fsync(self._handle.fileno())
                self._handle.close()

    def __enter__(self) -> "EventWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


_worker_state = threading.local()


def _worker_client(config: ApiExperimentConfig) -> AbliterationClient:
    client = getattr(_worker_state, "client", None)
    if client is None:
        client = AbliterationClient(config.provider)
        _worker_state.client = client
    return client


def _retryable_after_client(result: CompletionResult) -> bool:
    status = result.status_code
    return not result.success and (status is None or status == 429 or status >= 500)


def _resilient_complete(
    config: ApiExperimentConfig,
    messages: Sequence[Message],
    generation: GenerationConfig,
    events: EventWriter,
    *,
    task_type: str,
    sample_id: str,
    phase: str,
) -> CompletionResult:
    reconnect_cycle = 0
    while True:
        result = _worker_client(config).complete(messages, generation)
        events.write(
            task_type=task_type,
            sample_id=sample_id,
            phase=phase,
            result=result,
        )
        if not _retryable_after_client(result):
            return result

        delay = min(
            config.execution.reconnect_initial_seconds * (2**reconnect_cycle),
            config.execution.reconnect_max_seconds,
        )
        reconnect_cycle = min(reconnect_cycle + 1, 20)
        print(
            f"Transient {phase} failure for {task_type}/{sample_id}; "
            f"retrying in {delay:.0f}s",
            flush=True,
        )
        time.sleep(delay)


def _capability_task(
    config: ApiExperimentConfig,
    run_id: str,
    sample: CapabilitySample,
    events: EventWriter,
) -> tuple[dict[str, Any], float]:
    messages = capability_messages(sample)
    result = _resilient_complete(
        config,
        messages,
        config.target_generation,
        events,
        task_type="capability",
        sample_id=sample.sample_id,
        phase="target",
    )
    row = _capability_row(
        run_id,
        config.capability_dataset.name,
        sample,
        messages,
        result,
    )
    return row, _preferred_cost(result)


def _propensity_task(
    config: ApiExperimentConfig,
    run_id: str,
    sample: PropensitySample,
    events: EventWriter,
) -> tuple[dict[str, Any], float]:
    target_messages: list[Message] = [{"role": "user", "content": sample.question}]
    target_result = _resilient_complete(
        config,
        target_messages,
        config.target_generation,
        events,
        task_type="propensity",
        sample_id=sample.sample_id,
        phase="target",
    )
    row = _propensity_target_row(
        config,
        run_id,
        sample,
        target_messages,
        target_result,
    )
    cost = _preferred_cost(target_result)
    if not target_result.success:
        return row, cost

    judge_messages = strongreject_messages(sample.question, target_result.content)
    judge_result = _resilient_complete(
        config,
        judge_messages,
        config.judge_generation,
        events,
        task_type="propensity",
        sample_id=sample.sample_id,
        phase="judge",
    )
    _add_judgment(config, row, judge_messages, judge_result)
    return row, cost + _preferred_cost(judge_result)


_BOOLEAN_FIELDS = {
    "target_success",
    "capability_correct",
    "judge_success",
    "strongreject_useful_at_threshold",
}

_NUMBER_FIELDS = {
    "unsafe_probability",
    "target_status_code",
    "target_latency_seconds",
    "target_attempts",
    "target_prompt_tokens",
    "target_cached_input_tokens",
    "target_completion_tokens",
    "target_reasoning_tokens",
    "target_total_tokens",
    "target_estimated_cost_usd",
    "target_recalculated_cost_usd",
    "target_estimated_credits_used",
    "target_remaining_credits",
    "judge_status_code",
    "judge_latency_seconds",
    "judge_attempts",
    "judge_prompt_tokens",
    "judge_cached_input_tokens",
    "judge_completion_tokens",
    "judge_reasoning_tokens",
    "judge_total_tokens",
    "judge_estimated_cost_usd",
    "judge_recalculated_cost_usd",
    "judge_estimated_credits_used",
    "judge_remaining_credits",
    "strongreject_refusal",
    "strongreject_convincingness",
    "strongreject_specificity",
    "strongreject_score",
}


def _load_transcript(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in _BOOLEAN_FIELDS:
            value = row.get(field, "")
            if value:
                row[field] = value.lower() == "true"
        for field in _NUMBER_FIELDS:
            value = row.get(field, "")
            if value:
                row[field] = float(value)
    return rows


def _row_cost(row: Mapping[str, Any]) -> float:
    total = 0.0
    for prefix in ("target", "judge"):
        value = row.get(f"{prefix}_estimated_cost_usd")
        if not isinstance(value, (int, float)):
            value = row.get(f"{prefix}_recalculated_cost_usd")
        if isinstance(value, (int, float)):
            total += float(value)
    return total


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_summary(
    output_dir: Path,
    config: ApiExperimentConfig,
    summary: Mapping[str, Any],
) -> None:
    _atomic_json(output_dir / config.output.summary_json_filename, summary)
    csv_path = output_dir / config.output.summary_csv_filename
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric, value in summary.items():
            writer.writerow({"metric": metric, "value": value})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, csv_path)


def _summary(
    rows: Sequence[Mapping[str, Any]],
    config: ApiExperimentConfig,
    run_id: str,
    *,
    partial: bool,
) -> dict[str, Any]:
    result = compute_summary(rows)
    result.update(
        {
            "run_id": run_id,
            "model_id": config.provider.model_id,
            "useful_score_threshold": config.scoring.useful_score_threshold,
            "pricing_as_of": config.provider.pricing.as_of,
            "partial": partial,
            "workers": config.execution.workers,
        }
    )
    return result


def _write_manifest(
    output_dir: Path,
    config: ApiExperimentConfig,
    *,
    run_id: str,
    started_at: str,
    finished_at: str | None,
    status: str,
    capability_samples: int,
    propensity_samples: int,
    completed_rows: int,
    error: str | None = None,
) -> None:
    manifest = {
        "run_id": run_id,
        "started_at_utc": started_at,
        "updated_at_utc": _utc_now(),
        "finished_at_utc": finished_at,
        "status": status,
        "capability_samples": capability_samples,
        "propensity_samples": propensity_samples,
        "completed_rows": completed_rows,
        "error": error,
        "configuration": config.to_dict(),
    }
    _atomic_json(output_dir / "manifest.json", manifest)


def _resume_config_is_compatible(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> bool:
    previous_copy = json.loads(json.dumps(previous))
    current_copy = json.loads(json.dumps(current))
    try:
        previous_limit = float(
            previous_copy["execution"].pop("max_total_estimated_cost_usd")
        )
        current_limit = float(
            current_copy["execution"].pop("max_total_estimated_cost_usd")
        )
    except (KeyError, TypeError, ValueError):
        return False
    return previous_copy == current_copy and current_limit >= previous_limit


def _process_parallel(
    samples: Iterable[Any],
    task: Callable[[Any], tuple[dict[str, Any], float]],
    executor: ThreadPoolExecutor,
    writer: TranscriptWriter,
    rows: list[dict[str, Any]],
    *,
    config: ApiExperimentConfig,
    output_dir: Path,
    run_id: str,
    cumulative_cost: float,
    completed: int,
    total: int,
) -> tuple[float, int]:
    iterator = iter(samples)
    futures: dict[Future[tuple[dict[str, Any], float]], Any] = {}
    stopped_for_cost = False

    def submit_one() -> bool:
        try:
            sample = next(iterator)
        except StopIteration:
            return False
        futures[executor.submit(task, sample)] = sample
        return True

    while len(futures) < config.execution.workers and submit_one():
        pass

    while futures:
        done, _ = wait(futures, return_when=FIRST_COMPLETED)
        for future in done:
            futures.pop(future)
            row, cost = future.result()
            writer.write(row)
            rows.append(row)
            cumulative_cost += cost
            completed += 1

            progress_every = config.execution.progress_every
            if completed == total or (
                progress_every > 0 and completed % progress_every == 0
            ):
                print(
                    f"Completed {completed}/{total} samples; "
                    f"recorded cost ${cumulative_cost:.4f}",
                    flush=True,
                )
                _write_summary(
                    output_dir,
                    config,
                    _summary(rows, config, run_id, partial=True),
                )

            limit = config.execution.max_total_estimated_cost_usd
            if limit is not None and cumulative_cost >= limit:
                stopped_for_cost = True

        while (
            not stopped_for_cost
            and len(futures) < config.execution.workers
            and submit_one()
        ):
            pass

    if stopped_for_cost:
        limit = config.execution.max_total_estimated_cost_usd
        assert limit is not None
        raise CostLimitReached(
            f"Stopped after reaching the ${limit:.2f} recorded-cost ceiling. "
            "Concurrent in-flight calls may cause a small overshoot."
        )
    return cumulative_cost, completed


def run_experiment(config: ApiExperimentConfig) -> Path | None:
    if not config.execution.live:
        print("Dry run only: no API calls will be made.")
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False, sort_keys=True))
        print("Set execution.live=true only after supplying ABLIT_KEY and a cost limit.")
        return None

    if config.execution.max_total_estimated_cost_usd is None:
        raise ValueError(
            "Live runs require an explicit execution.max_total_estimated_cost_usd "
            "limit."
        )
    if config.execution.workers < 1:
        raise ValueError("execution.workers must be at least 1")
    if config.execution.reconnect_initial_seconds <= 0:
        raise ValueError("execution.reconnect_initial_seconds must be positive")
    if (
        config.execution.reconnect_max_seconds
        < config.execution.reconnect_initial_seconds
    ):
        raise ValueError(
            "execution.reconnect_max_seconds must be at least the initial delay"
        )

    run_id = config.output.run_id or _generated_run_id(config.run_name)
    output_dir = config.output.root_dir / run_id
    if output_dir.exists() and not config.execution.resume:
        raise FileExistsError(
            f"Run directory already exists; enable resume to reuse it: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=config.execution.resume)

    transcript_path = output_dir / config.output.transcript_filename
    rows = _load_transcript(transcript_path) if config.execution.resume else []
    completed_keys = {
        (str(row.get("task_type")), str(row.get("sample_id"))) for row in rows
    }
    cumulative_cost = sum(_row_cost(row) for row in rows)

    capability_samples = load_capability_samples(
        config.capability_dataset, config.execution.seed
    )
    propensity_samples = load_propensity_samples(
        config.propensity_dataset, config.execution.seed
    )
    pending_capability = [
        sample
        for sample in capability_samples
        if ("capability", sample.sample_id) not in completed_keys
    ]
    pending_propensity = [
        sample
        for sample in propensity_samples
        if ("propensity", sample.sample_id) not in completed_keys
    ]
    total = len(capability_samples) + len(propensity_samples)
    completed = len(rows)

    manifest_path = output_dir / "manifest.json"
    started_at = _utc_now()
    if config.execution.resume and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
        previous_config = previous_manifest.get("configuration")
        if not isinstance(previous_config, Mapping) or not _resume_config_is_compatible(
            previous_config, config.to_dict()
        ):
            raise ValueError(
                "Resume configuration differs from the existing run manifest. "
                "Only increasing the cost ceiling is permitted."
            )
        started_at = str(previous_manifest.get("started_at_utc") or started_at)

    print(
        f"Run {run_id}: {completed}/{total} samples already durable; "
        f"resuming with {config.execution.workers} workers at "
        f"${cumulative_cost:.4f} recorded cost",
        flush=True,
    )
    _write_manifest(
        output_dir,
        config,
        run_id=run_id,
        started_at=started_at,
        finished_at=None,
        status="running",
        capability_samples=len(capability_samples),
        propensity_samples=len(propensity_samples),
        completed_rows=completed,
    )

    try:
        with (
            TranscriptWriter(transcript_path, append=config.execution.resume) as writer,
            EventWriter(output_dir / "request_events.jsonl") as events,
            ThreadPoolExecutor(
                max_workers=config.execution.workers,
                thread_name_prefix="abliteration-api",
            ) as executor,
        ):
            capability_task = lambda sample: _capability_task(
                config, run_id, sample, events
            )
            cumulative_cost, completed = _process_parallel(
                pending_capability,
                capability_task,
                executor,
                writer,
                rows,
                config=config,
                output_dir=output_dir,
                run_id=run_id,
                cumulative_cost=cumulative_cost,
                completed=completed,
                total=total,
            )

            propensity_task = lambda sample: _propensity_task(
                config, run_id, sample, events
            )
            cumulative_cost, completed = _process_parallel(
                pending_propensity,
                propensity_task,
                executor,
                writer,
                rows,
                config=config,
                output_dir=output_dir,
                run_id=run_id,
                cumulative_cost=cumulative_cost,
                completed=completed,
                total=total,
            )

        final_summary = _summary(rows, config, run_id, partial=False)
        _write_summary(output_dir, config, final_summary)
        _write_manifest(
            output_dir,
            config,
            run_id=run_id,
            started_at=started_at,
            finished_at=_utc_now(),
            status="complete",
            capability_samples=len(capability_samples),
            propensity_samples=len(propensity_samples),
            completed_rows=completed,
        )
    except BaseException as exc:
        if rows:
            _write_summary(
                output_dir,
                config,
                _summary(rows, config, run_id, partial=True),
            )
        if isinstance(exc, KeyboardInterrupt):
            status = "interrupted"
        elif isinstance(exc, CostLimitReached):
            status = "cost_limit_reached"
        else:
            status = "failed"
        _write_manifest(
            output_dir,
            config,
            run_id=run_id,
            started_at=started_at,
            finished_at=_utc_now(),
            status=status,
            capability_samples=len(capability_samples),
            propensity_samples=len(propensity_samples),
            completed_rows=len(rows),
            error=str(exc),
        )
        raise

    print(f"Run complete. Local results: {output_dir.resolve()}", flush=True)
    return output_dir
