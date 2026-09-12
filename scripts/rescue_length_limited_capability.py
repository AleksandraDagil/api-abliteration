from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from safety_gap.api_evaluation.client import AbliterationClient, CompletionResult, Message
from safety_gap.api_evaluation.config import ApiExperimentConfig, GenerationConfig
from safety_gap.api_evaluation.datasets import choice_letters
from safety_gap.api_evaluation.metrics import compute_summary
from safety_gap.api_evaluation.runner import _load_transcript
from safety_gap.api_evaluation.scoring import parse_choice_letter
from safety_gap.api_evaluation.transcript import json_value


REASON = "length-limit rescue"
RUNS = (
    "abliterated_model_bio_full",
    "abliterated_model_large_bio_full",
)
TARGET_FIELDS = (
    "target_request_payload_json",
    "target_answer",
    "target_reasoning",
    "target_annotations_json",
    "target_raw_response_json",
    "target_success",
    "target_error_type",
    "target_error_message",
    "target_status_code",
    "target_response_id",
    "target_returned_model",
    "target_finish_reason",
    "target_latency_seconds",
    "target_attempts",
    "target_retry_delays_seconds_json",
    "target_prompt_tokens",
    "target_cached_input_tokens",
    "target_completion_tokens",
    "target_reasoning_tokens",
    "target_total_tokens",
    "target_estimated_cost_usd",
    "target_recalculated_cost_usd",
    "target_estimated_credits_used",
    "target_remaining_credits",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _load_env_key(path: Path, name: str) -> None:
    if os.getenv(name):
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if value:
            os.environ[name] = value
            return
    raise EnvironmentError(f"No non-empty {name} entry found in {path}")


def _config_from_manifest(run_dir: Path) -> ApiExperimentConfig:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    raw = manifest.get("configuration")
    if not isinstance(raw, Mapping):
        raise ValueError(f"Manifest configuration is missing in {run_dir}")
    normalized = dict(raw)
    if "datasets" not in normalized:
        capability = normalized.pop("capability_dataset", None)
        propensity = normalized.pop("propensity_dataset", None)
        if not isinstance(capability, Mapping) or not isinstance(propensity, Mapping):
            raise ValueError(f"Dataset configuration is missing in {run_dir}")
        normalized["datasets"] = {
            "capability": capability,
            "propensity": propensity,
        }
    return ApiExperimentConfig.from_mapping(normalized)


def _working_transcript(run_dir: Path) -> Path:
    rescued = run_dir / "transcript.length_limit_rescue.csv"
    relaxed = run_dir / "transcript.relaxed_parser.csv"
    if rescued.exists():
        return rescued
    return relaxed if relaxed.exists() else run_dir / "transcript.csv"


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _write_csv_atomic(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> Path:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    except PermissionError:
        fallback = path.with_name(f"{path.stem}.length_limit_rescue{path.suffix}")
        os.replace(temporary, fallback)
        print(
            f"Could not replace open CSV {path}; completed output is {fallback}",
            flush=True,
        )
        return fallback
    return path


def _preferred_cost(result: CompletionResult) -> float:
    if result.estimated_cost_usd is not None:
        return result.estimated_cost_usd
    return result.recalculated_cost_usd or 0.0


def _result_columns(result: CompletionResult) -> dict[str, Any]:
    return {
        "target_request_payload_json": json_value(result.request_payload),
        "target_answer": result.content,
        "target_reasoning": result.reasoning_content,
        "target_annotations_json": json_value(result.annotations),
        "target_raw_response_json": result.raw_response_json(),
        "target_success": result.success,
        "target_error_type": result.error_type or "",
        "target_error_message": result.error_message or "",
        "target_status_code": result.status_code,
        "target_response_id": result.response_id or "",
        "target_returned_model": result.returned_model or "",
        "target_finish_reason": result.finish_reason or "",
        "target_latency_seconds": result.latency_seconds,
        "target_attempts": result.attempts,
        "target_retry_delays_seconds_json": json_value(result.retry_delays_seconds),
        "target_prompt_tokens": result.prompt_tokens,
        "target_cached_input_tokens": result.cached_input_tokens,
        "target_completion_tokens": result.completion_tokens,
        "target_reasoning_tokens": result.reasoning_tokens,
        "target_total_tokens": result.total_tokens,
        "target_estimated_cost_usd": result.estimated_cost_usd,
        "target_recalculated_cost_usd": result.recalculated_cost_usd,
        "target_estimated_credits_used": result.estimated_credits_used,
        "target_remaining_credits": result.remaining_credits,
    }


class AuditWriter:
    def __init__(self, path: Path) -> None:
        self._handle = path.open("a", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()

    def write(
        self,
        *,
        run_id: str,
        model_id: str,
        sample_id: str,
        original_response_id: str,
        max_tokens: int,
        result: CompletionResult,
    ) -> None:
        event = {
            "event_type": REASON,
            "timestamp_utc": _utc_now(),
            "run_id": run_id,
            "model_id": model_id,
            "sample_id": sample_id,
            "original_response_id": original_response_id,
            "max_tokens": max_tokens,
            "success": result.success,
            "finish_reason": result.finish_reason,
            "status_code": result.status_code,
            "error_type": result.error_type,
            "error_message": result.error_message,
            "attempts": result.attempts,
            "latency_seconds": result.latency_seconds,
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
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()

    def __enter__(self) -> "AuditWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


_worker_state = threading.local()


def _worker_client(config: ApiExperimentConfig) -> AbliterationClient:
    clients = getattr(_worker_state, "clients", None)
    if clients is None:
        clients = {}
        _worker_state.clients = clients
    key = (config.provider.base_url, config.provider.model_id)
    if key not in clients:
        clients[key] = AbliterationClient(config.provider)
    return clients[key]


def _rescue_one(
    config: ApiExperimentConfig,
    run_id: str,
    row: Mapping[str, str],
    audit: AuditWriter,
) -> tuple[str, CompletionResult, int, float]:
    messages_raw = json.loads(row["target_messages_json"])
    if not isinstance(messages_raw, list):
        raise ValueError(f"Invalid messages for {run_id}/{row['sample_id']}")
    messages: list[Message] = [dict(message) for message in messages_raw]
    original_response_id = row.get("target_response_id", "")
    total_cost = 0.0
    final_result: CompletionResult | None = None
    final_budget = 8192

    for max_tokens in (8192, 16384):
        generation: GenerationConfig = replace(
            config.target_generation,
            max_tokens=max_tokens,
        )
        result = _worker_client(config).complete(messages, generation)
        audit.write(
            run_id=run_id,
            model_id=config.provider.model_id,
            sample_id=row["sample_id"],
            original_response_id=original_response_id,
            max_tokens=max_tokens,
            result=result,
        )
        total_cost += _preferred_cost(result)
        final_result = result
        final_budget = max_tokens
        if not result.success or result.finish_reason != "length":
            break

    assert final_result is not None
    return row["sample_id"], final_result, final_budget, total_cost


def _replace_row(row: dict[str, str], result: CompletionResult) -> None:
    row["timestamp_utc"] = _utc_now()
    row.update(_result_columns(result))
    choices = json.loads(row["choices_json"])
    letters = choice_letters(choices)
    parsed = parse_choice_letter(result.content, letters) if result.success else None
    row["parsed_answer_letter"] = parsed or ""
    row["capability_correct"] = str(parsed == row["correct_answer_letter"])


def _write_summaries(run_dir: Path, transcript: Path) -> None:
    summary_json = run_dir / "summary.json"
    summary_csv = run_dir / "summary.csv"
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    summary.update(compute_summary(_load_transcript(transcript)))

    temporary_json = summary_json.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_json, summary_json)

    temporary_csv = summary_csv.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows({"metric": key, "value": value} for key, value in summary.items())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_csv, summary_csv)


def _write_invalid_review(run_dir: Path, rows: Sequence[Mapping[str, str]]) -> None:
    fields = [
        "sample_id",
        "question",
        "choices_json",
        "correct_answer_letter",
        "target_answer",
        "target_reasoning",
        "target_finish_reason",
        "target_status_code",
        "target_completion_tokens",
        "target_total_tokens",
    ]
    invalid = []
    for row in rows:
        if row.get("task_type") != "capability":
            continue
        choices = json.loads(row["choices_json"])
        parsed = parse_choice_letter(row.get("target_answer", ""), choice_letters(choices))
        if not parsed:
            invalid.append({field: row.get(field, "") for field in fields})
    _write_csv_atomic(run_dir / "invalid_capability_review.csv", fields, invalid)


def _backup_once(source: Path, destination: Path) -> None:
    if not destination.exists():
        shutil.copy2(source, destination)


def rescue_run(
    repository: Path,
    run_id: str,
    *,
    max_additional_cost_usd: float,
    cost_already_used: float,
    base_url: str | None,
    api_key_env: str | None,
) -> float:
    run_dir = repository / "outputs" / "api_runs" / run_id
    config = _config_from_manifest(run_dir)
    if base_url or api_key_env:
        config = replace(
            config,
            provider=replace(
                config.provider,
                base_url=(base_url or config.provider.base_url).rstrip("/"),
                api_key_env=api_key_env or config.provider.api_key_env,
            ),
        )
    _load_env_key(repository / ".env", config.provider.api_key_env)
    transcript = _working_transcript(run_dir)
    fields, rows = _read_csv(transcript)

    invalid_rows = []
    for row in rows:
        if row.get("task_type") != "capability":
            continue
        choices = json.loads(row["choices_json"])
        parsed = parse_choice_letter(row.get("target_answer", ""), choice_letters(choices))
        if not parsed:
            invalid_rows.append(row)

    print(
        f"{run_id}: {len(invalid_rows)} length-limit rescue rows from {transcript.name}",
        flush=True,
    )
    if not invalid_rows:
        return cost_already_used

    _backup_once(
        transcript,
        run_dir / f"{transcript.stem}.pre_length_limit_rescue.csv",
    )
    _backup_once(run_dir / "summary.json", run_dir / "summary.pre_length_limit_rescue.json")
    _backup_once(run_dir / "summary.csv", run_dir / "summary.pre_length_limit_rescue.csv")

    row_by_id = {row["sample_id"]: row for row in rows if row.get("task_type") == "capability"}
    completed = 0
    replaced = 0
    failures = 0
    run_cost = 0.0
    audit_path = run_dir / "length_limit_rescue_events.jsonl"

    with AuditWriter(audit_path) as audit, ThreadPoolExecutor(
        max_workers=config.execution.workers,
        thread_name_prefix=f"rescue-{config.provider.model_id}",
    ) as executor:
        iterator = iter(invalid_rows)
        futures: dict[Future[tuple[str, CompletionResult, int, float]], Mapping[str, str]] = {}

        def submit_one() -> bool:
            if cost_already_used + run_cost >= max_additional_cost_usd:
                return False
            try:
                row = next(iterator)
            except StopIteration:
                return False
            future = executor.submit(_rescue_one, config, run_id, row, audit)
            futures[future] = row
            return True

        while len(futures) < config.execution.workers and submit_one():
            pass

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                original = futures.pop(future)
                sample_id, result, final_budget, cost = future.result()
                completed += 1
                run_cost += cost
                if result.success:
                    _replace_row(row_by_id[sample_id], result)
                    replaced += 1
                else:
                    failures += 1
                print(
                    f"{run_id}: {completed}/{len(invalid_rows)} sample={sample_id} "
                    f"budget={final_budget} finish={result.finish_reason or 'error'} "
                    f"replaced={result.success} rescue_cost=${run_cost:.4f}",
                    flush=True,
                )

            while len(futures) < config.execution.workers and submit_one():
                pass

    transcript = _write_csv_atomic(transcript, fields, rows)
    _write_summaries(run_dir, transcript)
    _write_invalid_review(run_dir, rows)
    incomplete = completed < len(invalid_rows)
    if incomplete:
        status = "cost_limit_reached"
    elif failures:
        status = "completed_with_failures"
    else:
        status = "complete"
    manifest = {
        "event_type": REASON,
        "status": status,
        "updated_at_utc": _utc_now(),
        "run_id": run_id,
        "source_transcript": str(transcript),
        "requested_rows": len(invalid_rows),
        "replaced_rows": replaced,
        "failed_rows": failures,
        "additional_estimated_cost_usd": run_cost,
        "first_max_tokens": 8192,
        "length_retry_max_tokens": 16384,
        "audit_log": str(audit_path),
    }
    (run_dir / "length_limit_rescue_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"{run_id}: {status}; "
        f"replaced={replaced}, failures={failures}, "
        f"additional cost=${run_cost:.4f}",
        flush=True,
    )
    if incomplete:
        raise RuntimeError(
            f"Rescue cost ceiling ${max_additional_cost_usd:.2f} reached after "
            f"{completed}/{len(invalid_rows)} rows; partial replacements were saved"
        )
    return cost_already_used + run_cost


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate only capability rows truncated by the token limit."
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=REPOSITORY_ROOT,
    )
    parser.add_argument(
        "--max-additional-cost-usd",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--base-url",
        help="Override the manifest endpoint for every rescue run.",
    )
    parser.add_argument(
        "--api-key-env",
        help="Use one pre-populated environment variable for every rescue run.",
    )
    args = parser.parse_args()
    repository = args.repository.resolve()
    cost = 0.0
    for run_id in RUNS:
        cost = rescue_run(
            repository,
            run_id,
            max_additional_cost_usd=args.max_additional_cost_usd,
            cost_already_used=cost,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
        )
    print(f"All rescue runs complete; total additional cost=${cost:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
