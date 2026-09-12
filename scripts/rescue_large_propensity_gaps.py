from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from safety_gap.api_evaluation.client import AbliterationClient, CompletionResult, Message
from safety_gap.api_evaluation.runner import _completion_columns
from safety_gap.api_evaluation.scoring import parse_strongreject, strongreject_messages
from safety_gap.api_evaluation.transcript import json_value
from scripts.rescue_length_limited_capability import (
    _backup_once,
    _config_from_manifest,
    _load_env_key,
    _preferred_cost,
    _read_csv,
    _utc_now,
    _write_csv_atomic,
    _write_summaries,
)


RUN_ID = "abliterated_model_large_bio_full"
REASON = "large propensity rescue"


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _messages(value: str, *, sample_id: str, stage: str) -> list[Message]:
    raw = json.loads(value)
    if not isinstance(raw, list):
        raise ValueError(f"Invalid {stage} messages for sample {sample_id}")
    return [dict(message) for message in raw]


def _needs_judge(row: Mapping[str, str], threshold: float) -> bool:
    if not _truthy(row.get("target_success")):
        return True
    if not _truthy(row.get("judge_success")):
        return True
    try:
        parse_strongreject(row.get("judge_answer", ""), threshold)
    except ValueError:
        return True
    return False


class AuditWriter:
    def __init__(self, path: Path) -> None:
        self._handle = path.open("a", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()

    def write(
        self,
        *,
        sample_id: str,
        stage: str,
        original_response_id: str,
        result: CompletionResult,
    ) -> None:
        event = {
            "event_type": REASON,
            "timestamp_utc": _utc_now(),
            "run_id": RUN_ID,
            "model_id": result.request_payload.get("model"),
            "sample_id": sample_id,
            "stage": stage,
            "original_response_id": original_response_id,
            "max_tokens": result.request_payload.get("max_tokens"),
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
        with self._lock:
            self._handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
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


def _client(config) -> AbliterationClient:
    client = getattr(_worker_state, "client", None)
    if client is None:
        client = AbliterationClient(config.provider)
        _worker_state.client = client
    return client


@dataclass
class RescueResult:
    sample_id: str
    target: CompletionResult | None
    judge: CompletionResult | None
    judge_messages: list[Message] | None
    cost: float


def _rescue_one(config, row: Mapping[str, str], audit: AuditWriter) -> RescueResult:
    sample_id = row["sample_id"]
    client = _client(config)
    cost = 0.0
    target_result: CompletionResult | None = None

    if not _truthy(row.get("target_success")):
        target_messages = _messages(
            row["target_messages_json"], sample_id=sample_id, stage="target"
        )
        target_result = client.complete(target_messages, config.target_generation)
        audit.write(
            sample_id=sample_id,
            stage="target",
            original_response_id=row.get("target_response_id", ""),
            result=target_result,
        )
        cost += _preferred_cost(target_result)
        if not target_result.success:
            return RescueResult(sample_id, target_result, None, None, cost)
        target_answer = target_result.content
    else:
        target_answer = row.get("target_answer", "")

    stored_judge_messages = row.get("judge_messages_json", "")
    if stored_judge_messages:
        judge_messages = _messages(
            stored_judge_messages, sample_id=sample_id, stage="judge"
        )
        if target_result is not None:
            # A regenerated target requires a matching fresh judge prompt.
            judge_messages = strongreject_messages(row["question"], target_answer)
    else:
        judge_messages = strongreject_messages(row["question"], target_answer)

    judge_result = client.complete(judge_messages, config.judge_generation)
    audit.write(
        sample_id=sample_id,
        stage="judge",
        original_response_id=row.get("judge_response_id", ""),
        result=judge_result,
    )
    cost += _preferred_cost(judge_result)
    return RescueResult(sample_id, target_result, judge_result, judge_messages, cost)


def _apply_result(config, row: dict[str, str], result: RescueResult) -> bool:
    row["timestamp_utc"] = _utc_now()
    if result.target is not None:
        row.update(_completion_columns(result.target, "target"))
    if result.judge is None:
        return False

    row["judge_messages_json"] = json_value(result.judge_messages)
    row.update(_completion_columns(result.judge, "judge"))
    row.update(
        {
            "judge_parse_error": "",
            "strongreject_refusal": "",
            "strongreject_convincingness": "",
            "strongreject_specificity": "",
            "strongreject_score": "",
            "strongreject_useful_at_threshold": "",
        }
    )
    if not result.judge.success:
        return False
    try:
        judgment = parse_strongreject(
            result.judge.content, config.scoring.useful_score_threshold
        )
    except ValueError as exc:
        row["judge_parse_error"] = str(exc)
        return False
    row.update(
        {
            "strongreject_refusal": str(judgment.refusal),
            "strongreject_convincingness": str(judgment.convincingness),
            "strongreject_specificity": str(judgment.specificity),
            "strongreject_score": str(judgment.score),
            "strongreject_useful_at_threshold": str(judgment.useful),
        }
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate missing large-model propensity targets and judgments."
    )
    parser.add_argument("--repository", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--max-additional-cost-usd", type=float, default=25.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repository = args.repository.resolve()
    run_dir = repository / "outputs" / "api_runs" / RUN_ID
    transcript = run_dir / "transcript.csv"
    config = _config_from_manifest(run_dir)
    fields, rows = _read_csv(transcript)
    threshold = config.scoring.useful_score_threshold
    gaps = [
        row
        for row in rows
        if row.get("task_type") == "propensity" and _needs_judge(row, threshold)
    ]
    missing_targets = sum(not _truthy(row.get("target_success")) for row in gaps)
    requested_judges = len(gaps)
    print(
        f"{RUN_ID}: target generations={missing_targets}, judge calls={requested_judges}",
        flush=True,
    )
    if args.dry_run or not gaps:
        return 0

    _load_env_key(repository / ".env", config.provider.api_key_env)
    _backup_once(transcript, run_dir / "transcript.pre_propensity_rescue.csv")
    _backup_once(run_dir / "summary.json", run_dir / "summary.pre_propensity_rescue.json")
    _backup_once(run_dir / "summary.csv", run_dir / "summary.pre_propensity_rescue.csv")

    row_by_id = {
        row["sample_id"]: row for row in rows if row.get("task_type") == "propensity"
    }
    audit_path = run_dir / "propensity_rescue_events.jsonl"
    run_cost = 0.0
    completed = 0
    resolved = 0
    failed = 0
    stop_submitting = False

    with AuditWriter(audit_path) as audit, ThreadPoolExecutor(
        max_workers=config.execution.workers,
        thread_name_prefix="large-propensity-rescue",
    ) as executor:
        iterator = iter(gaps)
        futures: dict[Future[RescueResult], Mapping[str, str]] = {}

        def submit_one() -> bool:
            nonlocal stop_submitting
            if stop_submitting or run_cost >= args.max_additional_cost_usd:
                stop_submitting = True
                return False
            try:
                row = next(iterator)
            except StopIteration:
                return False
            futures[executor.submit(_rescue_one, config, row, audit)] = row
            return True

        while len(futures) < config.execution.workers and submit_one():
            pass

        since_checkpoint = 0
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                original = futures.pop(future)
                result = future.result()
                completed += 1
                since_checkpoint += 1
                run_cost += result.cost
                status_codes = {
                    completion.status_code
                    for completion in (result.target, result.judge)
                    if completion is not None
                }
                if status_codes.intersection({401, 402}):
                    stop_submitting = True
                if _apply_result(config, row_by_id[result.sample_id], result):
                    resolved += 1
                else:
                    failed += 1
                print(
                    f"{completed}/{len(gaps)} sample={result.sample_id} "
                    f"target={'new' if result.target else 'stored'} "
                    f"judge={'ok' if result.judge and result.judge.success else 'failed'} "
                    f"resolved={resolved} cost=${run_cost:.4f}",
                    flush=True,
                )

            if since_checkpoint >= 5:
                transcript = _write_csv_atomic(transcript, fields, rows)
                since_checkpoint = 0
            while len(futures) < config.execution.workers and submit_one():
                pass

    transcript = _write_csv_atomic(transcript, fields, rows)
    _write_summaries(run_dir, transcript)
    shutil.copy2(transcript, run_dir / "transcript.relaxed_parser.csv")

    remaining = sum(
        1
        for row in rows
        if row.get("task_type") == "propensity" and _needs_judge(row, threshold)
    )
    status = "complete" if remaining == 0 else "completed_with_gaps"
    audit_events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    def event_cost(event: Mapping[str, Any]) -> float:
        estimated = event.get("estimated_cost_usd")
        if isinstance(estimated, (int, float)):
            return float(estimated)
        recalculated = event.get("recalculated_cost_usd")
        return float(recalculated) if isinstance(recalculated, (int, float)) else 0.0

    manifest = {
        "event_type": REASON,
        "status": status,
        "updated_at_utc": _utc_now(),
        "run_id": RUN_ID,
        "source_transcript": str(transcript),
        "requested_target_generations": missing_targets,
        "requested_judge_calls": requested_judges,
        "processed_rows": completed,
        "resolved_rows": resolved,
        "failed_rows": failed,
        "remaining_rows": remaining,
        "additional_estimated_cost_usd": run_cost,
        "cumulative_additional_estimated_cost_usd": sum(
            event_cost(event) for event in audit_events
        ),
        "cumulative_successful_target_calls": sum(
            event.get("stage") == "target" and event.get("success") is True
            for event in audit_events
        ),
        "cumulative_successful_judge_calls": sum(
            event.get("stage") == "judge" and event.get("success") is True
            for event in audit_events
        ),
        "audit_event_count": len(audit_events),
        "max_additional_cost_usd": args.max_additional_cost_usd,
        "audit_log": str(audit_path),
    }
    (run_dir / "propensity_rescue_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"{status}: resolved={resolved}/{len(gaps)}, remaining={remaining}, "
        f"additional cost=${run_cost:.4f}",
        flush=True,
    )
    return 0 if remaining == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
