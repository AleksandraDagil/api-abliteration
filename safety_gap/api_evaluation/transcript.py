from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping


TRANSCRIPT_FIELDS = [
    "run_id",
    "timestamp_utc",
    "task_type",
    "dataset_name",
    "sample_id",
    "question",
    "choices_json",
    "correct_answer_letter",
    "unsafe_probability",
    "target_messages_json",
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
    "parsed_answer_letter",
    "capability_correct",
    "judge_messages_json",
    "judge_request_payload_json",
    "judge_answer",
    "judge_reasoning",
    "judge_annotations_json",
    "judge_raw_response_json",
    "judge_success",
    "judge_error_type",
    "judge_error_message",
    "judge_status_code",
    "judge_response_id",
    "judge_returned_model",
    "judge_finish_reason",
    "judge_latency_seconds",
    "judge_attempts",
    "judge_retry_delays_seconds_json",
    "judge_prompt_tokens",
    "judge_cached_input_tokens",
    "judge_completion_tokens",
    "judge_reasoning_tokens",
    "judge_total_tokens",
    "judge_estimated_cost_usd",
    "judge_recalculated_cost_usd",
    "judge_estimated_credits_used",
    "judge_remaining_credits",
    "judge_parse_error",
    "strongreject_refusal",
    "strongreject_convincingness",
    "strongreject_specificity",
    "strongreject_score",
    "strongreject_useful_at_threshold",
]


def json_value(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class TranscriptWriter:
    def __init__(self, path: Path, *, append: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        existing = append and path.exists() and path.stat().st_size > 0
        mode = "a" if existing else "w"
        encoding = "utf-8" if existing else "utf-8-sig"
        self._handle = path.open(mode, encoding=encoding, newline="")
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=TRANSCRIPT_FIELDS,
            extrasaction="raise",
            quoting=csv.QUOTE_MINIMAL,
        )
        if not existing:
            self._writer.writeheader()
            self._sync()

    def _sync(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def write(self, row: Mapping[str, Any]) -> None:
        normalized = {field: row.get(field, "") for field in TRANSCRIPT_FIELDS}
        self._writer.writerow(normalized)
        # Preserve every completed API interaction if a long run is interrupted.
        self._sync()

    def close(self) -> None:
        if not self._handle.closed:
            self._sync()
            self._handle.close()

    def __enter__(self) -> "TranscriptWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
