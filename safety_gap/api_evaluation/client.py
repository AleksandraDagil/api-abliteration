from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from safety_gap.api_evaluation.config import GenerationConfig, ProviderConfig


Message = dict[str, str]


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "dict"):
        dumped = value.dict()
        if isinstance(dumped, dict):
            return dumped
    raise TypeError(f"Cannot convert {type(value).__name__} to a response dictionary")


def _number(value: Any, default: int | float | None = None) -> int | float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return default


def _nested(mapping: Mapping[str, Any], *path: str) -> Any:
    current: Any = mapping
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


@dataclass
class CompletionResult:
    success: bool
    request_payload: dict[str, Any]
    content: str = ""
    reasoning_content: str = ""
    annotations: Any = None
    finish_reason: str | None = None
    response_id: str | None = None
    returned_model: str | None = None
    latency_seconds: float = 0.0
    attempts: int = 0
    status_code: int | None = None
    prompt_tokens: int | None = None
    cached_input_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    recalculated_cost_usd: float | None = None
    estimated_credits_used: int | float | None = None
    remaining_credits: int | float | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None
    retry_delays_seconds: list[float] = field(default_factory=list)

    def raw_response_json(self) -> str:
        return json.dumps(self.raw_response, ensure_ascii=False, sort_keys=True)


class AbliterationClient:
    """Small measured wrapper around abliteration.ai's OpenAI-compatible endpoint."""

    def __init__(
        self,
        provider: ProviderConfig,
        *,
        api_key: str | None = None,
        create_completion: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.provider = provider
        self._sleep = sleep
        self._clock = clock

        if create_completion is not None:
            self._create_completion = create_completion
            return

        resolved_key = api_key or os.getenv(provider.api_key_env)
        if not resolved_key:
            raise EnvironmentError(
                f"Missing API key. Set the {provider.api_key_env} environment variable."
            )

        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The OpenAI Python SDK is required. Install requirements-api.txt."
            ) from exc

        sdk_client = OpenAI(
            base_url=provider.base_url,
            api_key=resolved_key,
            timeout=provider.timeout_seconds,
            # Retry here so that every retry is visible in our own metrics.
            max_retries=0,
        )
        self._create_completion = sdk_client.chat.completions.create

    def _request_payload(
        self,
        messages: Sequence[Message],
        generation: GenerationConfig,
    ) -> dict[str, Any]:
        extra_body: dict[str, Any] = {
            "include_reasoning": generation.include_reasoning,
        }
        if generation.web_search.enabled:
            extra_body["web_search_options"] = {
                "search_context_size": generation.web_search.search_context_size,
            }

        return {
            "model": self.provider.model_id,
            "messages": [dict(message) for message in messages],
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "max_tokens": generation.max_tokens,
            "reasoning_effort": generation.reasoning_effort,
            "extra_body": extra_body,
        }

    def _recalculated_cost(
        self,
        prompt_tokens: int | None,
        cached_input_tokens: int | None,
        completion_tokens: int | None,
    ) -> float | None:
        if prompt_tokens is None or completion_tokens is None:
            return None
        cached = max(0, cached_input_tokens or 0)
        uncached = max(0, prompt_tokens - cached)
        pricing = self.provider.pricing
        return (
            uncached * pricing.input_per_million_usd
            + cached * pricing.cached_input_per_million_usd
            + completion_tokens * pricing.output_per_million_usd
        ) / 1_000_000

    def _success_result(
        self,
        response: Any,
        request_payload: dict[str, Any],
        latency_seconds: float,
        attempts: int,
        retry_delays: list[float],
    ) -> CompletionResult:
        raw = _to_dict(response)
        choices = raw.get("choices") or []
        first_choice = choices[0] if choices and isinstance(choices[0], Mapping) else {}
        message = (
            first_choice.get("message", {}) if isinstance(first_choice, Mapping) else {}
        )
        if not isinstance(message, Mapping):
            message = {}
        usage = raw.get("usage") or {}
        if not isinstance(usage, Mapping):
            usage = {}

        prompt_tokens = _number(usage.get("prompt_tokens"))
        completion_tokens = _number(usage.get("completion_tokens"))
        total_tokens = _number(usage.get("total_tokens"))

        cached_tokens = _nested(usage, "prompt_tokens_details", "cached_tokens")
        if cached_tokens is None:
            cached_tokens = usage.get("cached_input_tokens")
        reasoning_tokens = _nested(usage, "completion_tokens_details", "reasoning_tokens")
        if reasoning_tokens is None:
            reasoning_tokens = usage.get("reasoning_tokens")

        prompt_tokens_int = int(prompt_tokens) if prompt_tokens is not None else None
        completion_tokens_int = (
            int(completion_tokens) if completion_tokens is not None else None
        )
        cached_tokens_int = int(cached_tokens) if _number(cached_tokens) is not None else None
        reasoning_tokens_int = (
            int(reasoning_tokens) if _number(reasoning_tokens) is not None else None
        )

        provider_cost = _number(raw.get("estimated_cost_usd"))
        estimated_credits = _number(raw.get("estimated_credits_used"))
        remaining_credits = _number(raw.get("remaining_credits"))

        content = message.get("content") or ""
        reasoning_content = (
            message.get("reasoning_content") or message.get("reasoning") or ""
        )
        return CompletionResult(
            success=True,
            request_payload=request_payload,
            content=str(content),
            reasoning_content=str(reasoning_content),
            annotations=message.get("annotations") or raw.get("citations"),
            finish_reason=(
                str(first_choice.get("finish_reason"))
                if isinstance(first_choice, Mapping)
                and first_choice.get("finish_reason") is not None
                else None
            ),
            response_id=str(raw.get("id")) if raw.get("id") is not None else None,
            returned_model=(
                str(raw.get("model")) if raw.get("model") is not None else None
            ),
            latency_seconds=latency_seconds,
            attempts=attempts,
            status_code=200,
            prompt_tokens=prompt_tokens_int,
            cached_input_tokens=cached_tokens_int,
            completion_tokens=completion_tokens_int,
            reasoning_tokens=reasoning_tokens_int,
            total_tokens=int(total_tokens) if total_tokens is not None else None,
            estimated_cost_usd=(float(provider_cost) if provider_cost is not None else None),
            recalculated_cost_usd=self._recalculated_cost(
                prompt_tokens_int, cached_tokens_int, completion_tokens_int
            ),
            estimated_credits_used=estimated_credits,
            remaining_credits=remaining_credits,
            raw_response=raw,
            retry_delays_seconds=retry_delays,
        )

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        status_code = getattr(exc, "status_code", None)
        return int(status_code) if isinstance(status_code, int) else None

    @staticmethod
    def _retry_after(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            return None
        value = headers.get("retry-after") or headers.get("Retry-After")
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _retryable(status_code: int | None) -> bool:
        return status_code is None or status_code == 429 or status_code >= 500

    def complete(
        self,
        messages: Sequence[Message],
        generation: GenerationConfig,
    ) -> CompletionResult:
        request_payload = self._request_payload(messages, generation)
        start = self._clock()
        retry_delays: list[float] = []
        attempts = 0
        last_error: Exception | None = None

        for attempt_index in range(self.provider.max_retries + 1):
            attempts += 1
            try:
                response = self._create_completion(**request_payload)
                return self._success_result(
                    response,
                    request_payload,
                    self._clock() - start,
                    attempts,
                    retry_delays,
                )
            except Exception as exc:  # SDK error classes vary by installed version.
                last_error = exc
                status_code = self._status_code(exc)
                if attempt_index >= self.provider.max_retries or not self._retryable(status_code):
                    break

                retry_after = self._retry_after(exc)
                if retry_after is None:
                    retry_after = min(
                        self.provider.retry_initial_seconds * (2**attempt_index),
                        self.provider.retry_max_seconds,
                    )
                retry_delays.append(retry_after)
                self._sleep(retry_after)

        assert last_error is not None
        return CompletionResult(
            success=False,
            request_payload=request_payload,
            latency_seconds=self._clock() - start,
            attempts=attempts,
            status_code=self._status_code(last_error),
            error_type=type(last_error).__name__,
            error_message=str(last_error),
            retry_delays_seconds=retry_delays,
        )
