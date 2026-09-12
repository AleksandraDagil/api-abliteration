from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required configuration value: {key}")
    return mapping[key]


@dataclass(frozen=True)
class PricingConfig:
    input_per_million_usd: float
    cached_input_per_million_usd: float
    output_per_million_usd: float
    as_of: str


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    api_key_env: str
    model_id: str
    timeout_seconds: float
    max_retries: int
    retry_initial_seconds: float
    retry_max_seconds: float
    pricing: PricingConfig


@dataclass(frozen=True)
class WebSearchConfig:
    enabled: bool
    search_context_size: str

    def __post_init__(self) -> None:
        if self.search_context_size not in {"low", "medium", "high"}:
            raise ValueError("search_context_size must be low, medium, or high")


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float
    top_p: float
    max_tokens: int
    reasoning_effort: str
    include_reasoning: bool
    web_search: WebSearchConfig


@dataclass(frozen=True)
class CapabilityDatasetConfig:
    name: str
    name_or_path: str
    subset: str
    split: str
    question_column: str
    choices_column: str
    answer_column: str
    sample_limit: int | None


@dataclass(frozen=True)
class PropensityDatasetConfig:
    name: str
    path: Path
    question_column: str
    unsafe_probability_column: str
    unsafe_probability_threshold: float
    sample_limit: int | None


@dataclass(frozen=True)
class ExecutionConfig:
    live: bool
    seed: int
    max_total_estimated_cost_usd: float | None
    progress_every: int
    workers: int
    resume: bool
    reconnect_initial_seconds: float
    reconnect_max_seconds: float


@dataclass(frozen=True)
class OutputConfig:
    root_dir: Path
    run_id: str | None
    transcript_filename: str
    summary_csv_filename: str
    summary_json_filename: str


@dataclass(frozen=True)
class ScoringConfig:
    useful_score_threshold: float


@dataclass(frozen=True)
class ApiExperimentConfig:
    run_name: str
    provider: ProviderConfig
    target_generation: GenerationConfig
    judge_generation: GenerationConfig
    capability_dataset: CapabilityDatasetConfig
    propensity_dataset: PropensityDatasetConfig
    execution: ExecutionConfig
    output: OutputConfig
    scoring: ScoringConfig

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["propensity_dataset"]["path"] = str(self.propensity_dataset.path)
        result["output"]["root_dir"] = str(self.output.root_dir)
        return result

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ApiExperimentConfig":
        provider_raw = _mapping(_required(raw, "provider"), "provider")
        pricing_raw = _mapping(_required(provider_raw, "pricing"), "provider.pricing")
        datasets_raw = _mapping(_required(raw, "datasets"), "datasets")
        capability_raw = _mapping(
            _required(datasets_raw, "capability"), "datasets.capability"
        )
        propensity_raw = _mapping(
            _required(datasets_raw, "propensity"), "datasets.propensity"
        )
        execution_raw = _mapping(_required(raw, "execution"), "execution")
        output_raw = _mapping(_required(raw, "output"), "output")
        scoring_raw = _mapping(_required(raw, "scoring"), "scoring")

        def parse_generation(name: str) -> GenerationConfig:
            generation_raw = _mapping(_required(raw, name), name)
            search_raw = _mapping(
                _required(generation_raw, "web_search"), f"{name}.web_search"
            )
            return GenerationConfig(
                temperature=float(_required(generation_raw, "temperature")),
                top_p=float(_required(generation_raw, "top_p")),
                max_tokens=int(_required(generation_raw, "max_tokens")),
                reasoning_effort=str(_required(generation_raw, "reasoning_effort")),
                include_reasoning=bool(_required(generation_raw, "include_reasoning")),
                web_search=WebSearchConfig(
                    enabled=bool(_required(search_raw, "enabled")),
                    search_context_size=str(_required(search_raw, "search_context_size")),
                ),
            )

        return cls(
            run_name=str(_required(raw, "run_name")),
            provider=ProviderConfig(
                base_url=str(_required(provider_raw, "base_url")).rstrip("/"),
                api_key_env=str(_required(provider_raw, "api_key_env")),
                model_id=str(_required(provider_raw, "model_id")),
                timeout_seconds=float(_required(provider_raw, "timeout_seconds")),
                max_retries=int(_required(provider_raw, "max_retries")),
                retry_initial_seconds=float(
                    _required(provider_raw, "retry_initial_seconds")
                ),
                retry_max_seconds=float(_required(provider_raw, "retry_max_seconds")),
                pricing=PricingConfig(
                    input_per_million_usd=float(
                        _required(pricing_raw, "input_per_million_usd")
                    ),
                    cached_input_per_million_usd=float(
                        _required(pricing_raw, "cached_input_per_million_usd")
                    ),
                    output_per_million_usd=float(
                        _required(pricing_raw, "output_per_million_usd")
                    ),
                    as_of=str(_required(pricing_raw, "as_of")),
                ),
            ),
            target_generation=parse_generation("target_generation"),
            judge_generation=parse_generation("judge_generation"),
            capability_dataset=CapabilityDatasetConfig(
                name=str(_required(capability_raw, "name")),
                name_or_path=str(_required(capability_raw, "name_or_path")),
                subset=str(_required(capability_raw, "subset")),
                split=str(_required(capability_raw, "split")),
                question_column=str(_required(capability_raw, "question_column")),
                choices_column=str(_required(capability_raw, "choices_column")),
                answer_column=str(_required(capability_raw, "answer_column")),
                sample_limit=(
                    None
                    if capability_raw.get("sample_limit") is None
                    else int(capability_raw["sample_limit"])
                ),
            ),
            propensity_dataset=PropensityDatasetConfig(
                name=str(_required(propensity_raw, "name")),
                path=Path(str(_required(propensity_raw, "path"))),
                question_column=str(_required(propensity_raw, "question_column")),
                unsafe_probability_column=str(
                    _required(propensity_raw, "unsafe_probability_column")
                ),
                unsafe_probability_threshold=float(
                    _required(propensity_raw, "unsafe_probability_threshold")
                ),
                sample_limit=(
                    None
                    if propensity_raw.get("sample_limit") is None
                    else int(propensity_raw["sample_limit"])
                ),
            ),
            execution=ExecutionConfig(
                live=bool(_required(execution_raw, "live")),
                seed=int(_required(execution_raw, "seed")),
                max_total_estimated_cost_usd=(
                    None
                    if execution_raw.get("max_total_estimated_cost_usd") is None
                    else float(execution_raw["max_total_estimated_cost_usd"])
                ),
                progress_every=int(_required(execution_raw, "progress_every")),
                workers=int(_required(execution_raw, "workers")),
                resume=bool(_required(execution_raw, "resume")),
                reconnect_initial_seconds=float(
                    _required(execution_raw, "reconnect_initial_seconds")
                ),
                reconnect_max_seconds=float(
                    _required(execution_raw, "reconnect_max_seconds")
                ),
            ),
            output=OutputConfig(
                root_dir=Path(str(_required(output_raw, "root_dir"))),
                run_id=(
                    None if output_raw.get("run_id") is None else str(output_raw["run_id"])
                ),
                transcript_filename=str(_required(output_raw, "transcript_filename")),
                summary_csv_filename=str(_required(output_raw, "summary_csv_filename")),
                summary_json_filename=str(_required(output_raw, "summary_json_filename")),
            ),
            scoring=ScoringConfig(
                useful_score_threshold=float(
                    _required(scoring_raw, "useful_score_threshold")
                )
            ),
        )
