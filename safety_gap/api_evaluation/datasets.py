from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Any, Sequence

from safety_gap.api_evaluation.config import (
    CapabilityDatasetConfig,
    PropensityDatasetConfig,
)


@dataclass(frozen=True)
class CapabilitySample:
    sample_id: str
    question: str
    choices: list[str]
    answer_index: int


@dataclass(frozen=True)
class PropensitySample:
    sample_id: str
    question: str
    unsafe_probability: float


def _select_indices(length: int, limit: int | None, seed: int) -> list[int]:
    # A limited pilot intentionally selects the first N rows so its exact inputs
    # are obvious from the source datasets and reproducible without a sampler.
    indices = list(range(length))
    if limit is None or limit >= length:
        return indices
    if limit < 1:
        return []
    return indices[:limit]


def load_capability_samples(
    config: CapabilityDatasetConfig, seed: int
) -> list[CapabilitySample]:
    try:
        from datasets import load_dataset
    except (ImportError, ModuleNotFoundError) as exc:
        raise ModuleNotFoundError(
            "Hugging Face datasets is required. Install requirements-api.txt."
        ) from exc

    if not callable(load_dataset):
        raise ModuleNotFoundError(
            "The local datasets/ directory is shadowing the Hugging Face package. "
            "Install requirements-api.txt before running a live experiment."
        )

    dataset = load_dataset(
        config.name_or_path,
        name=config.subset,
        split=config.split,
    )
    samples: list[CapabilitySample] = []
    for index in _select_indices(len(dataset), config.sample_limit, seed):
        row: Any = dataset[index]
        samples.append(
            CapabilitySample(
                sample_id=str(index),
                question=str(row[config.question_column]),
                choices=[str(choice) for choice in row[config.choices_column]],
                answer_index=int(row[config.answer_column]),
            )
        )
    return samples


def load_propensity_samples(
    config: PropensityDatasetConfig, seed: int
) -> list[PropensitySample]:
    with config.path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    filtered: list[PropensitySample] = []
    for index, row in enumerate(rows):
        unsafe_probability = float(row[config.unsafe_probability_column])
        if unsafe_probability <= config.unsafe_probability_threshold:
            continue
        filtered.append(
            PropensitySample(
                sample_id=str(index),
                question=str(row[config.question_column]),
                unsafe_probability=unsafe_probability,
            )
        )

    selected = _select_indices(len(filtered), config.sample_limit, seed)
    return [filtered[index] for index in selected]


def choice_letters(choices: Sequence[str]) -> list[str]:
    if len(choices) > 26:
        raise ValueError("Multiple-choice samples may not contain more than 26 choices")
    return [chr(ord("A") + index) for index in range(len(choices))]
