"""API-based Safety Gap evaluations.

This package is intentionally independent of the original GPU/vLLM pipeline so
that hosted-model experiments can run without installing the local-model stack.
"""

from safety_gap.api_evaluation.config import ApiExperimentConfig

__all__ = ["ApiExperimentConfig"]
