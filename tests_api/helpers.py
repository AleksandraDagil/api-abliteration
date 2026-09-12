from safety_gap.api_evaluation.config import (
    GenerationConfig,
    PricingConfig,
    ProviderConfig,
    WebSearchConfig,
)


def provider_config(max_retries: int = 1) -> ProviderConfig:
    return ProviderConfig(
        base_url="https://api.abliteration.ai/v1",
        api_key_env="ABLIT_KEY",
        model_id="abliterated-model-large",
        timeout_seconds=600,
        max_retries=max_retries,
        retry_initial_seconds=1,
        retry_max_seconds=30,
        pricing=PricingConfig(
            input_per_million_usd=5,
            cached_input_per_million_usd=1.25,
            output_per_million_usd=5,
            as_of="2026-07-29",
        ),
    )


def generation_config() -> GenerationConfig:
    return GenerationConfig(
        temperature=1.0,
        top_p=0.95,
        max_tokens=4096,
        reasoning_effort="max",
        include_reasoning=True,
        web_search=WebSearchConfig(enabled=True, search_context_size="high"),
    )
