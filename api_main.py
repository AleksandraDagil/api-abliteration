from __future__ import annotations

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Hydra is required for API experiments. Run: "
        "python -m pip install -r requirements-api.txt"
    ) from exc

from safety_gap.api_evaluation.config import ApiExperimentConfig
from safety_gap.api_evaluation.runner import run_experiment


@hydra.main(
    config_path="safety_gap/hydra_config",
    config_name="api_config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Resolved Hydra configuration must be a dictionary")
    run_experiment(ApiExperimentConfig.from_mapping(raw))


if __name__ == "__main__":
    main()
