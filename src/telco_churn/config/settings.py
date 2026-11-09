"""
Centralized configuration management separating settings from application logic.
Loads defaults from JSON config and applies environment variable overrides.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from telco_churn.utils.exceptions import ConfigurationError


@dataclass
class AppConfig:
    project_name: str = "skybrisk-telco-churn-analytics"
    version: str = "1.0.0"
    random_seed: int = 42
    test_size: float = 0.20
    cv_folds: int = 5
    raw_data_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    model_artifact_path: str = "models_store/churn_ensemble_model.pkl"
    reports_dir: str = "reports"
    figures_dir: str = "reports/figures"
    low_risk_max: float = 0.35
    medium_risk_max: float = 0.65
    intervention_cost_usd: float = 45.0
    retention_save_rate: float = 0.40
    ensemble_weights: Dict[str, float] = field(
        default_factory=lambda: {"lr": 3.2, "gbm": 2.0, "et": 1.2, "mlp": 1.2}
    )
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    max_batch_size: int = 1000

    def validate(self) -> "AppConfig":
        """Validates configuration invariants and raises ConfigurationError on invalid values."""
        if not (0.05 <= self.test_size <= 0.50):
            raise ConfigurationError(
                f"test_size must be between 0.05 and 0.50, got {self.test_size}",
                context={"test_size": self.test_size},
            )
        if self.cv_folds < 2 or self.cv_folds > 20:
            raise ConfigurationError(
                f"cv_folds must be between 2 and 20, got {self.cv_folds}",
                context={"cv_folds": self.cv_folds},
            )
        if not (0.0 < self.low_risk_max < self.medium_risk_max < 1.0):
            raise ConfigurationError(
                "Risk tier thresholds must satisfy 0 < low_risk_max < medium_risk_max < 1",
                context={
                    "low_risk_max": self.low_risk_max,
                    "medium_risk_max": self.medium_risk_max,
                },
            )
        if self.intervention_cost_usd < 0.0:
            raise ConfigurationError("intervention_cost_usd cannot be negative")
        if not (0.0 < self.retention_save_rate <= 1.0):
            raise ConfigurationError("retention_save_rate must be in (0.0, 1.0]")
        if not (1 <= self.api_port <= 65535):
            raise ConfigurationError(
                f"api_port must be between 1 and 65535, got {self.api_port}",
                context={"api_port": self.api_port},
            )
        if self.max_batch_size < 1:
            raise ConfigurationError("max_batch_size must be >= 1")
        return self


def load_config(config_path: Optional[str] = None, repo_root: Optional[str] = None) -> AppConfig:
    """
    Loads configuration from JSON file (if provided or found) and overrides with
    CHURN_* environment variables.
    """
    base_dir = repo_root or os.getcwd()
    resolved_cfg_path = config_path or os.environ.get(
        "CHURN_CONFIG_PATH", os.path.join(base_dir, "configs", "default_config.json")
    )

    raw_dict: Dict[str, Any] = {}
    if resolved_cfg_path and os.path.exists(resolved_cfg_path):
        try:
            with open(resolved_cfg_path, "r", encoding="utf-8") as f:
                raw_dict = json.load(f)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                f"Failed to parse JSON configuration at {resolved_cfg_path}: {exc}",
                context={"config_path": resolved_cfg_path},
            ) from exc
    elif config_path is not None:
        raise ConfigurationError(
            f"Explicitly requested config file does not exist: {config_path}",
            context={"config_path": config_path},
        )

    paths = raw_dict.get("paths", {})
    tiers = raw_dict.get("risk_tiers", {})
    roi = raw_dict.get("roi_simulation", {})
    api = raw_dict.get("api", {})

    def _resolve_path(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(base_dir, p)

    try:
        cfg = AppConfig(
            project_name=raw_dict.get("project_name", "skybrisk-telco-churn-analytics"),
            version=raw_dict.get("version", "1.0.0"),
            random_seed=int(os.environ.get("CHURN_RANDOM_SEED", raw_dict.get("random_seed", 42))),
            test_size=float(os.environ.get("CHURN_TEST_SIZE", raw_dict.get("test_size", 0.20))),
            cv_folds=int(os.environ.get("CHURN_CV_FOLDS", raw_dict.get("cv_folds", 5))),
            raw_data_path=_resolve_path(
                os.environ.get(
                    "CHURN_RAW_DATA_PATH",
                    paths.get("raw_data", "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
                )
            ),
            interim_dir=_resolve_path(
                os.environ.get("CHURN_INTERIM_DIR", paths.get("interim_dir", "data/interim"))
            ),
            processed_dir=_resolve_path(
                os.environ.get("CHURN_PROCESSED_DIR", paths.get("processed_dir", "data/processed"))
            ),
            model_artifact_path=_resolve_path(
                os.environ.get(
                    "CHURN_MODEL_ARTIFACT_PATH",
                    paths.get("model_artifact", "models_store/churn_ensemble_model.pkl"),
                )
            ),
            reports_dir=_resolve_path(
                os.environ.get("CHURN_REPORTS_DIR", paths.get("reports_dir", "reports"))
            ),
            figures_dir=_resolve_path(
                paths.get("figures_dir", "reports/figures")
            ),
            low_risk_max=float(tiers.get("low_max", 0.35)),
            medium_risk_max=float(tiers.get("medium_max", 0.65)),
            intervention_cost_usd=float(roi.get("intervention_cost_usd", 45.0)),
            retention_save_rate=float(roi.get("retention_save_rate", 0.40)),
            ensemble_weights=raw_dict.get(
                "ensemble_weights", {"lr": 3.2, "gbm": 2.0, "et": 1.2, "mlp": 1.2}
            ),
            api_host=os.environ.get("CHURN_API_HOST", api.get("host", "0.0.0.0")),
            api_port=int(os.environ.get("CHURN_API_PORT", api.get("port", 8080))),
            max_batch_size=int(api.get("max_batch_size", 1000)),
        )
    except ValueError as exc:
        raise ConfigurationError(f"Invalid type in configuration or environment variable: {exc}") from exc

    return cfg.validate()
