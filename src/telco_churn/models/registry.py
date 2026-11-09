"""
Model artifact persistence and loading with integrity checks.
"""

import os
import pickle
from typing import Dict, Any

from telco_churn.models.classifiers import SoftVotingPipeline
from telco_churn.utils.exceptions import ModelPersistenceError, ModelNotFittedError
from telco_churn.utils.logger import get_logger

logger = get_logger("telco_churn.models.registry")


def save_model_artifact(
    pipeline: SoftVotingPipeline,
    artifact_path: str,
    metadata: Dict[str, Any] = None,
) -> str:
    """Serializes a fitted SoftVotingPipeline and its metadata to disk."""
    if pipeline is None or not getattr(pipeline, "is_fitted_", False):
        raise ModelNotFittedError("Cannot save an unfitted model pipeline.")
    if not artifact_path:
        raise ModelPersistenceError("artifact_path must be a non-empty string.")

    try:
        os.makedirs(os.path.dirname(os.path.abspath(artifact_path)), exist_ok=True)
        payload = {
            "pipeline": pipeline,
            "metadata": metadata or {},
            "format_version": "1.0.0",
        }
        with open(artifact_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved model artifact to {artifact_path}")
        return artifact_path
    except Exception as exc:
        raise ModelPersistenceError(
            f"Failed to serialize model artifact to {artifact_path}: {exc}",
            context={"artifact_path": artifact_path},
        ) from exc


def load_model_artifact(artifact_path: str) -> Dict[str, Any]:
    """Loads and validates a serialized model artifact from disk."""
    if not artifact_path or not os.path.exists(artifact_path):
        raise ModelPersistenceError(
            f"Model artifact file does not exist: {artifact_path}",
            context={"artifact_path": artifact_path},
        )
    if os.path.getsize(artifact_path) == 0:
        raise ModelPersistenceError(
            f"Model artifact file is empty (0 bytes): {artifact_path}",
            context={"artifact_path": artifact_path},
        )

    try:
        with open(artifact_path, "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        raise ModelPersistenceError(
            f"Corrupted or incompatible model artifact at {artifact_path}: {exc}",
            context={"artifact_path": artifact_path},
        ) from exc

    if not isinstance(payload, dict) or "pipeline" not in payload:
        raise ModelPersistenceError(
            f"Invalid model artifact structure in {artifact_path}: missing 'pipeline' key."
        )

    pipe = payload["pipeline"]
    if not getattr(pipe, "is_fitted_", False):
        raise ModelPersistenceError("Loaded model pipeline is not marked as fitted.")

    logger.info(f"Loaded model artifact from {artifact_path}")
    return payload
