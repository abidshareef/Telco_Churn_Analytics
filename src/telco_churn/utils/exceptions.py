"""
Custom domain exception hierarchy for the Telco Churn Analytics platform.
Ensures every failure mode fails clearly with actionable diagnostic context.
"""

from typing import Optional, Dict, Any


class ChurnAppError(Exception):
    """Base exception for all application-specific errors in telco_churn."""

    def __init__(
        self,
        message: str,
        error_code: str = "CHURN_APP_ERROR",
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
        }


class ConfigurationError(ChurnAppError):
    """Raised when configuration files or environment variables are missing or invalid."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="CONFIGURATION_ERROR", context=context)


class DataIngestionError(ChurnAppError):
    """Raised when raw dataset files cannot be read, are empty, or fail integrity checks."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="DATA_INGESTION_ERROR", context=context)


class SchemaValidationError(ChurnAppError):
    """Raised when input data violates schema contracts, types, or domain bounds."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="SCHEMA_VALIDATION_ERROR", context=context)


class PreprocessingError(ChurnAppError):
    """Raised when data cleaning, feature engineering, or scaling encounters invalid state."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="PREPROCESSING_ERROR", context=context)


class ModelNotFittedError(ChurnAppError):
    """Raised when transform() or predict_proba() is called before fit()."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="MODEL_NOT_FITTED_ERROR", context=context)


class ModelPersistenceError(ChurnAppError):
    """Raised when saving or loading a serialized model artifact fails."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="MODEL_PERSISTENCE_ERROR", context=context)


class InferenceError(ChurnAppError):
    """Raised when real-time or batch prediction fails due to invalid payload or model error."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="INFERENCE_ERROR", context=context)
