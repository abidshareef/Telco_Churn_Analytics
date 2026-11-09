"""
Structured logging module with automatic secret redaction and configurable levels.
Never logs passwords, API keys, tokens, or raw sensitive credentials.
"""

import os
import re
import sys
import json
import logging
from datetime import datetime, timezone


SECRET_PATTERNS = [
    re.compile(
        r"(?i)(api[_-]?key|password|passwd|secret|token|authorization|bearer)\s*[:=]\s*['\"]?([^\s'\",;]+)['\"]?"
    ),
]


def redact_secrets(text: str) -> str:
    """Redacts any credential or token patterns from log messages."""
    if not isinstance(text, str):
        return str(text)
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(r"\1=***REDACTED***", redacted)
    return redacted


class SecretRedactingFilter(logging.Filter):
    """Logging filter that strips sensitive values before emission."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_secrets(record.msg)
        if record.args:
            record.args = tuple(
                redact_secrets(str(a)) if isinstance(a, str) else a for a in record.args
            )
        return True


class StructuredFormatter(logging.Formatter):
    """Formats log records as either structured JSON or clean human-readable console lines."""

    def __init__(self, use_json: bool = False):
        super().__init__()
        self.use_json = use_json

    def format(self, record: logging.LogRecord) -> str:
        msg = redact_secrets(record.getMessage())
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if self.use_json:
            payload = {
                "timestamp": ts,
                "level": record.levelname,
                "logger": record.name,
                "module": record.module,
                "message": msg,
            }
            return json.dumps(payload)
        return f"[{ts}] [{record.levelname:<7}] [{record.name}] {msg}"


def get_logger(name: str = "telco_churn", level: str = None) -> logging.Logger:
    """Returns a configured logger instance with secret redaction enabled."""
    env_level = (level or os.environ.get("CHURN_LOG_LEVEL", "INFO")).upper()
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    resolved_level = getattr(logging, env_level if env_level in valid_levels else "INFO")
    use_json = os.environ.get("CHURN_LOG_JSON", "false").lower() in ("1", "true", "yes")

    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(resolved_level)
        handler.addFilter(SecretRedactingFilter())
        handler.setFormatter(StructuredFormatter(use_json=use_json))
        logger.addHandler(handler)
    else:
        for h in logger.handlers:
            h.setLevel(resolved_level)

    return logger
