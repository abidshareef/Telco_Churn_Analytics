"""
Immutable data ingestion module.
Reads raw CSV datasets safely, computes SHA-256 integrity hashes, and never mutates raw files.
"""

import os
import hashlib
from typing import Tuple, Dict, Any
import pandas as pd

from telco_churn.utils.exceptions import DataIngestionError
from telco_churn.utils.logger import get_logger

logger = get_logger("telco_churn.data.ingestion")


def compute_file_sha256(file_path: str) -> str:
    """Computes the SHA-256 hex digest of a file to verify raw data immutability."""
    if not os.path.exists(file_path):
        raise DataIngestionError(
            f"Cannot compute checksum; file does not exist: {file_path}",
            context={"file_path": file_path},
        )
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ingest_raw_csv(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads the raw Telco Customer Churn CSV in read-only mode, returning a copy
    along with ingestion metadata (row count, column count, file size, SHA-256).
    """
    if not csv_path or not isinstance(csv_path, str):
        raise DataIngestionError("csv_path must be a non-empty string.")
    if not os.path.exists(csv_path):
        raise DataIngestionError(
            f"Raw dataset file not found at path: {csv_path}",
            context={"csv_path": csv_path},
        )
    if not os.path.isfile(csv_path):
        raise DataIngestionError(
            f"Expected a regular file at path, found directory or special file: {csv_path}",
            context={"csv_path": csv_path},
        )

    file_size = os.path.getsize(csv_path)
    if file_size == 0:
        raise DataIngestionError(
            f"Raw dataset file is empty (0 bytes): {csv_path}",
            context={"csv_path": csv_path},
        )

    sha256_hash = compute_file_sha256(csv_path)

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise DataIngestionError(
            f"Failed to parse CSV file at {csv_path}: {exc}",
            context={"csv_path": csv_path},
        ) from exc

    if df.empty or len(df) == 0:
        raise DataIngestionError(
            f"Parsed CSV contains 0 rows: {csv_path}",
            context={"csv_path": csv_path},
        )

    metadata = {
        "source_path": csv_path,
        "file_size_bytes": file_size,
        "sha256_checksum": sha256_hash,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
    }
    logger.info(
        f"Ingested raw dataset: {df.shape[0]} rows, {df.shape[1]} cols "
        f"(SHA-256: {sha256_hash[:12]}...)"
    )
    return df.copy(deep=True), metadata
