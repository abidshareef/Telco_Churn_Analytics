#!/usr/bin/env python3
"""
CLI Entry Point: Starts the Production HTTP REST API Service for Real-Time &
Batch Customer Churn Prediction.
"""

import os
import sys
import json
import signal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from telco_churn.config.settings import load_config
from telco_churn.models.registry import load_model_artifact
from telco_churn.services.retention import ChurnInferenceService
from telco_churn.services.api_server import build_http_server
from telco_churn.utils.logger import get_logger

logger = get_logger("telco_churn.cli.serve_api")


def main() -> int:
    try:
        cfg = load_config(repo_root=REPO_ROOT)
        artifact = load_model_artifact(cfg.model_artifact_path)
        pipeline = artifact["pipeline"]
        meta = artifact.get("metadata", {})

        bench_json = os.path.join(cfg.reports_dir, "benchmark_results.json")
        bench_meta = {}
        if os.path.exists(bench_json):
            with open(bench_json, "r", encoding="utf-8") as f:
                full_rep = json.load(f)
                bench_meta = {
                    "best_model": full_rep.get("best_model"),
                    "optimal_threshold": full_rep.get("optimal_threshold"),
                    "roi_summary": full_rep.get("retention_roi_simulation", {}),
                }

        service = ChurnInferenceService(
            pipeline=pipeline,
            model_name=meta.get("model_name", "Proposed Hybrid Soft-Voting Ensemble"),
            low_risk_max=cfg.low_risk_max,
            medium_risk_max=cfg.medium_risk_max,
            max_batch_size=cfg.max_batch_size,
        )

        server = build_http_server(cfg.api_host, cfg.api_port, service, bench_meta)

        def handle_sigterm(signum, frame):
            logger.info("Received shutdown signal; stopping HTTP server gracefully...")
            server.shutdown()

        signal.signal(signal.SIGINT, handle_sigterm)
        signal.signal(signal.SIGTERM, handle_sigterm)

        logger.info(f"Starting Skybrisk Churn REST API on http://{cfg.api_host}:{cfg.api_port}")
        server.serve_forever()
        return 0
    except Exception as exc:
        logger.error(f"Failed to start REST API server: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
