#!/usr/bin/env python3
"""
CLI Entry Point: Runs the End-to-End Telco Customer Churn Data, Training,
Evaluation, and Artifact Generation Pipeline.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from telco_churn.config.settings import load_config
from telco_churn.pipelines.training_pipeline import ChurnTrainingOrchestrator
from telco_churn.utils.logger import get_logger

logger = get_logger("telco_churn.cli.run_pipeline")


def main() -> int:
    try:
        cfg = load_config(repo_root=REPO_ROOT)
        orchestrator = ChurnTrainingOrchestrator(config=cfg)
        report = orchestrator.run(generate_plots=True)
        roi = report["retention_roi_simulation"]
        logger.info(
            f"Summary: Best={report['best_model']} | "
            f"Captured={roi['True_Churners_Captured_in_Target']}/{roi['Total_Actual_Churners_in_Test']} "
            f"({roi['Target_Capture_Recall_Rate']*100:.2f}%) | Net Benefit=${roi['Net_Annual_Benefit_USD']:,.2f}"
        )
        return 0
    except Exception as exc:
        logger.error(f"Pipeline execution failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
