"""
Production HTTP REST API Server for Real-Time & Batch Churn Prediction.
Endpoints:
  - GET  /health         : Liveness probe
  - GET  /ready          : Readiness probe (verifies model is loaded and fitted)
  - GET  /metrics        : Operational telemetry & model benchmark summary
  - POST /predict        : Score a single subscriber JSON object
  - POST /predict/batch  : Score a batch of subscriber JSON objects ({"records": [...]})
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Any, Optional

from telco_churn.services.retention import ChurnInferenceService
from telco_churn.utils.exceptions import ChurnAppError, SchemaValidationError, InferenceError
from telco_churn.utils.logger import get_logger

logger = get_logger("telco_churn.services.api")


def create_request_handler(
    inference_service: ChurnInferenceService,
    benchmark_metadata: Optional[Dict[str, Any]] = None,
):
    """Factory creating a request handler bound to the active ChurnInferenceService."""

    class ChurnAPIRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format_str: str, *args: Any) -> None:
            logger.info("%s - %s" % (self.address_string(), format_str % args))

        def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> Any:
            content_len = int(self.headers.get("Content-Length", 0))
            if content_len <= 0:
                raise InferenceError("Request body is empty.")
            if content_len > 10 * 1024 * 1024:
                raise InferenceError("Request payload exceeds 10 MB limit.")
            raw_bytes = self.rfile.read(content_len)
            try:
                return json.loads(raw_bytes.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in request body: {exc}") from exc

        def do_GET(self) -> None:
            path = self.path.split("?")[0].rstrip("/") or "/"
            if path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "healthy",
                        "service": "skybrisk-telco-churn-api",
                        "version": "1.0.0",
                    },
                )
            elif path == "/ready":
                is_ready = (
                    inference_service is not None
                    and inference_service.pipeline is not None
                    and getattr(inference_service.pipeline, "is_fitted_", False)
                )
                status_code = 200 if is_ready else 503
                self._send_json(
                    status_code,
                    {
                        "ready": is_ready,
                        "model_name": inference_service.model_name if is_ready else None,
                    },
                )
            elif path == "/metrics":
                self._send_json(
                    200,
                    {
                        "requests_served": inference_service.requests_served,
                        "records_scored": inference_service.records_scored,
                        "model_name": inference_service.model_name,
                        "optimal_threshold": round(
                            float(inference_service.pipeline.optimal_threshold), 4
                        ),
                        "benchmark_summary": benchmark_metadata or {},
                    },
                )
            else:
                self._send_json(
                    404,
                    {
                        "error_code": "NOT_FOUND",
                        "message": f"Endpoint '{path}' not found.",
                    },
                )

        def do_POST(self) -> None:
            path = self.path.split("?")[0].rstrip("/")
            try:
                payload = self._read_json_body()
                if path == "/predict":
                    if not isinstance(payload, dict):
                        raise InferenceError("POST /predict expects a single JSON object.")
                    result = inference_service.predict_records([payload])
                    self._send_json(
                        200,
                        {
                            "status": "success",
                            "model_name": result["model_name"],
                            "latency_ms": result["latency_ms"],
                            "prediction": result["predictions"][0],
                        },
                    )
                elif path == "/predict/batch":
                    if not isinstance(payload, dict) or "records" not in payload:
                        raise InferenceError(
                            "POST /predict/batch expects a JSON object with a 'records' list."
                        )
                    result = inference_service.predict_records(payload["records"])
                    self._send_json(
                        200,
                        {
                            "status": "success",
                            **result,
                        },
                    )
                else:
                    self._send_json(
                        404,
                        {
                            "error_code": "NOT_FOUND",
                            "message": f"Endpoint '{path}' not found.",
                        },
                    )
            except ValueError as exc:
                self._send_json(
                    400,
                    {
                        "error_code": "BAD_REQUEST_JSON",
                        "message": str(exc),
                    },
                )
            except (SchemaValidationError, InferenceError) as exc:
                self._send_json(422, exc.to_dict())
            except ChurnAppError as exc:
                logger.error(f"Application error on {path}: {exc.message}")
                self._send_json(500, exc.to_dict())
            except Exception as exc:
                logger.error(f"Unexpected error on {path}: {exc}")
                self._send_json(
                    500,
                    {
                        "error_code": "INTERNAL_SERVER_ERROR",
                        "message": "An unexpected internal error occurred.",
                    },
                )

    return ChurnAPIRequestHandler


def build_http_server(
    host: str,
    port: int,
    inference_service: ChurnInferenceService,
    benchmark_metadata: Optional[Dict[str, Any]] = None,
) -> ThreadingHTTPServer:
    """Instantiates a multi-threaded HTTP server bound to (host, port)."""
    handler_cls = create_request_handler(inference_service, benchmark_metadata)
    return ThreadingHTTPServer((host, port), handler_cls)
