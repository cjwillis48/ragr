"""OpenTelemetry tracing setup.

When OTEL_EXPORTER_OTLP_ENDPOINT is unset, all calls here are no-ops and
trace.get_tracer(...) returns OTel's default NoOpTracer — so manual span
context managers in service code remain zero-cost.
"""

import logging
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app import __version__

logger = logging.getLogger("ragr.telemetry")

tracer = trace.get_tracer("ragr")


def setup_tracing(service_name: str) -> None:
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return

    resource = Resource.create({"service.name": service_name, "service.version": __version__})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
    trace.set_tracer_provider(provider)
    logger.info("tracing_enabled", extra={"service": service_name, "endpoint": endpoint})


def instrument_fastapi(app) -> None:
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # lazy: avoid import cost when tracing is off

    FastAPIInstrumentor.instrument_app(app)
