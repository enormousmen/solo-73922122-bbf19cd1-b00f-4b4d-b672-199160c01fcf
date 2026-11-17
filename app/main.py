"""FastAPI HTTP layer for the micro-batch image classifier.

Endpoints
---------
POST /predict                 submit an image + deadline_ms (from acceptance);
                              optional camera_id enables monitoring
GET  /orders/{order_id}       query status / result by order id
POST /orders/{order_id}/cancel  cancel a queued or in-flight order
GET  /stats                   capacity and state counters (demo/observability)
GET  /health                  liveness

Camera monitoring (/docs has the interactive forms for all of these)
--------------------------------------------------------------------
PUT    /cameras/{camera_id}/reference  set/update the 10-class reference (%)
GET    /cameras/{camera_id}/reference  read the current reference
DELETE /cameras/{camera_id}/reference  remove reference and its window
GET    /cameras/{camera_id}/distribution  window size, class shares, TVD, verdict
GET    /cameras                        summaries for all configured cameras

Interactive docs are served at /docs (Swagger UI).
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from .config import Settings
from .engine import COMPLETED, BatchEngine, Order
from .monitoring import NUM_CLASSES, MonitoringRegistry
from .preprocessing import preprocess


def order_view(o: Order) -> dict:
    remaining_ms = max(0, round((o.deadline_at - time.monotonic()) * 1000))
    return {
        "order_id": o.id,
        "status": o.state,
        "deadline_ms": o.deadline_ms,
        "remaining_ms": remaining_ms,
        "batch_id": o.batch_id,
        # Echoed for traceability; None when submitted without camera_id.
        "camera_id": o.camera_id,
        "camera_version": o.camera_version,
        "result": o.result if o.state == COMPLETED else None,
    }


class ReferenceBody(BaseModel):
    # strict=True + ge=0 on each item: rejects booleans/floats/strings and
    # negatives *per element*; the sum-to-100 rule is checked in the endpoint
    # so it can return a precise message.
    percentages: list[Annotated[int, Field(strict=True, ge=0)]] = Field(
        ...,
        description=f"Exactly {NUM_CLASSES} non-negative integer percentages summing to 100",
        examples=[[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]],
    )

    @field_validator("percentages")
    @classmethod
    def _check_length_and_sum(cls, value: list[int]) -> list[int]:
        if len(value) != NUM_CLASSES:
            raise ValueError(f"exactly {NUM_CLASSES} percentages are required, got {len(value)}")
        if sum(value) != 100:
            raise ValueError(f"percentages must sum to 100, got {sum(value)}")
        return value


class ReferenceView(BaseModel):
    camera_id: str
    version: int
    percentages: list[int]
    updated_at: str


class DistributionView(BaseModel):
    """Window snapshot; field meanings are documented in app/monitoring.py."""

    camera_id: str
    reference_version: int
    reference_percentages: list[int]
    reference_updated_at: str
    window_limit: int
    sample_count: int
    counts: list[int]
    proportions: list[float]
    percentages: list[float]
    tvd: float | None
    threshold: float
    status: str = Field(description="insufficient_sample | normal | anomaly")
    anomaly: bool
    message: str


class ReferenceWithDistribution(ReferenceView):
    distribution: DistributionView


class CameraList(BaseModel):
    count: int
    cameras: list[DistributionView]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    monitoring = MonitoringRegistry()

    def on_completed(order: Order) -> None:
        if order.result is None:  # defensive; completed orders always have a result
            return
        monitoring.record_completion(
            order.camera_id,
            order.camera_version,
            order.result["predicted_class"],
        )

    engine = BatchEngine(
        model_path=settings.model_path,
        max_capacity=settings.max_capacity,
        batch_size=settings.batch_size,
        batch_wait_ms=settings.batch_wait_ms,
        inference_delay_ms=settings.inference_delay_ms,
        on_completed=on_completed,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await engine.start()
        yield
        await engine.stop()

    app = FastAPI(
        title="Micro-batch Image Classifier",
        version="1.0.0",
        description=(
            "Time-limited image classification for multiple cameras. "
            "Single-process micro-batching: at most 16 queued/in-flight images, "
            "batches of up to 4, idle model waits up to 50 ms to fill a batch, "
            "one batch infers at a time. Cancelled/expired orders never yield results. "
            "Optional per-camera monitoring: set a 10-class reference (%) with "
            "PUT /cameras/{camera_id}/reference, pass camera_id on /predict, then "
            "read class shares / total-variation distance over the last 20 eligible "
            "completions from GET /cameras/{camera_id}/distribution "
            "(TVD > 0.3 => anomaly, < 20 completions => insufficient_sample)."
        ),
        lifespan=lifespan,
    )
    app.state.engine = engine
    app.state.settings = settings
    app.state.monitoring = monitoring

    @app.post("/predict", status_code=201)
    async def predict(
        file: UploadFile = File(...),
        deadline_ms: int = Form(...),
        camera_id: str | None = Form(
            None,
            description="Optional camera identifier; enables distribution monitoring",
        ),
    ):
        """Submit an image with a deadline counted from acceptance.

        `camera_id` is optional: without it the order is classified exactly as
        before and never enters any monitoring window.  When it is given but no
        reference is configured for that camera, the order is still accepted
        and classified normally, but is not eligible for monitoring.
        """
        if deadline_ms <= 0:
            raise HTTPException(400, "deadline_ms must be a positive integer")
        camera_id = camera_id.strip() if camera_id is not None else None
        if camera_id == "":
            camera_id = None
        data = await file.read()
        try:
            tensor = preprocess(data)
        except ValueError as exc:
            raise HTTPException(400, f"invalid image: {exc}") from exc
        # Snapshot the reference version at acceptance, before the order can
        # be batched.  A later reference update therefore cannot make this
        # frame eligible for the new window.
        camera_version = monitoring.accepted_version(camera_id) if camera_id else None
        order = engine.submit(tensor, deadline_ms, camera_id, camera_version)
        if order is None:
            raise HTTPException(
                429,
                f"capacity full: at most {engine.max_capacity} images queued or in-flight",
            )
        return order_view(order)

    @app.get("/orders/{order_id}")
    async def get_order(order_id: str):
        order = engine.get(order_id)
        if order is None:
            raise HTTPException(404, "unknown order_id")
        return order_view(order)

    @app.post("/orders/{order_id}/cancel")
    async def cancel_order(order_id: str):
        order = engine.cancel(order_id)
        if order is None:
            raise HTTPException(404, "unknown order_id")
        if order.state == COMPLETED:
            raise HTTPException(409, "order already completed; cannot cancel")
        return order_view(order)

    @app.get("/stats")
    async def stats():
        return engine.stats()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ------------------------------------------------------------------ #
    # camera monitoring
    # ------------------------------------------------------------------ #
    @app.get("/cameras", response_model=CameraList)
    async def list_cameras():
        """Summaries for every camera with a configured reference."""
        cameras = monitoring.list_cameras()
        return {"count": len(cameras), "cameras": cameras}

    @app.put("/cameras/{camera_id}/reference", response_model=ReferenceWithDistribution)
    async def put_reference(camera_id: str, body: ReferenceBody):
        """Create or update a camera's reference distribution.

        Updating replaces the window immediately, so a previously reported
        anomaly disappears at once.  Orders accepted under the previous
        version are no longer eligible even if they complete afterwards.
        """
        camera_id = camera_id.strip()
        if not camera_id:
            raise HTTPException(400, "camera_id must not be empty")
        reference = monitoring.set_reference(camera_id, body.percentages)
        return {**reference.view(), "distribution": monitoring.snapshot(camera_id)}

    @app.get("/cameras/{camera_id}/reference", response_model=ReferenceView)
    async def get_reference(camera_id: str):
        reference = monitoring.get_reference(camera_id)
        if reference is None:
            raise HTTPException(404, "unknown camera_id")
        return reference.view()

    @app.delete("/cameras/{camera_id}/reference")
    async def delete_reference(camera_id: str):
        """Remove a reference (and its window).  Frames already accepted keep
        completing normally but are no longer monitored."""
        if not monitoring.remove_reference(camera_id):
            raise HTTPException(404, "unknown camera_id")
        return {"camera_id": camera_id, "deleted": True}

    @app.get("/cameras/{camera_id}/distribution", response_model=DistributionView)
    async def get_distribution(camera_id: str):
        """Class shares over the last 20 eligible completions + TVD verdict.

        `status` is one of `insufficient_sample` (< 20 eligible completions),
        `anomaly` (TVD > 0.3) or `normal`.
        """
        snapshot = monitoring.snapshot(camera_id)
        if snapshot is None:
            raise HTTPException(404, "unknown camera_id")
        return snapshot

    return app


app = create_app()
