"""Single-process micro-batch inference engine.

Concurrency model
-----------------
* Everything except the blocking model forward pass runs on the FastAPI
  asyncio event loop, so no locks are needed for the shared state below.
* Exactly one worker task forms batches and exactly one executor thread runs
  the TorchScript model, therefore at most ONE batch is ever actually
  inferring at any moment.
* Capacity accounting: an order occupies a slot while it is queued AND while
  its batch is computing (even if it was cancelled or expired mid-flight).
  The slot is released only when the batch compute actually finishes.

State machine per order
-----------------------
queued -> processing -> completed
queued -> cancelled            (cancelled before entering a batch; never runs)
queued -> expired              (deadline passed while queued; never runs)
processing -> cancelled        (marked immediately, result discarded, slot held)
processing -> expired          (deadline passed mid-flight; result discarded)
processing -> failed           (model raised; defensive only)
Late results are written back ONLY to orders still in `processing` whose
deadline has not passed, so a late result can never revive a cancelled or
expired order.

Monitoring hook
---------------
`on_completed` is called exactly once for every order that transitions to
`completed` (never for cancelled, expired or failed orders).  It runs on the
event-loop thread, so the monitoring registry needs no locks.  Camera
identity and the reference version in force at acceptance are stored on the
order; the registry decides whether the completion is still eligible.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

log = logging.getLogger("batch-engine")

QUEUED = "queued"
PROCESSING = "processing"
COMPLETED = "completed"
CANCELLED = "cancelled"
EXPIRED = "expired"
FAILED = "failed"


# eq=False: orders are compared by identity. The default dataclass __eq__
# would compare tensors element-wise and raise on multi-element tensors.
@dataclass(eq=False)
class Order:
    id: str
    tensor: torch.Tensor
    deadline_ms: int
    deadline_at: float  # time.monotonic() at which the order expires
    state: str = QUEUED
    result: Optional[dict] = None
    batch_id: Optional[str] = None
    created_at: float = field(default_factory=time.monotonic)
    # Monitoring metadata: None when submitted without camera_id (or when the
    # camera had no reference at acceptance, in which case the frame is
    # classified normally but can never enter a monitoring window).
    camera_id: Optional[str] = None
    camera_version: Optional[int] = None


class BatchEngine:
    def __init__(
        self,
        model_path: str,
        max_capacity: int = 16,
        batch_size: int = 4,
        batch_wait_ms: int = 50,
        inference_delay_ms: int = 0,
        on_completed: Optional[Callable[["Order"], None]] = None,
    ) -> None:
        self.max_capacity = max_capacity
        self.batch_size = batch_size
        self.batch_wait_s = batch_wait_ms / 1000.0
        self.inference_delay_s = inference_delay_ms / 1000.0
        # Called on the event loop for each order that reaches `completed`.
        self.on_completed = on_completed

        # Single-threaded CPU inference keeps results deterministic and makes
        # the "one batch at a time" guarantee easy to reason about.
        torch.set_num_threads(1)
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()

        self._pending: deque[Order] = deque()
        self._orders: dict[str, Order] = {}
        self._inflight = 0  # orders inside the batch currently being computed
        self._not_empty = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self._worker: Optional[asyncio.Task] = None
        # Counters for /stats and tests.
        self.batches_started = 0
        self.last_batch_size = 0

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        self._worker = asyncio.create_task(self._run(), name="batch-worker")

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------ #
    # helpers (event-loop thread only)
    # ------------------------------------------------------------------ #
    def _purge_expired_pending(self) -> None:
        """Mark queued orders whose deadline passed; they never enter the model."""
        now = time.monotonic()
        kept: list[Order] = []
        for o in self._pending:
            if o.state == QUEUED and o.deadline_at <= now:
                o.state = EXPIRED
            else:
                kept.append(o)
        if len(kept) != len(self._pending):
            self._pending.clear()
            self._pending.extend(kept)

    def _refresh_state(self, o: Order) -> None:
        """Lazy expiry when an order is observed."""
        if o.state in (QUEUED, PROCESSING) and o.deadline_at <= time.monotonic():
            if o.state == QUEUED:
                try:
                    self._pending.remove(o)
                except ValueError:
                    pass
            o.state = EXPIRED

    def capacity_used(self) -> int:
        self._purge_expired_pending()
        return len(self._pending) + self._inflight

    def stats(self) -> dict:
        self._purge_expired_pending()
        counts: dict[str, int] = {}
        for o in self._orders.values():
            self._refresh_state(o)
            counts[o.state] = counts.get(o.state, 0) + 1
        return {
            "capacity_used": len(self._pending) + self._inflight,
            "max_capacity": self.max_capacity,
            "queued": len(self._pending),
            "inflight": self._inflight,
            "batches_started": self.batches_started,
            "last_batch_size": self.last_batch_size,
            "orders": counts,
        }

    # ------------------------------------------------------------------ #
    # public API used by the HTTP layer
    # ------------------------------------------------------------------ #
    def submit(
        self,
        tensor: torch.Tensor,
        deadline_ms: int,
        camera_id: Optional[str] = None,
        camera_version: Optional[int] = None,
    ) -> Optional[Order]:
        """Accept an image, or return None when at capacity (16 by default).

        `camera_id`/`camera_version` are monitoring metadata only: they are
        the camera and reference version in force at acceptance.  They do not
        affect capacity, batching, deadlines or the result.
        """
        if self.capacity_used() >= self.max_capacity:
            return None
        order = Order(
            id=uuid.uuid4().hex[:12],
            tensor=tensor,
            deadline_ms=deadline_ms,
            deadline_at=time.monotonic() + deadline_ms / 1000.0,
            camera_id=camera_id,
            camera_version=camera_version,
        )
        self._orders[order.id] = order
        self._pending.append(order)
        self._not_empty.set()
        return order

    def get(self, order_id: str) -> Optional[Order]:
        o = self._orders.get(order_id)
        if o is not None:
            self._refresh_state(o)
        return o

    def cancel(self, order_id: str) -> Optional[Order]:
        o = self._orders.get(order_id)
        if o is None:
            return None
        self._refresh_state(o)
        if o.state == QUEUED:
            try:
                self._pending.remove(o)
            except ValueError:
                pass
            o.state = CANCELLED
        elif o.state == PROCESSING:
            # Mark immediately. The slot stays occupied until the batch
            # compute actually ends; no new batch is started because of it.
            o.state = CANCELLED
        return o

    # ------------------------------------------------------------------ #
    # batching worker
    # ------------------------------------------------------------------ #
    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await self._not_empty.wait()
            if not self._pending:
                # The event can still be set after a batch was drained. With a
                # zero batch window the inner loop below never clears it, so
                # clear it here to avoid spinning on an empty queue.
                self._not_empty.clear()
                continue

            # Batch-formation window: the idle model waits at most
            # `batch_wait_s` for more work, up to `batch_size` images.
            batch: list[Order] = []
            window_end = loop.time() + self.batch_wait_s
            while len(batch) < self.batch_size:
                self._purge_expired_pending()
                while self._pending and len(batch) < self.batch_size:
                    batch.append(self._pending.popleft())
                if len(batch) >= self.batch_size:
                    break
                remaining = window_end - loop.time()
                if remaining <= 0:
                    break
                self._not_empty.clear()
                try:
                    await asyncio.wait_for(self._not_empty.wait(), remaining)
                except asyncio.TimeoutError:
                    break

            # Final gate: cancelled/expired orders never enter the model.
            now = time.monotonic()
            batch = [o for o in batch if o.state == QUEUED and o.deadline_at > now]
            if not batch:
                continue

            batch_id = uuid.uuid4().hex[:8]
            for o in batch:
                o.state = PROCESSING
                o.batch_id = batch_id
            self._inflight = len(batch)
            self.batches_started += 1
            self.last_batch_size = len(batch)
            tensors = torch.stack([o.tensor for o in batch])

            try:
                scores = await loop.run_in_executor(self._executor, self._infer, tensors)
            except Exception:  # noqa: BLE001 - defensive: model failure must not kill the worker
                log.exception("inference failed for batch %s", batch_id)
                for o in batch:
                    if o.state == PROCESSING:
                        o.state = FAILED
                self._inflight = 0
                continue

            now = time.monotonic()
            for o, sc in zip(batch, scores):
                if o.state != PROCESSING:
                    # Cancelled or expired mid-flight: result is discarded and
                    # the late result must NOT revive the order.
                    continue
                if o.deadline_at <= now:
                    o.state = EXPIRED
                    continue
                o.state = COMPLETED
                predicted = max(range(len(sc)), key=lambda i: sc[i])
                o.result = {"predicted_class": predicted, "scores": sc}
                if self.on_completed is not None:
                    try:
                        self.on_completed(o)
                    except Exception:  # noqa: BLE001 - observability must not break inference
                        log.exception("completion hook failed for order %s", o.id)
            self._inflight = 0

    # ------------------------------------------------------------------ #
    # runs on the single executor thread — never touches shared state
    # ------------------------------------------------------------------ #
    def _infer(self, batch_tensor: torch.Tensor) -> list[list[float]]:
        if self.inference_delay_s > 0:
            time.sleep(self.inference_delay_s)  # demo: simulate a slow model
        with torch.inference_mode():
            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
        return [[float(v) for v in row] for row in probs]
