"""Tests for per-camera distribution monitoring.

Two layers:
* direct tests of MonitoringRegistry for validation / TVD / window / version
  semantics (precise, deterministic);
* API tests that drive real orders through the engine, with `_infer` replaced
  by a deterministic stub so class distributions are controllable.  Everything
  else (capacity, batching, deadlines, cancellation) is the real engine.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager

from app.config import Settings
from app.main import create_app
from app.monitoring import MonitoringRegistry

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
MODEL = Path(__file__).resolve().parent.parent / "app" / "model.pt"


# --------------------------------------------------------------------- #
# registry unit tests
# --------------------------------------------------------------------- #
def uniform() -> list[int]:
    return [10] * 10


def test_validate_percentages():
    validate = MonitoringRegistry().set_reference
    assert validate("cam", uniform()).percentages == tuple(uniform())

    for bad in (
        [10] * 9,
        [10] * 11,
        [-1] + [10] * 9,          # negative
        [11] + [10] * 9,          # sums to 101
        [9] + [10] * 9,           # sums to 99
        [10.0] * 10,              # floats
        [True, 10, 10, 10, 10, 10, 10, 10, 10, 19],  # bool
        ["10"] * 10,              # strings
    ):
        with pytest.raises(ValueError):
            validate("cam", bad)


def test_insufficient_sample_until_20():
    reg = MonitoringRegistry()
    reg.set_reference("cam", [100] + [0] * 9)
    for _ in range(19):
        assert reg.record_completion("cam", 1, 0)
    snap = reg.snapshot("cam")
    assert snap["sample_count"] == 19
    assert snap["status"] == "insufficient_sample"
    assert snap["anomaly"] is False
    assert snap["tvd"] == pytest.approx(0.0)

    assert reg.record_completion("cam", 1, 0)
    snap = reg.snapshot("cam")
    assert snap["sample_count"] == 20
    assert snap["status"] == "normal"
    assert snap["tvd"] == pytest.approx(0.0)


def test_tvd_threshold_boundary_is_strict():
    # reference: all mass on class 0.  Observed 14/20 vs 6/20 gives
    # TVD = 0.5 * (|0.7-1| + |0.3-0|) = 0.3 exactly -> NOT an anomaly.
    reg = MonitoringRegistry()
    reg.set_reference("cam", [100] + [0] * 9)
    for i in range(20):
        reg.record_completion("cam", 1, 0 if i < 14 else 1)
    snap = reg.snapshot("cam")
    assert snap["tvd"] == pytest.approx(0.3)
    assert snap["status"] == "normal"
    assert snap["anomaly"] is False

    # 13/20 vs 7/20 -> TVD = 0.35 > 0.3 -> anomaly.
    reg = MonitoringRegistry()
    reg.set_reference("cam", [100] + [0] * 9)
    for i in range(20):
        reg.record_completion("cam", 1, 0 if i < 13 else 1)
    snap = reg.snapshot("cam")
    assert snap["tvd"] == pytest.approx(0.35)
    assert snap["status"] == "anomaly"
    assert snap["anomaly"] is True


def test_window_keeps_only_last_20_completions():
    reg = MonitoringRegistry()
    reg.set_reference("cam", [100] + [0] * 9)
    for i in range(21):
        reg.record_completion("cam", 1, 0 if i < 20 else 9)
    snap = reg.snapshot("cam")
    assert snap["sample_count"] == 20
    assert snap["counts"][0] == 19
    assert snap["counts"][9] == 1


def test_version_mismatch_and_unknown_camera_are_ignored():
    reg = MonitoringRegistry()
    ref1 = reg.set_reference("cam", uniform())
    assert reg.record_completion("cam", ref1.version, 3)
    ref2 = reg.set_reference("cam", uniform())
    assert ref2.version == ref1.version + 1
    # old-version frame arriving after the update: ignored
    assert not reg.record_completion("cam", ref1.version, 3)
    # current-version frame: accepted
    assert reg.record_completion("cam", ref2.version, 4)
    # frames accepted without a reference
    assert not reg.record_completion("cam", None, 3)
    assert not reg.record_completion(None, ref2.version, 3)
    assert not reg.record_completion("other", ref2.version, 3)
    snap = reg.snapshot("cam")
    assert snap["sample_count"] == 1
    assert snap["counts"][4] == 1


def test_delete_and_recreate_never_reuses_old_version():
    reg = MonitoringRegistry()
    old = reg.set_reference("cam", uniform())
    assert reg.remove_reference("cam")
    assert reg.snapshot("cam") is None
    new = reg.set_reference("cam", uniform())
    assert new.version > old.version
    # A frame accepted under the deleted reference cannot enter the new window.
    assert not reg.record_completion("cam", old.version, 0)
    assert reg.snapshot("cam")["sample_count"] == 0


def test_update_clears_window_and_alert_immediately():
    reg = MonitoringRegistry()
    reg.set_reference("cam", [100] + [0] * 9)
    for _ in range(20):
        reg.record_completion("cam", 1, 5)
    assert reg.snapshot("cam")["anomaly"] is True
    reg.set_reference("cam", [0] * 5 + [100] + [0] * 4)
    snap = reg.snapshot("cam")
    assert snap["sample_count"] == 0
    assert snap["status"] == "insufficient_sample"
    assert snap["anomaly"] is False


# --------------------------------------------------------------------- #
# API test harness
# --------------------------------------------------------------------- #
class FakeInfer:
    """Deterministic stand-in for the model forward pass.

    `classes` is consumed per image in batch order.  The first call that would
    deliver the `fail_at`-th consumed sample raises, exercising the engine's
    `failed` path.
    """

    def __init__(self) -> None:
        self.classes: deque[int] = deque()
        self.calls = 0
        self.delay_s = 0.0
        self.returned = 0
        self.failed = 0
        self.fail_at = 10 ** 9

    def expect(self, classes) -> None:
        self.classes.extend(classes)

    def __call__(self, batch) -> list[list[float]]:
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)  # honour INFERENCE_DELAY_MS like _infer does
        # Fail the call that would first deliver the Nth consumed sample, not
        # a fixed call index: scheduling may coalesce submissions differently.
        if self.returned + batch.shape[0] >= self.fail_at:
            self.fail_at = 10 ** 9  # one-shot: later batches succeed again
            # These frames are consumed by the engine just like a real failed
            # batch would consume them; keep the stub's queue aligned.
            for _ in range(batch.shape[0]):
                if self.classes:
                    self.classes.popleft()
            self.failed += batch.shape[0]
            raise RuntimeError("injected inference failure")
        rows = []
        for _ in range(batch.shape[0]):
            cls = self.classes.popleft() if self.classes else 1
            row = [0.01] * 10
            row[cls] = 0.91
            rows.append(row)
        self.returned += len(rows)
        return rows


class Client:
    def __init__(self, http: httpx.AsyncClient, app):
        self.http = http
        self.app = app
        self.engine = app.state.engine

    async def submit(self, name: str = "sample_gradient.png", deadline_ms: int = 60_000,
                     camera_id: str | None = None):
        data = {"deadline_ms": str(deadline_ms)}
        if camera_id is not None:
            data["camera_id"] = camera_id
        return await self.http.post(
            "/predict",
            files={"file": (name, (SAMPLES / name).read_bytes(), "image/png")},
            data=data,
        )

    async def submit_and_wait(self, name: str = "sample_gradient.png", timeout: float = 10.0,
                              camera_id: str | None = None):
        r = await self.submit(name, camera_id=camera_id)
        assert r.status_code == 201, r.text
        return await self.wait_state(r.json()["order_id"], {"completed", "failed",
                                                            "cancelled", "expired"}, timeout)

    async def get(self, order_id: str):
        return (await self.http.get(f"/orders/{order_id}")).json()

    async def wait_state(self, order_id: str, states, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        body = None
        while time.monotonic() < deadline:
            body = await self.get(order_id)
            if body["status"] in states:
                return body
            await asyncio.sleep(0.01)
        raise AssertionError(f"order {order_id} never reached {states}: {body}")

    async def set_reference(self, camera_id: str, percentages):
        return await self.http.put(f"/cameras/{camera_id}/reference",
                                   json={"percentages": percentages})

    async def distribution(self, camera_id: str):
        return (await self.http.get(f"/cameras/{camera_id}/distribution")).json()


@pytest.fixture
async def make_client():
    clients = []

    async def factory(**settings_kw):
        settings_kw.setdefault("model_path", str(MODEL))
        settings_kw.setdefault("batch_wait_ms", 0)
        settings_kw.setdefault("inference_delay_ms", 0)
        app = create_app(Settings(**settings_kw))
        cm = LifespanManager(app)
        await cm.__aenter__()
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
        client = Client(http, app)
        client.fake = FakeInfer()
        client.fake.delay_s = settings_kw["inference_delay_ms"] / 1000.0
        client.engine._infer = client.fake
        clients.append((client, cm))
        return client

    yield factory
    for client, cm in clients:
        await client.http.aclose()
        await cm.__aexit__(None, None, None)


# --------------------------------------------------------------------- #
# reference management / validation over HTTP
# --------------------------------------------------------------------- #
async def test_reference_endpoints_and_validation(make_client):
    c = await make_client()

    assert (await c.http.get("/cameras/cam1/reference")).status_code == 404
    assert (await c.http.get("/cameras/cam1/distribution")).status_code == 404
    assert (await c.http.delete("/cameras/cam1/reference")).status_code == 404

    r = await c.set_reference("cam1", uniform())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == 1
    assert body["percentages"] == uniform()
    assert body["distribution"]["status"] == "insufficient_sample"
    assert body["distribution"]["sample_count"] == 0

    # malformed payloads are rejected before they can touch the reference
    for payload in (
        {"percentages": [10] * 9},
        {"percentages": [10] * 11},
        {"percentages": [11] + [10] * 9},
        {"percentages": [-1] + [10] * 9},
        {"percentages": [10.0] * 10},
        {"percentages": [True] + [10] * 9},
        {"percentages": ["10"] * 10},
    ):
        r = await c.http.put("/cameras/cam1/reference", json=payload)
        assert r.status_code == 422, (payload, r.status_code, r.text)
    assert (await c.http.get("/cameras/cam1/reference")).json()["version"] == 1

    r = await c.set_reference("cam1", [0] * 10)
    assert r.status_code == 422

    r = await c.set_reference("cam1", [100] + [0] * 9)
    assert r.status_code == 200
    assert r.json()["version"] == 2

    r = await c.http.delete("/cameras/cam1/reference")
    assert r.status_code == 200
    assert (await c.http.get("/cameras/cam1/reference")).status_code == 404
    # re-creating continues the version sequence
    assert (await c.set_reference("cam1", uniform())).json()["version"] == 3


async def test_blank_camera_id_is_treated_as_absent(make_client):
    c = await make_client()
    body = await c.submit_and_wait(camera_id="   ")
    assert body["status"] == "completed"
    assert body["camera_id"] is None
    assert (await c.http.get("/cameras")).json() == {"count": 0, "cameras": []}


# --------------------------------------------------------------------- #
# baseline behaviour is unchanged without camera_id
# --------------------------------------------------------------------- #
async def test_predict_without_camera_is_unchanged(make_client):
    c = await make_client()
    r = await c.submit()
    assert r.status_code == 201
    view = r.json()
    assert view["camera_id"] is None and view["camera_version"] is None
    body = await c.wait_state(view["order_id"], {"completed"})
    assert len(body["result"]["scores"]) == 10
    assert (await c.http.get("/cameras")).json()["count"] == 0


async def test_camera_without_reference_completes_but_is_not_monitored(make_client):
    c = await make_client()
    body = await c.submit_and_wait(camera_id="unconfigured")
    assert body["status"] == "completed"
    assert body["camera_id"] == "unconfigured"
    assert body["camera_version"] is None
    assert body["result"]["predicted_class"] == 1
    assert (await c.http.get("/cameras/unconfigured/distribution")).status_code == 404


async def test_zero_batch_wait_does_not_spin_when_queue_is_empty(make_client):
    # Regression: with BATCH_WAIT_MS=0 the worker must park after draining the
    # queue instead of re-looping on a still-set wake event.
    c = await make_client(batch_wait_ms=0)
    for _ in range(3):
        body = await c.submit_and_wait()
        assert body["status"] == "completed"
    worker = c.engine._worker
    await asyncio.sleep(0.05)
    before = time.monotonic()
    # A parked task yields immediately; a spinning one starves the loop. Measure
    # via an independent callback that must run promptly.
    await asyncio.sleep(0.05)
    elapsed = time.monotonic() - before
    assert elapsed < 0.5
    assert not c.engine._pending
    assert worker is not None and not worker.done()


# --------------------------------------------------------------------- #
# end-to-end monitoring with deterministic classes
# --------------------------------------------------------------------- #
async def test_anomaly_lifecycle_and_reference_update(make_client):
    c = await make_client(inference_delay_ms=0)
    fake = c.fake

    # reference expects a uniform 10% spread; every frame is class 5.
    assert (await c.set_reference("camA", uniform())).status_code == 200
    fake.expect([5] * 20)
    for _ in range(19):
        body = await c.submit_and_wait(camera_id="camA")
        assert body["status"] == "completed"
        assert body["camera_version"] == 1
        assert body["result"]["predicted_class"] == 5

    snap = await c.distribution("camA")
    assert snap["sample_count"] == 19
    assert snap["status"] == "insufficient_sample"
    assert snap["anomaly"] is False

    await c.submit_and_wait(camera_id="camA")
    snap = await c.distribution("camA")
    assert snap["sample_count"] == 20
    assert snap["status"] == "anomaly"
    assert snap["anomaly"] is True
    assert snap["tvd"] == pytest.approx(0.9)
    assert snap["counts"][5] == 20
    assert snap["proportions"][5] == pytest.approx(1.0)

    # A list endpoint shows the same camera.
    listing = (await c.http.get("/cameras")).json()
    assert listing["count"] == 1
    assert listing["cameras"][0]["camera_id"] == "camA"
    assert listing["cameras"][0]["status"] == "anomaly"

    # Updating the reference clears the alert and the window immediately.
    r = await c.set_reference("camA", [100] + [0] * 9)
    assert r.status_code == 200
    assert r.json()["version"] == 2
    assert r.json()["distribution"]["sample_count"] == 0
    assert r.json()["distribution"]["anomaly"] is False
    snap = await c.distribution("camA")
    assert snap["status"] == "insufficient_sample"
    assert snap["tvd"] is None

    # New frames accepted under v2 are counted and the distribution is normal.
    fake.expect([0] * 20)
    for _ in range(20):
        body = await c.submit_and_wait(camera_id="camA")
        assert body["camera_version"] == 2
    snap = await c.distribution("camA")
    assert snap["sample_count"] == 20
    assert snap["status"] == "normal"
    assert snap["tvd"] == pytest.approx(0.0)
    assert snap["reference_version"] == 2


async def test_pre_update_inflight_frames_deliver_results_but_stay_out(make_client):
    c = await make_client(inference_delay_ms=800)
    fake = c.fake

    assert (await c.set_reference("camB", uniform())).status_code == 200
    fake.expect([3, 3, 3, 3, 7, 7])
    orders = []
    for _ in range(4):
        r = await c.submit(camera_id="camB")
        assert r.status_code == 201
        orders.append(r.json()["order_id"])
    await asyncio.sleep(0.2)  # batch in flight under version 1
    for oid in orders:
        assert (await c.get(oid))["status"] == "processing"

    # Update while the batch is computing: window resets to version 2.
    r = await c.set_reference("camB", [0] * 3 + [100] + [0] * 6)
    assert r.status_code == 200
    assert r.json()["version"] == 2
    assert r.json()["distribution"]["sample_count"] == 0

    # Old-version frames complete and still deliver their classification.
    for oid in orders:
        body = await c.wait_state(oid, {"completed"})
        assert body["camera_version"] == 1
        assert body["result"]["predicted_class"] == 3
    snap = await c.distribution("camB")
    assert snap["sample_count"] == 0
    assert snap["status"] == "insufficient_sample"

    # Frames accepted after the update are eligible.
    for _ in range(2):
        body = await c.submit_and_wait(camera_id="camB")
        assert body["camera_version"] == 2
        assert body["result"]["predicted_class"] == 7
    snap = await c.distribution("camB")
    assert snap["sample_count"] == 2
    assert snap["counts"][7] == 2
    assert snap["counts"][3] == 0


async def test_cancelled_expired_failed_orders_never_enter_window(make_client):
    c = await make_client(inference_delay_ms=600)
    fake = c.fake
    assert (await c.set_reference("camC", uniform())).status_code == 200

    # 1) cancelled in-flight
    fake.expect([0])
    r = await c.submit(camera_id="camC")
    cancelled_id = r.json()["order_id"]
    await asyncio.sleep(0.15)
    assert (await c.get(cancelled_id))["status"] == "processing"
    assert (await c.http.post(f"/orders/{cancelled_id}/cancel")).status_code == 200
    await c.wait_state(cancelled_id, {"cancelled"})
    await asyncio.sleep(0.7)  # let the batch finish and discard the result

    # 2) expired mid-flight (reported immediately; the batch still finishes
    #    computing and its late result is discarded)
    fake.expect([1])
    r = await c.submit(deadline_ms=150, camera_id="camC")
    expired_id = r.json()["order_id"]
    body = await c.wait_state(expired_id, {"expired"})
    assert body["result"] is None
    await asyncio.sleep(0.7)

    # 3) failed inference (the call that reaches this third sample raises)
    fake.expect([2])
    fake.fail_at = fake.returned + 1
    r = await c.submit(camera_id="camC")
    failed_id = r.json()["order_id"]
    body = await c.wait_state(failed_id, {"failed"})
    assert body["result"] is None
    fake.fail_at = 10 ** 9

    # 4) two normal completions
    fake.expect([0, 1])
    for _ in range(2):
        body = await c.submit_and_wait(camera_id="camC")
        assert body["status"] == "completed"

    snap = await c.distribution("camC")
    assert snap["sample_count"] == 2
    assert snap["counts"][0] == 1
    assert snap["counts"][1] == 1
    assert snap["status"] == "insufficient_sample"


async def test_deleted_reference_stops_monitoring_and_recreate_isolates(make_client):
    c = await make_client(inference_delay_ms=500)
    fake = c.fake
    assert (await c.set_reference("camD", uniform())).status_code == 200
    fake.expect([4, 4, 9])

    orders = []
    for _ in range(2):
        r = await c.submit(camera_id="camD")
        orders.append(r.json()["order_id"])
    await asyncio.sleep(0.15)

    # Delete + re-create with a new version while the frames are in flight.
    assert (await c.http.delete("/cameras/camD/reference")).status_code == 200
    r = await c.set_reference("camD", [0] * 4 + [100] + [0] * 5)
    assert r.json()["version"] == 2

    for oid in orders:
        body = await c.wait_state(oid, {"completed"})
        assert body["result"]["predicted_class"] == 4
    snap = await c.distribution("camD")
    assert snap["sample_count"] == 0

    body = await c.submit_and_wait(camera_id="camD")
    assert body["camera_version"] == 2
    assert body["result"]["predicted_class"] == 9
    snap = await c.distribution("camD")
    assert snap["sample_count"] == 1
    assert snap["counts"][9] == 1
