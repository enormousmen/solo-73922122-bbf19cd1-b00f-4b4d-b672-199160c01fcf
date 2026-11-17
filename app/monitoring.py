"""Per-camera class-distribution monitoring.

A camera has a *reference distribution*: 10 non-negative integer percentages
that sum to 100 (index = predicted class).  Every order accepted for that
camera records the reference version in force at acceptance time.  When the
order completes successfully, its predicted class is appended to the camera's
sliding window **only if** the version it was accepted under is still the
current version.

Consequences (and why the extra version bookkeeping is needed):

* Updating a reference creates a new version and *immediately* empties the
  window, so an old alert disappears at once.
* Frames accepted before an update but still queued/in-flight when the update
  happens are rejected by the version check when they eventually complete.
  They still return their classification result normally; they simply cannot
  enter the new window.
* Deleting and re-creating a camera keeps the version counter increasing, so
  a frame accepted under the deleted reference can never be counted against
  the re-created one.
* Cancelled, expired and failed orders are never reported to this registry by
  the engine, so they cannot enter the window.

The window keeps the 20 most recently *completed* eligible orders.  A verdict
is only issued once at least 20 eligible completions exist; before that the
status is ``insufficient_sample``.  The distance is total variation distance:

    TVD = 0.5 * sum_i |observed_proportion_i - reference_proportion_i|

with reference proportions derived from the integer percentages.  A TVD
strictly greater than 0.3 is an anomaly (exactly 0.3 is not, hence the tiny
epsilon absorbing floating-point summation error).

All methods run on the asyncio event-loop thread (the engine's completion
callback is invoked from the batch worker task), so no locks are required.
"""
from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("camera-monitoring")

NUM_CLASSES = 10
WINDOW_SIZE = 20
TVD_THRESHOLD = 0.3
# "strictly greater than 0.3": absorb float summation noise at the boundary.
_TVD_EPS = 1e-9


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def validate_percentages(percentages: Iterable[int]) -> tuple[int, ...]:
    """Validate a 10-class reference distribution.

    Values must be *integers* (booleans rejected), non-negative and sum to 100.
    Raises ValueError with a human-readable reason.
    """
    values = tuple(percentages)
    if len(values) != NUM_CLASSES:
        raise ValueError(f"exactly {NUM_CLASSES} percentages are required, got {len(values)}")
    for i, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"percentages[{i}] must be an integer, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"percentages[{i}] must be non-negative, got {value}")
    total = sum(values)
    if total != 100:
        raise ValueError(f"percentages must sum to 100, got {total}")
    return values


@dataclass(frozen=True)
class CameraReference:
    camera_id: str
    percentages: tuple[int, ...]
    version: int
    updated_at: str

    def view(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "version": self.version,
            "percentages": list(self.percentages),
            "updated_at": self.updated_at,
        }


@dataclass
class _CameraState:
    reference: CameraReference
    # Completion order, capped at the 20 most recent eligible completions.
    window: deque[int] = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))


class MonitoringRegistry:
    """In-memory registry of camera references and eligibility windows."""

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        tvd_threshold: float = TVD_THRESHOLD,
    ) -> None:
        self.window_size = window_size
        self.tvd_threshold = tvd_threshold
        self._states: dict[str, _CameraState] = {}
        # Versions are retained even after a reference is deleted, so that a
        # re-created camera never reuses a version that old in-flight frames
        # were accepted under.
        self._last_version: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # reference management
    # ------------------------------------------------------------------ #
    def set_reference(self, camera_id: str, percentages: Iterable[int]) -> CameraReference:
        """Create or update a reference; the window is emptied immediately."""
        values = validate_percentages(percentages)
        version = self._last_version.get(camera_id, 0) + 1
        self._last_version[camera_id] = version
        reference = CameraReference(
            camera_id=camera_id,
            percentages=values,
            version=version,
            updated_at=_utc_now_iso(),
        )
        # Replacing the state drops the previous window (and therefore any
        # previous verdict) atomically in event-loop terms.
        self._states[camera_id] = _CameraState(reference=reference)
        return reference

    def remove_reference(self, camera_id: str) -> bool:
        """Delete a camera's reference and its window; version counter is kept."""
        return self._states.pop(camera_id, None) is not None

    def get_reference(self, camera_id: str) -> Optional[CameraReference]:
        state = self._states.get(camera_id)
        return state.reference if state is not None else None

    def accepted_version(self, camera_id: str) -> Optional[int]:
        """Reference version to stamp on a newly accepted frame, if configured."""
        state = self._states.get(camera_id)
        return state.reference.version if state is not None else None

    # ------------------------------------------------------------------ #
    # eligibility window
    # ------------------------------------------------------------------ #
    def record_completion(
        self,
        camera_id: Optional[str],
        accepted_version: Optional[int],
        predicted_class: int,
    ) -> bool:
        """Record a successfully completed order if it is still eligible.

        Returns True when the class was appended to the window.  The version
        equality check is what keeps pre-update in-flight frames out of the
        new reference window.
        """
        if camera_id is None or accepted_version is None:
            return False
        state = self._states.get(camera_id)
        if state is None or accepted_version != state.reference.version:
            return False
        if not 0 <= predicted_class < NUM_CLASSES:
            log.warning(
                "ignoring out-of-range predicted class %r for camera %s",
                predicted_class,
                camera_id,
            )
            return False
        state.window.append(predicted_class)
        return True

    # ------------------------------------------------------------------ #
    # read side
    # ------------------------------------------------------------------ #
    def snapshot(self, camera_id: str) -> Optional[dict]:
        """Current distribution, TVD and verdict, or None for unknown cameras."""
        state = self._states.get(camera_id)
        if state is None:
            return None
        ref = state.reference
        counts = [0] * NUM_CLASSES
        for predicted_class in state.window:
            counts[predicted_class] += 1
        sample_count = sum(counts)

        if sample_count:
            proportions = [count / sample_count for count in counts]
            reference = [value / 100.0 for value in ref.percentages]
            raw_tvd = 0.5 * sum(
                abs(observed - expected)
                for observed, expected in zip(proportions, reference)
            )
            tvd: Optional[float] = round(raw_tvd, 6)
        else:
            proportions = [0.0] * NUM_CLASSES
            raw_tvd = None
            tvd = None

        if sample_count < self.window_size:
            status = "insufficient_sample"
            anomaly = False
            message = f"样本不足：{sample_count}/{self.window_size}"
        elif raw_tvd is not None and raw_tvd > self.tvd_threshold + _TVD_EPS:
            status = "anomaly"
            anomaly = True
            message = f"分布异常：总变差距离 {tvd} > {self.tvd_threshold}"
        else:
            status = "normal"
            anomaly = False
            message = "分布正常"

        return {
            "camera_id": camera_id,
            "reference_version": ref.version,
            "reference_percentages": list(ref.percentages),
            "reference_updated_at": ref.updated_at,
            "window_limit": self.window_size,
            "sample_count": sample_count,
            "counts": counts,
            # Fractions in [0, 1]; index = class id, sum = 1 when non-empty.
            "proportions": [round(value, 6) for value in proportions],
            # Same distribution expressed as percentages for operator display.
            "percentages": [round(value * 100.0, 2) for value in proportions],
            "tvd": tvd,
            "threshold": self.tvd_threshold,
            "status": status,
            "anomaly": anomaly,
            "message": message,
        }

    def list_cameras(self) -> list[dict]:
        """Snapshots for all configured cameras, sorted by camera id."""
        return [self.snapshot(camera_id) for camera_id in sorted(self._states)]
