"""Bounded, best-effort M0 telemetry. No media or third-party imports.

Event producers never wait for the collector lock or writer. Drain, quantiles,
and serialization belong to a diagnostic consumer, not an inference callback.
This module deliberately does not implement synchronization or frame policy.
"""
from collections import Counter, deque
from dataclasses import dataclass
import json
import math
import threading
import time
from typing import Optional


SCHEMA_VERSION = 1
MAX_FIELDS = 48
MAX_TEXT = 160
MAX_METRICS = 64
MAX_COUNTERS = 64


@dataclass(frozen=True)
class ObservationOrigin:
    generation_id: str
    frame_id: int
    arrival_mono_ns: int
    config_version: int
    timestamp_quality: str = "ARRIVAL_ESTIMATE"

    def relation_to(self, current, now_ns):
        """Arrival-based ages, NOT exposure age or presentation latency."""
        if current is None:
            return {"relation": "UNKNOWN", "age_ms": None,
                    "frame_gap": None}
        if self.generation_id != current.generation_id:
            return {"relation": "OTHER_GENERATION", "age_ms": None,
                    "frame_gap": None}
        if self.arrival_mono_ns > now_ns:
            return {"relation": "INVALID_TIME", "age_ms": None,
                    "frame_gap": None}
        gap = current.frame_id - self.frame_id
        if gap < 0:
            relation = "FUTURE_SOURCE"
        elif self.config_version != current.config_version:
            relation = "OTHER_CONFIG"
        elif gap == 0:
            relation = "SAME_FRAME"
        else:
            relation = "OLDER_FRAME"
        return {"relation": relation,
                "age_ms": (now_ns - self.arrival_mono_ns) / 1_000_000,
                "frame_gap": gap}


class NullTelemetry:
    enabled = False
    event_capacity = 0
    samples_per_metric = 0

    @property
    def events_queued(self) -> int:
        return 0

    def emit(self, event, **fields):
        return False

    def sample(self, name, milliseconds):
        return False

    def count(self, name, amount=1):
        return False

    def drain(self, limit=256):
        return []

    def report(self):
        return {"schema_version": SCHEMA_VERSION, "enabled": False}


class BoundedTelemetry:
    """All retained collections bounded; a full queue drops diagnostic data.

    Metric/counter names must be fixed vocabulary, never per-frame labels.
    Counts are best-effort under contention; capture/output authoritative
    counters should be maintained by their owning worker and exported.
    """
    enabled = True

    def __init__(self, stream, session_id, event_capacity=2048,
                 samples_per_metric=512, clock_ns=time.monotonic_ns):
        if stream not in ("video", "audio"):
            raise ValueError("stream must be video or audio")
        if not isinstance(session_id, str) or len(session_id) > MAX_TEXT:
            raise ValueError("invalid session_id")
        if not 16 <= event_capacity <= 16384:
            raise ValueError("event_capacity out of range")
        if not 8 <= samples_per_metric <= 4096:
            raise ValueError("samples_per_metric out of range")
        self.stream = stream
        self.session_id = session_id
        self.clock_ns = clock_ns
        self.event_capacity = event_capacity
        self.samples_per_metric = samples_per_metric
        self._events = deque()
        self._samples = {}
        self._counts = Counter()
        self._queue_dropped = 0
        self._lock = threading.Lock()

    @staticmethod
    def _valid_name(value):
        return isinstance(value, str) and 0 < len(value) <= MAX_TEXT

    def emit(self, event, **fields):
        # Strict scalar whitelist excludes arrays, bytes, pixels and PCM.
        if not self._valid_name(event) or len(fields) > MAX_FIELDS:
            return False
        for key, value in fields.items():
            if not self._valid_name(key):
                return False
            if value is not None and type(value) not in (str, int, float, bool):
                return False
            if isinstance(value, str) and len(value) > MAX_TEXT:
                return False
            if isinstance(value, float) and not math.isfinite(value):
                return False
            if isinstance(value, int) and value.bit_length() > 64:
                return False
        reserved = {"schema_version", "stream", "session_id", "event",
                    "observed_mono_ns"}
        if reserved.intersection(fields):
            return False
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if len(self._events) >= self.event_capacity:
                self._queue_dropped += 1
                return False
            self._events.append({
                "schema_version": SCHEMA_VERSION,
                "stream": self.stream, "session_id": self.session_id,
                "event": event, "observed_mono_ns": self.clock_ns(), **fields,
            })
            return True
        finally:
            self._lock.release()

    def sample(self, name, milliseconds):
        if not self._valid_name(name):
            return False
        if type(milliseconds) not in (int, float):
            return False
        if not math.isfinite(milliseconds) or milliseconds < 0:
            return False
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if name not in self._samples:
                if len(self._samples) >= MAX_METRICS:
                    return False
                self._samples[name] = deque(maxlen=self.samples_per_metric)
            self._samples[name].append(float(milliseconds))
            return True
        finally:
            self._lock.release()

    def count(self, name, amount=1):
        if not self._valid_name(name) or type(amount) is not int or amount < 0:
            return False
        if amount.bit_length() > 64:
            return False
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if name not in self._counts and len(self._counts) >= MAX_COUNTERS:
                return False
            self._counts[name] = min(2**64 - 1, self._counts[name] + amount)
            return True
        finally:
            self._lock.release()

    @property
    def events_queued(self) -> int:
        with self._lock:
            return len(self._events)

    def drain(self, limit=256):
        if not 1 <= limit <= self.event_capacity:
            raise ValueError("invalid drain limit")
        with self._lock:
            return [self._events.popleft()
                    for _ in range(min(limit, len(self._events)))]

    def report(self):
        # Snapshot first, quantiles outside the lock.
        with self._lock:
            values = {k: tuple(v) for k, v in self._samples.items()}
            counts = dict(self._counts)
            dropped = self._queue_dropped
            queued = len(self._events)
        metrics = {}
        for name, samples in values.items():
            ordered = sorted(samples)

            def percentile(p):
                return ordered[max(0, math.ceil(p * len(ordered)) - 1)]

            metrics[name] = {
                "retained_n": len(ordered), "p50_ms": percentile(.50),
                "p95_ms": percentile(.95), "p99_ms": percentile(.99),
                "max_ms": ordered[-1], "mean_ms": sum(ordered) / len(ordered),
            }
        return {
            "schema_version": SCHEMA_VERSION, "enabled": True,
            "stream": self.stream, "session_id": self.session_id,
            "sample_policy": "last_n_per_metric_nearest_rank",
            "samples_per_metric": self.samples_per_metric,
            "metrics": metrics, "counts_best_effort": counts,
            "events_queue_dropped": dropped, "events_queued": queued,
            "contention_loss_possible": True,
        }


def event_json(record):
    """Consumer-only serialization; stdout/logging is not called by emit()."""
    return json.dumps(record, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def duration_ms(start_ns, end_ns):
    if end_ns < start_ns:
        raise ValueError("monotonic duration regressed")
    return (end_ns - start_ns) / 1_000_000


def provenance_fields(current: Optional[ObservationOrigin],
                      source: Optional[ObservationOrigin], now_ns):
    relation = (source.relation_to(current, now_ns) if source else
                {"relation": "UNKNOWN", "age_ms": None, "frame_gap": None})
    return {
        "current_generation_id": current.generation_id if current else None,
        "current_frame_id": current.frame_id if current else None,
        "source_generation_id": source.generation_id if source else None,
        "source_frame_id": source.frame_id if source else None,
        "source_config_version": source.config_version if source else None,
        "source_timestamp_quality": source.timestamp_quality if source else None,
        "source_relation": relation["relation"],
        "source_arrival_age_ms": relation["age_ms"],
        "source_frame_gap": relation["frame_gap"],
    }
