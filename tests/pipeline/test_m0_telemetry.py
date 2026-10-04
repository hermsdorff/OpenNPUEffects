import json
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "bin"))
from npu_pipeline.telemetry import (
    BoundedTelemetry, NullTelemetry, ObservationOrigin, duration_ms,
    event_json, provenance_fields,
)


class FakeClock:
    def __init__(self):
        self.now = 1_000_000_000

    def __call__(self):
        return self.now

    def advance_ms(self, ms):
        self.now += int(ms * 1_000_000)


class ReorderedExecutor:
    """Minimal identity fixture; does not emulate OpenVINO scheduling."""
    def __init__(self):
        self.pending = {}

    def submit(self, origin, callback):
        self.pending[origin.frame_id] = (origin, callback)

    def finish(self, frame_id):
        origin, callback = self.pending.pop(frame_id)
        callback(origin)


class M0TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.t = BoundedTelemetry("video", "test", event_capacity=16,
                                  samples_per_metric=8, clock_ns=self.clock)

    def origin(self, frame=1, generation="g1", cfg=1, arrival=None):
        return ObservationOrigin(generation, frame,
                                 self.clock() if arrival is None else arrival, cfg)

    def test_null_does_not_read_clock_or_store_data(self):
        n = NullTelemetry()
        self.assertFalse(n.emit("anything", pcm=b"raw"))
        self.assertFalse(n.sample("stage", 3))
        self.assertFalse(n.count("frames"))
        self.assertEqual(n.drain(), [])
        self.assertEqual(n.events_queued, 0)
        self.assertFalse(n.report()["enabled"])

    def test_origin_is_immutable(self):
        o = self.origin()
        with self.assertRaises(AttributeError):
            o.frame_id = 2

    def test_exact_old_and_unknown(self):
        old = self.origin()
        self.clock.advance_ms(40)
        current = self.origin(2)
        fields = provenance_fields(current, old, self.clock())
        self.assertEqual(fields["source_relation"], "OLDER_FRAME")
        self.assertEqual(fields["source_frame_gap"], 1)
        self.assertEqual(fields["source_arrival_age_ms"], 40)
        self.assertEqual(provenance_fields(current, current, self.clock())
                         ["source_relation"], "SAME_FRAME")
        self.assertIsNone(provenance_fields(current, None, self.clock())
                          ["source_arrival_age_ms"])

    def test_generation_and_config_not_treated_as_exact(self):
        old = self.origin()
        self.assertEqual(old.relation_to(self.origin(generation="g2"), self.clock())
                         ["relation"], "OTHER_GENERATION")
        self.assertEqual(old.relation_to(self.origin(cfg=2), self.clock())
                         ["relation"], "OTHER_CONFIG")

    def test_future_and_invalid_time_are_visible(self):
        self.assertEqual(self.origin(3).relation_to(self.origin(2), self.clock())
                         ["relation"], "FUTURE_SOURCE")
        future = self.origin(arrival=self.clock() + 1)
        self.assertEqual(future.relation_to(self.origin(), self.clock())
                         ["relation"], "INVALID_TIME")

    def test_event_queue_is_bounded(self):
        for i in range(1000):
            self.t.emit("capture", frame_id=i)
        r = self.t.report()
        self.assertEqual(r["events_queued"], 16)
        self.assertEqual(r["events_queue_dropped"], 984)
        self.assertEqual(len(self.t.drain(limit=16)), 16)

    def test_lock_contention_drops_without_waiting(self):
        done = threading.Event()
        results = []
        self.t._lock.acquire()
        try:
            def producer():
                results.extend([self.t.emit("callback"),
                                self.t.sample("callback_ms", 1),
                                self.t.count("callback")])
                done.set()
            worker = threading.Thread(target=producer, daemon=True)
            worker.start()
            self.assertTrue(done.wait(timeout=1))
            self.assertEqual(results, [False, False, False])
        finally:
            self.t._lock.release()
        worker.join(timeout=1)

    def test_pixels_pcm_and_invalid_scalars_are_rejected(self):
        for value in (b"pcm", bytearray(5), [1, 2], {"pixels": 1},
                      float("nan"), float("inf"), "x" * 161, 2**80):
            self.assertFalse(self.t.emit("event", payload=value))
        self.assertFalse(self.t.emit("event", observed_mono_ns=12))
        self.assertFalse(self.t.emit("x" * 161))
        self.assertTrue(self.t.emit("event", state="UNKNOWN", value=None))

    def test_quantiles_bounded_and_documented(self):
        for i in range(1, 101):
            self.t.sample("infer", i)
        m = self.t.report()["metrics"]["infer"]
        self.assertEqual(m["retained_n"], 8)
        self.assertEqual(m["p50_ms"], 96)
        self.assertEqual(m["p95_ms"], 100)
        self.assertEqual(m["p99_ms"], 100)
        self.assertEqual(m["max_ms"], 100)

    def test_metric_and_counter_cardinality_are_bounded(self):
        for i in range(200):
            self.t.sample("metric_" + str(i), 1)
            self.t.count("counter_" + str(i))
        r = self.t.report()
        self.assertEqual(len(r["metrics"]), 64)
        self.assertEqual(len(r["counts_best_effort"]), 64)

    def test_reordered_callback_reports_origin_not_current(self):
        executor = ReorderedExecutor()
        current = self.origin(2)
        def callback(source):
            self.t.emit("primary_callback", **provenance_fields(
                current, source, self.clock()))
        executor.submit(self.origin(1), callback)
        executor.submit(self.origin(2), callback)
        executor.finish(2)
        executor.finish(1)
        events = self.t.drain(limit=16)
        self.assertEqual([e["source_frame_id"] for e in events], [2, 1])
        self.assertEqual(events[1]["source_relation"], "OLDER_FRAME")

    def test_timeout_and_reuse_are_independent_events(self):
        self.t.emit("primary_wait", requested_frame_id=2, timed_out=True)
        self.t.emit("primary_selected", reason="TIMEOUT_FALLBACK",
                    **provenance_fields(self.origin(2), self.origin(1),
                                        self.clock()))
        events = self.t.drain(limit=16)
        self.assertTrue(events[0]["timed_out"])
        self.assertEqual(events[1]["source_relation"], "OLDER_FRAME")

    def test_serialization_and_duration(self):
        self.t.emit("capture", frame_id=1)
        e = self.t.drain(limit=16)[0]
        self.assertEqual(json.loads(event_json(e)), e)
        self.assertEqual(duration_ms(10, 2_000_010), 2)
        with self.assertRaises(ValueError):
            duration_ms(2, 1)

    def test_invalid_limits(self):
        for kwargs in ({"event_capacity": 1}, {"samples_per_metric": 1}):
            with self.assertRaises(ValueError):
                BoundedTelemetry("video", "test", **kwargs)
        with self.assertRaises(ValueError):
            BoundedTelemetry("other", "test")
        with self.assertRaises(ValueError):
            self.t.drain(limit=17)

    def test_events_queued_property(self):
        self.assertEqual(self.t.events_queued, 0)
        self.t.emit("capture", frame_id=1)
        self.t.emit("capture", frame_id=2)
        self.assertEqual(self.t.events_queued, 2)
        drained = self.t.drain(limit=1)
        self.assertEqual(len(drained), 1)
        self.assertEqual(self.t.events_queued, 1)
        self.t.drain(limit=1)
        self.assertEqual(self.t.events_queued, 0)


if __name__ == "__main__":
    unittest.main()
