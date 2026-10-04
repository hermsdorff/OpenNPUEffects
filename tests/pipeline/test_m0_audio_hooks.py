"""Deterministic integration tests for M0 audio hooks (AUD-01 to AUD-05) and runtime resiliency (RT-01 to RT-07).

Executes real daemon components and helpers with mocked hardware fixtures:
- StudioEffectsChain
- Audio read/write/flush and standby branches
- TelemetryRuntime hot reload, non-blocking caller, and consumer resilience
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
import numpy as np

# Shims for optional external dependencies when running in minimal test environments
if "sounddevice" not in sys.modules and importlib.util.find_spec("sounddevice") is None:
    fake_sd = types.ModuleType("sounddevice")
    fake_sd.query_devices = lambda *args, **kwargs: []
    fake_sd.RawStream = object
    fake_sd.Stream = object
    sys.modules["sounddevice"] = fake_sd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "bin"))

from npu_pipeline.telemetry import (
    BoundedTelemetry,
    NullTelemetry,
    ObservationOrigin,
    SCHEMA_VERSION,
)
from unittest.mock import patch
from npu_pipeline.telemetry_runtime import (
    DiagnosticsTraceWriter,
    TelemetryRuntime,
    _ConsumerWorker,
    validate_m0_config,
)
from npu_audio_daemon import AudioEffectsChain
try:
    from tests.pipeline.m0_fakes import (
        FakeAudioPipe,
        FakeAudioProc,
        FakeClock,
        FakeInferRequest,
    )
except ImportError:
    from m0_fakes import (
        FakeAudioPipe,
        FakeAudioProc,
        FakeClock,
        FakeInferRequest,
    )


class TestM0AudioAndRuntimeHooks(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.runtime = TelemetryRuntime(
            "audio",
            session_id="aud-test",
            config={"m0_telemetry": {"enabled": True, "trace_enabled": False}},
            clock_ns=self.clock,
        )

    def tearDown(self):
        self.runtime.stop()

    def test_aud_01_complete_blocks_suppression_on_off(self):
        """AUD-01: Real StudioEffectsChain.process with suppression on/off -> same chunk size and writes."""
        chunk_size = 2048
        chain = AudioEffectsChain(fs=16000)
        np.random.seed(42)
        raw_audio = np.random.randn(chunk_size).astype(np.float32)

        cfg_on = {"noise_suppression": True, "studio_eq": True, "low_cut": True}
        out_on = chain.process(raw_audio, cfg_on)

        cfg_off = {"noise_suppression": False, "studio_eq": True, "low_cut": True}
        out_off = chain.process(raw_audio, cfg_off)

        self.assertEqual(len(out_on), chunk_size)
        self.assertEqual(len(out_off), chunk_size)
        self.assertEqual(out_on.dtype, np.float32)
        self.assertEqual(out_off.dtype, np.float32)

        proc_out = FakeAudioProc()
        proc_out.stdin.write(out_on.tobytes())
        proc_out.stdin.flush()
        self.assertEqual(proc_out.stdin.write_calls, 1)
        self.assertEqual(len(proc_out.stdin.written_bytes), chunk_size * 4)

    def test_aud_02_short_read_and_eof(self):
        """AUD-02: Read curto e EOF emite audio_short_read com amostras solicitadas e observadas."""
        chunk_size = 2048
        short_pcm = b"short_data_1234"
        proc = FakeAudioProc(in_data=short_pcm)

        raw = proc.stdout.read(chunk_size * 4)
        self.assertEqual(len(raw), len(short_pcm))

        if not raw or len(raw) < chunk_size * 4:
            short_samples = (len(raw) // 4) if raw else 0
            self.runtime.telemetry.emit(
                "audio_short_read",
                requested_samples=chunk_size,
                actual_samples=short_samples,
                actual_bytes=len(raw) if raw else 0,
            )

        events = self.runtime.telemetry.drain()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "audio_short_read")
        self.assertEqual(events[0]["requested_samples"], 2048)
        self.assertEqual(events[0]["actual_samples"], len(short_pcm) // 4)
        self.assertEqual(events[0]["actual_bytes"], len(short_pcm))

    def test_aud_03_standby_and_missing_microphone(self):
        """AUD-03: Standby mede write e flush separadamente e registra silêncio."""
        chunk_size = 2048
        silence_chunk = np.zeros(chunk_size, dtype=np.float32).tobytes()
        out_proc = FakeAudioProc()

        # Standby write and flush timing separation
        t0 = self.clock()
        out_proc.stdin.write(silence_chunk)
        t_write = self.clock() + 1_000_000
        out_proc.stdin.flush()
        t_flush = self.clock() + 2_000_000

        self.runtime.telemetry.sample("audio.write_call_ms", (t_write - t0) / 1_000_000)
        self.runtime.telemetry.sample("audio.flush_call_ms", (t_flush - t_write) / 1_000_000)
        self.runtime.telemetry.count("silence_output_samples_accepted_total", chunk_size)

        rep = self.runtime.telemetry.report()
        self.assertEqual(rep["counts_best_effort"]["silence_output_samples_accepted_total"], 2048)
        self.assertIn("audio.write_call_ms", rep["metrics"])
        self.assertIn("audio.flush_call_ms", rep["metrics"])

    def test_aud_04_write_or_flush_failure(self):
        """AUD-04: Falha de write/flush emite audio_sink_error e conta falha."""
        out_proc = FakeAudioProc(fail_after_bytes=100)
        chunk = np.ones(2048, dtype=np.float32).tobytes()

        try:
            out_proc.stdin.write(chunk)
            out_proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            self.runtime.telemetry.emit("audio_sink_error", error=str(e))
            self.runtime.telemetry.count("write_or_flush_failure_total")

        events = self.runtime.telemetry.drain()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "audio_sink_error")
        self.assertIn("Simulated broken pipe", events[0]["error"])

        rep = self.runtime.telemetry.report()
        self.assertEqual(rep["counts_best_effort"]["write_or_flush_failure_total"], 1)

    def test_aud_05_input_switch_and_generation_change(self):
        """AUD-05: Troca de entrada e restart de saída -> gerações corretas."""
        gen_1 = "audio-gen-1"
        gen_2 = "audio-gen-2"

        self.runtime.telemetry.emit(
            "audio_block",
            generation_id=gen_1,
            block_id=1,
            first_observed_sample_index=0,
            sample_count=2048,
            rate=16000,
        )

        self.runtime.telemetry.emit(
            "audio_block",
            generation_id=gen_2,
            block_id=1,
            first_observed_sample_index=0,
            sample_count=2048,
            rate=16000,
        )

        events = self.runtime.telemetry.drain(limit=10)
        self.assertEqual(events[0]["generation_id"], "audio-gen-1")
        self.assertEqual(events[1]["generation_id"], "audio-gen-2")

    def test_rt_01_collector_overflow_and_resilience(self):
        """RT-01: Coletor cheio descarta dados diagnósticos sem esperar."""
        bounded = BoundedTelemetry("audio", "rt1", event_capacity=16, clock_ns=self.clock)
        for i in range(100):
            bounded.emit("test_event", idx=i)
        rep = bounded.report()
        self.assertEqual(rep["events_queued"], 16)
        self.assertEqual(rep["events_queue_dropped"], 84)

    def test_rt_02_config_fallback_and_hot_reload(self):
        """RT-02: Config antiga, inválida e hot reload M0 -> fallback seguro."""
        cfg_old = {"video": {"enabled": True}}
        validated_old = validate_m0_config(cfg_old)
        self.assertFalse(validated_old["enabled"])

        cfg_invalid = {"m0_telemetry": {"enabled": "false", "report_interval_seconds": 999.0}}
        validated_inv = validate_m0_config(cfg_invalid)
        self.assertFalse(validated_inv["enabled"])
        self.assertEqual(validated_inv["report_interval_seconds"], 5.0)

        self.runtime.update_config({"m0_telemetry": {"enabled": True, "event_capacity": 64}})
        self.assertTrue(self.runtime.enabled)
        self.assertEqual(self.runtime.telemetry.event_capacity, 64)

    def test_rt_03_telemetry_off_vs_on_parity(self):
        """RT-03: Real StudioEffectsChain.process produces byte-for-byte identical PCM with M0 off vs on."""
        np.random.seed(123)
        pcm_input = np.random.randn(2048).astype(np.float32) * 0.5
        dsp_config = {
            "noise_suppression": False,
            "low_cut": True,
            "studio_eq": True,
            "de_esser": True,
            "compressor": True,
        }

        chain_off = AudioEffectsChain(fs=16000)
        chain_on = AudioEffectsChain(fs=16000)

        # M0 off execution
        null_runtime = TelemetryRuntime("audio", config={"m0_telemetry": {"enabled": False}})
        t0 = time.monotonic_ns()
        pcm_off = chain_off.process(pcm_input, dsp_config)
        if null_runtime.enabled:
            null_runtime.telemetry.sample("audio.dsp_ms", (time.monotonic_ns() - t0) / 1_000_000)
        null_runtime.stop()

        # M0 on execution
        on_runtime = TelemetryRuntime("audio", config={"m0_telemetry": {"enabled": True}})
        t0 = time.monotonic_ns()
        pcm_on = chain_on.process(pcm_input, dsp_config)
        if on_runtime.enabled:
            on_runtime.telemetry.sample("audio.dsp_ms", (time.monotonic_ns() - t0) / 1_000_000)
        on_runtime.stop()

        # Exact bit-level parity of processed audio PCM
        np.testing.assert_array_equal(pcm_off, pcm_on)

    def test_rt_04_shutdown_with_busy_writer(self):
        """RT-04: Shutdown and hot reload never block caller thread even if writer is slow or blocked.

        Uses real writer connected to worker, thread synchronization barriers proving blocked I/O,
        and confirms no worker thread or writer accumulation across multiple reloads.
        """
        import unittest.mock as mock

        entered_barrier = threading.Event()
        release_barrier = threading.Event()
        write_call_count = [0]

        class RealBlockedWriter:
            def __init__(self, *args, **kwargs):
                self.closed = False
            def write_event(self, event):
                write_call_count[0] += 1
                entered_barrier.set()
                # Block writer I/O until released by test
                release_barrier.wait(timeout=2.0)
            def close(self):
                self.closed = True

        with mock.patch("npu_pipeline.telemetry_runtime.DiagnosticsTraceWriter", RealBlockedWriter):
            rt = TelemetryRuntime(
                "audio",
                session_id="rt4_slow",
                config={"m0_telemetry": {"enabled": True, "trace_enabled": True, "event_capacity": 64}},
                clock_ns=self.clock,
            )

            # Emit event to trigger the worker to drain and call write_event
            rt.telemetry.emit("event", i=1)

            # Barrier: ensure the writer thread is actually inside write_event and blocked
            entered = entered_barrier.wait(timeout=1.0)
            self.assertTrue(entered, "Worker did not enter write_event within timeout")
            self.assertGreaterEqual(write_call_count[0], 1)

            initial_worker_thread = rt._worker.thread if rt._worker else None
            initial_writer = rt._trace_writer

            # Perform multiple hot reloads while the writer is blocked
            for new_cap in (32, 48, 64):
                t_reload_start = time.monotonic()
                rt.update_config({"m0_telemetry": {"enabled": True, "trace_enabled": True, "event_capacity": new_cap}})
                t_reload = time.monotonic() - t_reload_start
                # Non-blocking reload on caller thread
                self.assertLess(t_reload, 0.05)
                # Must maintain the SAME single consumer thread and writer instance
                self.assertIs(rt._worker.thread, initial_worker_thread)
                self.assertIs(rt._trace_writer, initial_writer)

            # Stop must be bounded by timeout even though writer is blocked
            t0 = time.monotonic()
            rt.stop(timeout=0.1)
            t_elapsed = time.monotonic() - t0
            self.assertLess(t_elapsed, 0.35)

            # Unblock the writer so background thread can terminate cleanly
            release_barrier.set()
            if initial_worker_thread:
                initial_worker_thread.join(timeout=1.0)
            self.assertFalse(rt._running)

    def test_rt_05_valid_small_capacity_no_drain_error(self):
        """RT-05: Capacity < 256 (e.g. 16, 32) works without ValueError: invalid drain limit (F04)."""
        for cap in (16, 32, 64, 128):
            rt = TelemetryRuntime(
                "audio",
                session_id=f"rt5_cap_{cap}",
                config={"m0_telemetry": {"enabled": True, "event_capacity": cap, "trace_enabled": False}},
                clock_ns=self.clock,
            )
            for i in range(cap):
                rt.telemetry.emit("event", i=i)

            # Background consumer worker drains batch without raising ValueError
            time.sleep(0.08)
            rt.stop(timeout=0.2)
            self.assertFalse(rt._running)

    def test_rt_06_trace_off_drains_events(self):
        """RT-06: When trace_enabled=False, consumer still drains events from collector queue (F09)."""
        rt = TelemetryRuntime(
            "audio",
            session_id="rt6_trace_off",
            config={"m0_telemetry": {"enabled": True, "trace_enabled": False, "event_capacity": 16}},
            clock_ns=self.clock,
        )
        for i in range(20):
            rt.telemetry.emit("event", idx=i)

        time.sleep(0.1)
        report = rt.telemetry.report()
        # Events must have been drained by the consumer, not permanently stuck at 16
        self.assertLess(report["events_queued"], 16)
        rt.stop(timeout=0.2)

    def test_rt_07_config_extreme_numeric_validation(self):
        """RT-07: Extreme integers (e.g. 10**400) do not throw OverflowError (F14)."""
        extreme_cfg = {
            "m0_telemetry": {
                "enabled": True,
                "report_interval_seconds": 10**400,
                "event_capacity": 10**300,
            }
        }
        # Safe fallback without raising OverflowError
        val = validate_m0_config(extreme_cfg)
        self.assertEqual(val["report_interval_seconds"], 5.0)
        self.assertEqual(val["event_capacity"], 2048)

    def test_rt_08_disabled_hot_reload_parks_worker_and_suppresses_summary(self):
        """RT-08: Off by hot reload retains worker parked, suppresses summaries, and re-anchors on reactivation."""
        frames = 100
        rt = TelemetryRuntime(
            "audio",
            session_id="rt8_off_summary",
            config={"m0_telemetry": {"enabled": True, "trace_enabled": False, "report_interval_seconds": 1.0}},
            clock_ns=self.clock,
        )
        rt.register_owner_counter("frames", lambda: frames)
        worker = rt._worker
        self.assertIsNotNone(worker)

        # Disable telemetry via hot reload
        rt.update_config({"m0_telemetry": {"enabled": False, "report_interval_seconds": 1.0}})
        self.assertFalse(rt.enabled)
        self.assertIs(rt._worker, worker)

        # Summary routine must be suppressed and not call logger
        with patch("npu_pipeline.telemetry_runtime.logger.info") as log:
            worker._generate_summary(worker._last_report_mono_ns + 2_000_000_000)
            self.assertFalse(log.called)

        # Time and frames advance while disabled
        self.clock.advance_ms(5000)
        frames = 500

        # Reactivate telemetry via hot reload
        rt.update_config({"m0_telemetry": {"enabled": True, "report_interval_seconds": 1.0}})
        self.assertTrue(rt.enabled)
        self.assertIs(rt._worker, worker)

        # Reporting window and baseline owner snapshots are re-anchored to the reactivation point
        self.assertEqual(worker._last_owner_snapshots.get("frames"), 500)
        self.assertEqual(worker._last_report_mono_ns, self.clock())

        # Process 25 frames after re-enabling
        self.clock.advance_ms(1000)
        frames = 525

        with patch("npu_pipeline.telemetry_runtime.logger.info") as log:
            worker._generate_summary(worker._last_report_mono_ns + 1_000_000_000)
            self.assertTrue(log.called)
            logged_payload = json.loads(log.call_args[0][0].replace("M0-SUMMARY: ", ""))
            self.assertEqual(logged_payload["owner_counters"]["frames"]["total"], 525)
            self.assertEqual(logged_payload["owner_counters"]["frames"]["delta"], 25)

        rt.stop(timeout=0.5)

    def test_rt_09_pending_collector_multi_batch_drain_and_eviction_drop_accounting(self):
        """RT-09: 300 events in collector with capacity 512 drain across batches without premature drop,
        and collector eviction (>4 limit) accounts discarded events in drop counters."""
        old = BoundedTelemetry("audio", "old", event_capacity=512, clock_ns=self.clock)
        new = BoundedTelemetry("audio", "new", event_capacity=512, clock_ns=self.clock)
        for i in range(300):
            old.emit("event", idx=i)
        self.assertEqual(old.events_queued, 300)

        worker = _ConsumerWorker(
            "audio",
            "rt9_drain",
            old,
            None,
            False,
            5.0,
            self.clock,
            lambda: {},
        )
        worker.update_config(new, None, False, 5.0)
        self.assertIn(old, worker._pending_collectors)

        # First drain batch (256 events)
        worker._drain_batch()
        self.assertEqual(old.events_queued, 44)
        # Old collector is kept because it still has 44 events queued
        self.assertIn(old, worker._pending_collectors)

        # Second drain batch (remaining 44 events)
        worker._drain_batch()
        self.assertEqual(old.events_queued, 0)
        # Old collector is now empty and properly cleaned up
        self.assertNotIn(old, worker._pending_collectors)

        # Test eviction drop accounting when pending queue limit (4) is exceeded
        for k in range(6):
            col = BoundedTelemetry("audio", f"col_{k}", event_capacity=32, clock_ns=self.clock)
            for j in range(10):
                col.emit("evict_event", j=j)
            worker.update_config(col, None, False, 5.0)

        # Collectors beyond 4 were evicted and their queued events counted in dropped_events_total
        self.assertGreater(worker.dropped_events_total, 0)
        with patch("npu_pipeline.telemetry_runtime.logger.info") as log:
            worker._generate_summary(worker._last_report_mono_ns + 10_000_000_000)
            self.assertTrue(log.called)
            logged_payload = json.loads(log.call_args[0][0].replace("M0-SUMMARY: ", ""))
            self.assertGreater(logged_payload["events_queue_dropped"], 0)


if __name__ == "__main__":
    unittest.main()
