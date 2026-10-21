"""Deterministic tests for AudioCaptureReader, AudioOutputWriter, and CaptureRetryPolicy.

Covers AUD-SW-05 scenarios:
- T01: Single complete block (2048 samples / 8192 bytes, no loss/duplication)
- T02: Fragmented reads with arbitrary chunk sizes (including non-multiples of 4)
- T03: Live producer with no data (poll returns, stall fires at deadline)
- T04: Trickling bytes (absolute deadline fires despite trickling)
- T10: EOF and descriptor error handling (no partial inference, discarded bytes tracked)
- T16: Output sink stalled / write deadline protection
- T17: Cancellation and detachment during stall
- Memory & cycle stability: 1,000 simulated stall/switch cycles
"""
import fcntl
import io
import os
import select
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "bin"))

from npu_pipeline.audio_capture import (
    CaptureReadStatus,
    AudioCaptureState,
    CaptureBlockResult,
    WriteResult,
    AudioCaptureReader,
    AudioOutputWriter,
    CaptureRetryPolicy,
    AudioSwitchCoordinator,
)


class FakeControllableClock:
    """Deterministic monotonic clock for timing tests."""

    def __init__(self, start_sec: float = 1000.0):
        self.current_time = start_sec

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds


class TestAudioCaptureUnit(unittest.TestCase):

    def test_t01_single_complete_block(self):
        """T01: Exactly 8192 bytes (2048 float32 samples) returned without duplication or loss."""
        clock = FakeControllableClock()
        reader = AudioCaptureReader(
            block_size_bytes=8192,
            poll_timeout_sec=0.025,
            block_deadline_sec=0.750,
            clock=clock,
        )

        expected_data = bytes([i % 256 for i in range(8192)])
        stream = io.BytesIO(expected_data)

        reader.attach(stream, generation_id="gen-1", epoch=1)

        result = reader.read_step()
        self.assertEqual(result.status, CaptureReadStatus.COMPLETE)
        self.assertEqual(len(result.data), 8192)
        self.assertEqual(result.data, expected_data)
        self.assertEqual(result.epoch, 1)
        self.assertEqual(result.generation_id, "gen-1")
        self.assertEqual(reader.buffer_len, 0)
        self.assertEqual(reader.total_discarded_bytes, 0)

    def test_t02_fragments_of_varying_sizes(self):
        """T02: Byte-by-byte reconstruction across arbitrary fragment boundaries (including non-multiples of 4)."""
        clock = FakeControllableClock()
        reader = AudioCaptureReader(
            block_size_bytes=8192,
            poll_timeout_sec=0.025,
            block_deadline_sec=0.750,
            clock=clock,
        )

        # Uneven chunks: 3 bytes, 7 bytes, 13 bytes, 8000 bytes, 169 bytes (total = 8192)
        fragments = [
            b"abc",
            b"1234567",
            b"abcdefghijklm",
            b"X" * 8000,
            b"Y" * 169,
        ]
        self.assertEqual(sum(len(f) for f in fragments), 8192)
        full_expected = b"".join(fragments)

        class FragmentStream:
            def __init__(self, frags):
                self.frags = list(frags)

            def read(self, n):
                if self.frags:
                    return self.frags.pop(0)
                return b""

        frag_stream = FragmentStream(fragments)
        reader.attach(frag_stream, generation_id="gen-frag", epoch=1)

        # First 4 reads must return PENDING
        for _ in range(4):
            res = reader.read_step()
            self.assertEqual(res.status, CaptureReadStatus.PENDING)

        # 5th read completes the 8192-byte block
        res = reader.read_step()
        self.assertEqual(res.status, CaptureReadStatus.COMPLETE)
        self.assertEqual(len(res.data), 8192)
        self.assertEqual(res.data, full_expected)

    def test_t03_live_producer_without_data_stall(self):
        """T03: Live producer with no data returns PENDING then fires STALL after formation deadline."""
        clock = FakeControllableClock(100.0)
        reader = AudioCaptureReader(
            block_size_bytes=8192,
            poll_timeout_sec=0.025,
            block_deadline_sec=0.750,
            initial_tolerance_sec=1.500,
            clock=clock,
        )

        # Stream that returns empty bytes without closing (live producer)
        class LiveSilentStream:
            def read(self, n):
                return b""

        proc = SimpleNamespace(poll=lambda: None, terminated=False)
        reader.attach(LiveSilentStream(), generation_id="gen-stall", epoch=1)

        # Initial read starts timer and returns PENDING
        res = reader.read_step()
        self.assertEqual(res.status, CaptureReadStatus.PENDING)

        # Advance within initial tolerance (1.5s)
        clock.advance(0.500)
        res = reader.read_step()
        self.assertEqual(res.status, CaptureReadStatus.PENDING)

        # Advance past 1.5s -> triggers STALL exactly once
        clock.advance(1.100)
        res = reader.read_step()
        self.assertEqual(res.status, CaptureReadStatus.STALL)
        self.assertEqual(res.reason, "deadline_exceeded")

        # Next read restarts deadline timer and returns PENDING
        res = reader.read_step()
        self.assertEqual(res.status, CaptureReadStatus.PENDING)

    def test_t04_trickling_bytes_absolute_deadline(self):
        """T04: Trickling bytes (e.g. 10 bytes per poll) cannot prolong block assembly past absolute deadline."""
        clock = FakeControllableClock(200.0)
        reader = AudioCaptureReader(
            block_size_bytes=8192,
            poll_timeout_sec=0.025,
            block_deadline_sec=0.750,
            initial_tolerance_sec=0.750,
            clock=clock,
        )

        class TrickleStream:
            def read(self, n):
                return b"trickle123"  # 10 bytes

        reader.attach(TrickleStream(), generation_id="gen-trickle", epoch=1)

        # Simulate 10 polls over 0.25s
        for _ in range(10):
            clock.advance(0.025)
            res = reader.read_step()
            self.assertEqual(res.status, CaptureReadStatus.PENDING)

        self.assertEqual(reader.buffer_len, 100)

        # Now advance past 0.750s total deadline
        clock.advance(0.600)  # total elapsed: 0.850s > 0.750s
        res = reader.read_step()
        self.assertEqual(res.status, CaptureReadStatus.STALL)
        self.assertEqual(res.discarded_bytes, 100)
        self.assertEqual(reader.total_discarded_bytes, 100)
        self.assertEqual(reader.buffer_len, 0)

    def test_t10_eof_and_descriptor_error(self):
        """T10: Producer EOF with partial bytes discards partial and reports EOF; OS error reports ERROR."""
        clock = FakeControllableClock()
        reader = AudioCaptureReader(block_size_bytes=8192, clock=clock)

        # 1. EOF scenario with partial
        class ClosingStream:
            def __init__(self):
                self.calls = 0

            def read(self, n):
                self.calls += 1
                if self.calls == 1:
                    return b"partial_4096" + (b"\x00" * 4084)  # 4096 bytes
                return b""

        proc = SimpleNamespace(poll=lambda: 0)  # terminated process
        stream = ClosingStream()
        reader.attach(stream, generation_id="gen-eof", epoch=1)

        res1 = reader.read_step()
        self.assertEqual(res1.status, CaptureReadStatus.PENDING)
        self.assertEqual(reader.buffer_len, 4096)

        # Next read sees process terminated / stream ended -> EOF
        reader._proc = proc
        res2 = reader.read_step()
        self.assertEqual(res2.status, CaptureReadStatus.EOF)
        self.assertEqual(res2.discarded_bytes, 4096)
        self.assertEqual(reader.buffer_len, 0)
        self.assertEqual(reader.total_discarded_bytes, 4096)

        # 2. OS Error scenario
        class ErrorStream:
            def read(self, n):
                raise OSError("Simulated I/O failure")

        reader.attach(ErrorStream(), generation_id="gen-err", epoch=2)
        res3 = reader.read_step()
        self.assertEqual(res3.status, CaptureReadStatus.ERROR)
        self.assertIn("Simulated I/O failure", res3.reason)

    def test_t16_output_sink_stall_deadline(self):
        """T16: Output writer enforces bounded deadline (500 ms) and returns error on stall."""
        clock = FakeControllableClock()
        writer = AudioOutputWriter(block_size_bytes=8192, write_deadline_sec=0.500, clock=clock)

        # Stalled stdin that fails select.select
        fake_stdin = SimpleNamespace(closed=False, fileno=lambda: 999)
        fake_proc = SimpleNamespace(stdin=fake_stdin)

        with patch("select.select", return_value=([], [], [])):
            res = writer.write_block(fake_proc, b"\x00" * 8192)
            self.assertFalse(res.success)
            self.assertEqual(res.error, "write_timeout_sink_stalled")

    def test_t16b_real_pipe_writer_obeys_deadline_with_partial_capacity(self):
        """T16b: Real OS pipe with only 4,096 free bytes never blocks writer beyond write_deadline_sec."""
        rfd, wfd = os.pipe()
        writer_stream = None
        try:
            pipe_size = fcntl.fcntl(wfd, 1031, 4096)  # F_SETPIPE_SZ
            os.write(wfd, b"F" * max(0, pipe_size - 4096))
            writer_stream = os.fdopen(wfd, "wb", buffering=8192)
            proc = SimpleNamespace(stdin=writer_stream)
            writer = AudioOutputWriter(block_size_bytes=8192, poll_timeout_sec=0.015, write_deadline_sec=0.05)

            t0 = time.monotonic()
            res = writer.write_block(proc, b"P" * 8192, wait_for_completion=True)
            elapsed = time.monotonic() - t0

            self.assertLess(elapsed, 0.25, "Writer blocked past deadline on partial pipe capacity")
            self.assertFalse(res.success)
            self.assertEqual(res.samples_written, 0)
            self.assertEqual(res.bytes_written, 4096)
            self.assertEqual(res.partial_discarded_bytes, 4096)
            self.assertEqual(res.original_block_bytes, 8192)
            self.assertTrue(res.requires_reconnect)
            self.assertEqual(res.error, "write_timeout_sink_stalled")
        finally:
            if writer_stream is not None:
                try:
                    raw = writer_stream.detach() if hasattr(writer_stream, "detach") else None
                    if raw is not None:
                        raw.close()
                    else:
                        writer_stream.close()
                except Exception:
                    pass
            try:
                os.close(rfd)
            except Exception:
                pass

    def test_t16c_real_pipe_writer_completes_when_consumer_drains(self):
        """T16c: Real OS pipe with partial capacity completes full 8,192-byte block when consumer drains."""
        rfd, wfd = os.pipe()
        writer_stream = None
        drained = bytearray()
        try:
            pipe_size = fcntl.fcntl(wfd, 1031, 4096)
            prefill_len = max(0, pipe_size - 4096)
            if prefill_len > 0:
                os.write(wfd, b"F" * prefill_len)
            writer_stream = os.fdopen(wfd, "wb", buffering=0)
            proc = SimpleNamespace(stdin=writer_stream)
            writer = AudioOutputWriter(block_size_bytes=8192, poll_timeout_sec=0.010, write_deadline_sec=0.300)

            def consumer():
                time.sleep(0.025)
                if prefill_len > 0:
                    os.read(rfd, prefill_len)
                while len(drained) < 8192:
                    chunk = os.read(rfd, 8192 - len(drained))
                    if not chunk:
                        break
                    drained.extend(chunk)

            t = threading.Thread(target=consumer, daemon=True)
            t.start()
            payload = (b"ABCD" * 2048)
            res = writer.write_block(proc, payload, wait_for_completion=True)
            t.join(timeout=0.5)

            self.assertTrue(res.success)
            self.assertEqual(res.samples_written, 2048)
            self.assertEqual(res.bytes_written, 8192)
            self.assertEqual(res.original_block_bytes, 8192)
            self.assertEqual(bytes(drained), payload)
        finally:
            if writer_stream is not None:
                try:
                    writer_stream.close()
                except Exception:
                    pass
            try:
                os.close(rfd)
            except Exception:
                pass

    def test_t16d_partial_writer_rejected_never_counts_full_block(self):
        """T16d: A stream accepting only 4 of 8,192 bytes must never report success=True or samples_written=2048."""
        class ShortWriter:
            closed = False

            def write(self, data):
                return 4

            def flush(self):
                return None

        proc = SimpleNamespace(stdin=ShortWriter())
        writer = AudioOutputWriter(block_size_bytes=8192, write_deadline_sec=0.05)
        res = writer.write_block(proc, b"x" * 8192)
        self.assertFalse(res.success)
        self.assertEqual(res.samples_written, 0)
        self.assertEqual(res.bytes_written, 4)
        self.assertEqual(res.partial_discarded_bytes, 8188)
        self.assertEqual(res.original_block_bytes, 8192)
        self.assertTrue(res.requires_reconnect)

    def test_t16e_write_block_default_yields_within_single_poll_step(self):
        """T16e: Default write_block yields after one bounded poll step without waiting for full deadline."""
        rfd, wfd = os.pipe()
        writer_stream = None
        try:
            pipe_size = fcntl.fcntl(wfd, 1031, 8192)
            os.write(wfd, b"X" * max(0, pipe_size - 4096))
            writer_stream = os.fdopen(wfd, "wb", buffering=0)
            proc = SimpleNamespace(stdin=writer_stream)
            writer = AudioOutputWriter(block_size_bytes=8192, poll_timeout_sec=0.020, write_deadline_sec=0.500)

            t0 = time.monotonic()
            res1 = writer.write_block(proc, b"A" * 8192)
            elapsed = time.monotonic() - t0

            self.assertLess(elapsed, 0.120, "Default write_block must yield after one poll step")
            self.assertTrue(res1.pending)
            self.assertFalse(res1.success)
            self.assertEqual(res1.bytes_written, 4096)
            self.assertEqual(res1.partial_discarded_bytes, 0)
            self.assertEqual(res1.original_block_bytes, 8192)
            self.assertTrue(writer.has_pending)
            self.assertEqual(writer.pending_remaining_bytes, 4096)

            # Drain the pipe and complete on the next step
            os.read(rfd, 65536)
            res2 = writer.write_step(proc)
            self.assertTrue(res2.success)
            self.assertFalse(res2.pending)
            self.assertEqual(res2.samples_written, 2048)
            self.assertEqual(res2.bytes_written, 8192)
            self.assertEqual(res2.original_block_bytes, 8192)
        finally:
            if writer_stream is not None:
                try:
                    writer_stream.close()
                except Exception:
                    pass
            try:
                os.close(rfd)
            except Exception:
                pass

    def test_t16f_overlapping_write_step_rejected_and_cancel_pending_separates_accounting(self):
        """T16f: write_step rejects new block while partial is pending; cancel_pending separates delivered vs discarded."""
        class ShortSink:
            closed = False

            def __init__(self):
                self.received = bytearray()

            def write(self, data):
                self.received.extend(data[:4])
                return 4

            def flush(self):
                pass

        sink = ShortSink()
        proc = SimpleNamespace(stdin=sink)
        writer = AudioOutputWriter(block_size_bytes=8192)

        first = writer.write_step(proc, data=b"A" * 8192, metadata={"block_id": 1})
        self.assertTrue(first.pending)
        self.assertEqual(first.bytes_written, 4)
        self.assertEqual(first.partial_discarded_bytes, 0)
        self.assertEqual(first.original_block_bytes, 8192)

        # Attempting to start a new block B while A is still pending must be rejected without writing B
        second = writer.write_step(proc, data=b"B" * 8192, metadata={"block_id": 2})
        self.assertFalse(second.success)
        self.assertTrue(second.pending)
        self.assertTrue(second.requires_reconnect)
        self.assertEqual(second.error, "pending_write_already_in_progress")
        self.assertEqual(bytes(sink.received), b"AAAA")

        # Explicit cancellation reports 4 bytes delivered and 8188 bytes discarded, requiring reconnect
        cancelled = writer.cancel_pending(reason="source_switch")
        self.assertFalse(cancelled.success)
        self.assertFalse(cancelled.pending)
        self.assertEqual(cancelled.bytes_written, 4)
        self.assertEqual(cancelled.partial_discarded_bytes, 8188)
        self.assertEqual(cancelled.original_block_bytes, 8192)
        self.assertTrue(cancelled.requires_reconnect)
        self.assertEqual(cancelled.metadata, {"block_id": 1})
        self.assertEqual(writer.total_discarded_bytes, 8188)
        self.assertFalse(writer.has_pending)

    def test_t03b_real_capture_pipe_responsive_without_new_bytes(self):
        """T03b: Real OS capture pipe without incoming bytes returns PENDING within poll timeout."""
        rfd, wfd = os.pipe()
        reader_stream = os.fdopen(rfd, "rb", buffering=0)
        try:
            proc = SimpleNamespace(stdout=reader_stream, poll=lambda: None)
            reader = AudioCaptureReader(block_size_bytes=8192, poll_timeout_sec=0.02, block_deadline_sec=0.10)
            reader.attach(proc, generation_id="gen-real", epoch=1, is_first_block=False)
            t0 = time.monotonic()
            res = reader.read_step(timeout_sec=0.02)
            elapsed = time.monotonic() - t0
            self.assertEqual(res.status, CaptureReadStatus.PENDING)
            self.assertLess(elapsed, 0.15)
        finally:
            reader_stream.close()
            os.close(wfd)

    def test_first_byte_time_and_pending_liveness_gauges(self):
        """Reader tracks first_byte_time, is_pending, and pending_age_ms accurately."""
        clock = FakeControllableClock(100.0)
        reader = AudioCaptureReader(block_size_bytes=8192, clock=clock)
        chunks = [b"A" * 4096, b"B" * 4096]
        stream = SimpleNamespace(read=lambda n: chunks.pop(0) if chunks else b"")
        reader.attach(stream, generation_id="gen-1", epoch=1, is_first_block=False)

        self.assertFalse(reader.is_pending)
        self.assertEqual(reader.pending_age_ms(), 0.0)

        res1 = reader.read_step()
        self.assertEqual(res1.status, CaptureReadStatus.PENDING)
        self.assertTrue(reader.is_pending)
        self.assertEqual(reader.first_byte_time, 100.0)

        clock.advance(0.050)
        self.assertAlmostEqual(reader.pending_age_ms(), 50.0, places=2)

        res2 = reader.read_step()
        self.assertEqual(res2.status, CaptureReadStatus.COMPLETE)
        self.assertEqual(res2.first_byte_time, 100.0)
        self.assertFalse(reader.is_pending)
        self.assertEqual(reader.pending_age_ms(), 0.0)

    def test_t17_cancel_and_detach_during_stall(self):
        """T17: Detach and cancel during stall discard partials immediately without hanging."""
        clock = FakeControllableClock()
        reader = AudioCaptureReader(block_size_bytes=8192, clock=clock)

        stream = io.BytesIO(b"unfinished_fragment_1234")
        reader.attach(stream, generation_id="gen-1", epoch=1)

        res = reader.read_step()
        self.assertEqual(res.status, CaptureReadStatus.PENDING)
        self.assertEqual(reader.buffer_len, len(b"unfinished_fragment_1234"))

        # Invalidate via epoch change
        discarded = reader.set_epoch(2)
        self.assertEqual(discarded, len(b"unfinished_fragment_1234"))
        self.assertEqual(reader.buffer_len, 0)

        # Detach
        discarded_detach = reader.detach()
        self.assertEqual(discarded_detach, 0)

    def test_retry_policy_backoff_and_reset(self):
        """Retry policy follows (250, 500, 1000, 2000) ms schedule and resets after 10 successes."""
        clock = FakeControllableClock(10.0)
        policy = CaptureRetryPolicy(
            backoff_steps=(0.250, 0.500, 1.000, 2.000),
            stable_blocks_for_reset=10,
            clock=clock,
        )

        self.assertTrue(policy.can_retry())

        # Failure 1 -> 250ms
        b1 = policy.record_failure()
        self.assertEqual(b1, 0.250)
        self.assertFalse(policy.can_retry())
        clock.advance(0.240)
        self.assertFalse(policy.can_retry())
        clock.advance(0.015)
        self.assertTrue(policy.can_retry())

        # Failure 2 -> 500ms
        b2 = policy.record_failure()
        self.assertEqual(b2, 0.500)
        clock.advance(0.501)
        self.assertTrue(policy.can_retry())

        # Failure 3 -> 1000ms
        b3 = policy.record_failure()
        self.assertEqual(b3, 1.000)
        clock.advance(1.001)
        self.assertTrue(policy.can_retry())

        # Failure 4 -> 2000ms
        b4 = policy.record_failure()
        self.assertEqual(b4, 2.000)
        clock.advance(2.001)
        self.assertTrue(policy.can_retry())

        # Consecutive successes reset backoff
        for _ in range(9):
            policy.record_success()
            self.assertEqual(policy.consecutive_failures, 4)

        policy.record_success()  # 10th success
        self.assertEqual(policy.consecutive_failures, 0)
        self.assertEqual(policy.current_backoff, 0.250)

    def test_simulated_1000_cycles_memory_stability(self):
        """Stress test: 1,000 rapid cycles of attach, partial reads, stall, cancel, and detach."""
        clock = FakeControllableClock()
        reader = AudioCaptureReader(block_size_bytes=8192, clock=clock)

        for i in range(1000):
            data = bytes([i % 128]) * (i % 2000)
            stream = io.BytesIO(data)
            reader.attach(stream, generation_id=f"gen-{i}", epoch=i)
            reader.read_step()
            if i % 3 == 0:
                clock.advance(2.0)
                reader.read_step()  # may stall
            elif i % 3 == 1:
                reader.set_epoch(i + 1)
            reader.detach()

            # Invariant: buffer must not grow unboundedly
            self.assertLessEqual(reader.buffer_len, 8192)


if __name__ == "__main__":
    unittest.main()
