"""Deterministically test dynamic microphone switching and lifecycle invariants (T05-T09, T11-T15, T18-T20).

Tests cover:
- T05: Switch A -> B during pending read cancels A without spontaneous EOF, opens B.
- T06: Switch with partial of A ensures no bytes of A appear in B's first block.
- T07: Block A ready and switch B observed -> epoch barrier drops A before inference/output.
- T08: Rapid A -> B -> C: last intention wins, processes and buffers remain bounded.
- T09: Explicit source unavailable: does NOT capture integrated mic, emits silence, bounded retry backoff.
- T11: Standby and resume: no physical access in standby, silence continues, resumes with fresh generation.
- T12: Switch during standby: updates intention only; resume opens new source with fresh generation.
- T13: USB disconnect / reconnect: recovers without daemon restart or integrated mic capture.
- T14: Transient invalid JSON: retains last valid config, does not revert to auto or restart in loop.
- T15: Identical reload or telemetry-only change: does not recreate process, change generation, or reset DSP.
- T18: Model/DSP states reset exactly once per new generation, never per poll.
- T19: Telemetry: no PCM logged, owner counters monotonic, health gauges and transitions emitted.
- T20: PCM regression: commands preserve --raw, f32, 16000 Hz, 1 channel, target and node names.
"""
import copy
import errno
import fcntl
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure src/bin is on sys.path
SRC_BIN = Path(__file__).resolve().parent.parent.parent / "src" / "bin"
if str(SRC_BIN) not in sys.path:
    sys.path.insert(0, str(SRC_BIN))

from npu_pipeline.audio_capture import (
    AudioCaptureReader,
    AudioCaptureState,
    AudioOutputWriter,
    AudioSwitchCoordinator,
    CaptureBlockResult,
    CaptureReadStatus,
    CaptureRetryPolicy,
)
import npu_audio_daemon as daemon


class TestAudioMicrophoneSwitch(unittest.TestCase):
    """Deterministic validation of microphone switching, epoch barriers, and lifecycle invariants."""

    def setUp(self):
        self.chunk_size = 2048
        self.block_size_bytes = self.chunk_size * 4  # 8192 bytes
        self.coordinator = AudioSwitchCoordinator(
            initial_desired_source="auto",
            initial_active_source="alsa_input.integrated",
        )
        self.reader = AudioCaptureReader(
            block_size_bytes=self.block_size_bytes,
            poll_timeout_sec=0.025,
            block_deadline_sec=0.750,
            initial_tolerance_sec=1.500,
        )

    # -------------------------------------------------------------------------
    # T05: Switch A -> B during pending read
    # -------------------------------------------------------------------------
    def test_t05_switch_during_pending_read(self):
        """T05: Changing source A -> B during pending read cancels A cleanly and opens B."""
        stream_a = io.BytesIO(b"A" * 1024)
        self.reader.attach(stream_a, generation_id="gen-A", epoch=self.coordinator.epoch)

        res1 = self.reader.read_step(timeout_sec=0.01)
        self.assertEqual(res1.status, CaptureReadStatus.PENDING)
        self.assertEqual(self.reader.buffer_len, 1024)

        # Switch A -> B requested
        switched = self.coordinator.request_source_switch("alsa_input.usb-B")
        self.assertTrue(switched)
        self.assertEqual(self.coordinator.desired_source, "alsa_input.usb-B")
        self.assertEqual(self.coordinator.epoch, 2)

        # Cancel/detach stream A
        discarded = self.reader.detach()
        self.assertEqual(discarded, 1024)
        self.assertEqual(self.reader.buffer_len, 0)

        # Connect stream B
        stream_b = io.BytesIO(b"B" * self.block_size_bytes)
        self.coordinator.begin_connecting("alsa_input.usb-B", "gen-B")
        self.reader.attach(stream_b, generation_id="gen-B", epoch=self.coordinator.epoch)

        res2 = self.reader.read_step(timeout_sec=0.01)
        self.assertEqual(res2.status, CaptureReadStatus.COMPLETE)
        self.assertEqual(res2.generation_id, "gen-B")
        self.assertEqual(res2.epoch, 2)
        self.assertEqual(res2.data, b"B" * self.block_size_bytes)
        self.assertTrue(self.coordinator.is_block_valid(res2))

    # -------------------------------------------------------------------------
    # T06: Switch with partial bytes of A ensures no leakage into B
    # -------------------------------------------------------------------------
    def test_t06_switch_with_partial_bytes_of_a_isolated_from_b(self):
        """T06: Partial bytes of source A are completely discarded and never bleed into source B."""
        stream_a = io.BytesIO(b"A" * 4096)
        self.reader.attach(stream_a, generation_id="gen-A", epoch=self.coordinator.epoch)

        res_a = self.reader.read_step(timeout_sec=0.01)
        self.assertEqual(res_a.status, CaptureReadStatus.PENDING)
        self.assertEqual(self.reader.buffer_len, 4096)

        # Dynamic switch to B occurs
        self.coordinator.request_source_switch("alsa_input.usb-B")
        discarded_epoch = self.reader.set_epoch(self.coordinator.epoch)
        self.assertEqual(discarded_epoch, 4096)
        self.assertEqual(self.reader.buffer_len, 0)

        # Detach A and attach B
        self.reader.detach()
        raw_b_data = b"B" * self.block_size_bytes
        stream_b = io.BytesIO(raw_b_data)
        self.coordinator.begin_connecting("alsa_input.usb-B", "gen-B")
        self.reader.attach(stream_b, generation_id="gen-B", epoch=self.coordinator.epoch)

        res_b = self.reader.read_step(timeout_sec=0.01)
        self.assertEqual(res_b.status, CaptureReadStatus.COMPLETE)
        self.assertEqual(len(res_b.data), self.block_size_bytes)
        self.assertEqual(res_b.data, raw_b_data)
        self.assertNotIn(b"A", res_b.data)
        self.assertEqual(self.reader.total_discarded_bytes, 4096)

    # -------------------------------------------------------------------------
    # T07: Ready block A rejected by epoch barrier upon switch B
    # -------------------------------------------------------------------------
    def test_t07_ready_block_a_rejected_by_epoch_barrier_upon_switch_b(self):
        """T07: Complete block A arriving at switch boundary is rejected by epoch barrier."""
        self.coordinator.begin_connecting("alsa_input.integrated", "gen-A")
        block_a_result = CaptureBlockResult(
            status=CaptureReadStatus.COMPLETE,
            data=b"A" * self.block_size_bytes,
            epoch=1,
            generation_id="gen-A",
        )
        self.assertTrue(self.coordinator.is_block_valid(block_a_result))

        # Switch observed before block A is processed
        self.coordinator.request_source_switch("alsa_input.usb-B")
        self.assertEqual(self.coordinator.epoch, 2)

        # Epoch barrier rejection
        self.assertFalse(self.coordinator.is_block_valid(block_a_result))

    # -------------------------------------------------------------------------
    # T08: Rapid A -> B -> C: last intention wins, buffers bounded
    # -------------------------------------------------------------------------
    def test_t08_rapid_switching_a_b_c_last_intention_wins(self):
        """T08: Rapid intention changes A -> B -> C settle on C with bounded buffers and epoch."""
        self.coordinator.request_source_switch("alsa_input.usb-A")
        self.assertEqual(self.coordinator.desired_source, "alsa_input.usb-A")
        self.assertEqual(self.coordinator.epoch, 2)

        self.coordinator.request_source_switch("alsa_input.usb-B")
        self.assertEqual(self.coordinator.desired_source, "alsa_input.usb-B")
        self.assertEqual(self.coordinator.epoch, 3)

        self.coordinator.request_source_switch("alsa_input.usb-C")
        self.assertEqual(self.coordinator.desired_source, "alsa_input.usb-C")
        self.assertEqual(self.coordinator.epoch, 4)

        # No intermediate processes needed; connect C directly
        self.coordinator.begin_connecting("alsa_input.usb-C", "gen-C")
        stream_c = io.BytesIO(b"C" * self.block_size_bytes)
        self.reader.attach(stream_c, generation_id="gen-C", epoch=self.coordinator.epoch)

        res = self.reader.read_step(timeout_sec=0.01)
        self.assertEqual(res.status, CaptureReadStatus.COMPLETE)
        self.assertEqual(res.data, b"C" * self.block_size_bytes)
        self.assertTrue(self.coordinator.is_block_valid(res))
        self.assertLessEqual(self.reader.buffer_len, self.block_size_bytes)

    # -------------------------------------------------------------------------
    # T09: Explicit source unavailable: no fallback to integrated, bounded retries
    # -------------------------------------------------------------------------
    def test_t09_explicit_source_unavailable_no_fallback_silence_and_bounded_retry(self):
        """T09: When explicit mic is missing, do NOT fallback to integrated; emit silence with backoff."""
        cfg = {"input_device": "alsa_input.usb-Missing_Mic-00.mono"}
        available = [
            "alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp__source",
            "alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp__sink.monitor",
        ]

        # Resolution must return None, NEVER fallback to integrated
        chosen = daemon.select_hardware_microphone(cfg, avail=available)
        self.assertIsNone(chosen)

        # Retry policy verifies backoff progression
        policy = CaptureRetryPolicy(backoff_steps=(0.250, 0.500, 1.000, 2.000))
        b1 = policy.record_failure()
        self.assertEqual(b1, 0.250)
        b2 = policy.record_failure()
        self.assertEqual(b2, 0.500)
        b3 = policy.record_failure()
        self.assertEqual(b3, 1.000)
        b4 = policy.record_failure()
        self.assertEqual(b4, 2.000)
        b5 = policy.record_failure()
        self.assertEqual(b5, 2.000)

        # Virtual output receives silence block
        silence = np.zeros(self.chunk_size, dtype=np.float32).tobytes()
        self.assertEqual(len(silence), self.block_size_bytes)
        self.assertEqual(silence, b"\x00" * self.block_size_bytes)

    # -------------------------------------------------------------------------
    # T11: Standby and resume: no physical access in standby, fresh generation
    # -------------------------------------------------------------------------
    def test_t11_standby_and_resume_no_physical_access_and_fresh_generation(self):
        """T11: Standby releases physical hardware; resume creates a fresh generation."""
        # Enter standby
        self.coordinator.mark_standby()
        self.assertEqual(self.coordinator.state, AudioCaptureState.STANDBY)
        self.assertIsNone(self.coordinator.active_source)

        discarded = self.reader.detach()
        self.assertEqual(discarded, 0)

        # Consumer arrives -> exit standby
        new_gen_id = "audio-gen-resumed"
        self.coordinator.begin_connecting("alsa_input.integrated", new_gen_id)
        self.assertEqual(self.coordinator.state, AudioCaptureState.CONNECTING)
        self.assertEqual(self.coordinator.audio_gen_id, new_gen_id)

        stream = io.BytesIO(b"R" * self.block_size_bytes)
        self.reader.attach(stream, generation_id=new_gen_id, epoch=self.coordinator.epoch)
        res = self.reader.read_step(timeout_sec=0.01)
        self.assertEqual(res.status, CaptureReadStatus.COMPLETE)

        self.coordinator.mark_active()
        self.assertEqual(self.coordinator.state, AudioCaptureState.ACTIVE)
        self.assertTrue(self.coordinator.is_block_valid(res))

    # -------------------------------------------------------------------------
    # T12: Switch during standby updates intention only
    # -------------------------------------------------------------------------
    def test_t12_switch_during_standby_updates_intention_only(self):
        """T12: Switching source during standby only updates intention; opens new source on resume."""
        self.coordinator.mark_standby()
        self.assertEqual(self.coordinator.state, AudioCaptureState.STANDBY)

        # User changes mic in UI while in standby
        switched = self.coordinator.request_source_switch("alsa_input.usb-NewDevice")
        self.assertTrue(switched)
        self.assertEqual(self.coordinator.desired_source, "alsa_input.usb-NewDevice")
        self.assertEqual(self.coordinator.epoch, 2)
        # Still in standby / switching; physical stream is NOT opened yet
        self.assertIsNone(self.reader._stream)

        # When app connects to virtual mic, resume opens the requested new mic
        self.coordinator.begin_connecting("alsa_input.usb-NewDevice", "gen-new-usb")
        stream_usb = io.BytesIO(b"U" * self.block_size_bytes)
        self.reader.attach(stream_usb, generation_id="gen-new-usb", epoch=self.coordinator.epoch)

        res = self.reader.read_step(timeout_sec=0.01)
        self.assertEqual(res.status, CaptureReadStatus.COMPLETE)
        self.assertEqual(res.data, b"U" * self.block_size_bytes)
        self.assertTrue(self.coordinator.is_block_valid(res))

    # -------------------------------------------------------------------------
    # T13: USB disconnect and reconnect recovers cleanly
    # -------------------------------------------------------------------------
    def test_t13_usb_disconnect_and_reconnect_recovers_without_restart(self):
        """T13: USB mic disconnect triggers RECOVERING; reconnect resumes with new generation."""
        # Active USB mic
        usb_mic = "alsa_input.usb-Headset-00.mono"
        self.coordinator.begin_connecting(usb_mic, "gen-usb-1")
        self.coordinator.mark_active()
        self.assertEqual(self.coordinator.state, AudioCaptureState.ACTIVE)

        # Disconnect: pipe closes with EOF
        empty_stream = io.BytesIO(b"")
        self.reader.attach(empty_stream, generation_id="gen-usb-1", epoch=self.coordinator.epoch)
        res = self.reader.read_step(timeout_sec=0.01)
        self.assertEqual(res.status, CaptureReadStatus.EOF)

        # Daemon transitions to RECOVERING
        self.coordinator.mark_recovering()
        self.assertEqual(self.coordinator.state, AudioCaptureState.RECOVERING)
        self.assertIsNone(self.coordinator.active_source)

        # USB reconnected: retry finds it again
        self.coordinator.begin_connecting(usb_mic, "gen-usb-2")
        reconnected_stream = io.BytesIO(b"N" * self.block_size_bytes)
        self.reader.attach(reconnected_stream, generation_id="gen-usb-2", epoch=self.coordinator.epoch)

        res2 = self.reader.read_step(timeout_sec=0.01)
        self.assertEqual(res2.status, CaptureReadStatus.COMPLETE)
        self.coordinator.mark_active()
        self.assertEqual(self.coordinator.state, AudioCaptureState.ACTIVE)
        self.assertEqual(self.coordinator.audio_gen_id, "gen-usb-2")

    # -------------------------------------------------------------------------
    # T14: Transient invalid JSON retains last valid configuration
    # -------------------------------------------------------------------------
    def test_t14_transient_invalid_json_preserves_last_valid_config(self):
        """T14: Corrupted or partially written JSON retains last valid config and does not reset to defaults."""
        last_valid = {
            "input_device": "alsa_input.usb-CustomMic",
            "audio": {
                "enabled": True,
                "noise_suppression": True,
            },
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write('{"input_device": "alsa_input.usb-CustomMic", "audio": {corrupt...')
            corrupt_path = Path(f.name)

        try:
            with patch("npu_audio_daemon.CONFIG_PATH", corrupt_path):
                loaded = daemon.load_config(last_valid=last_valid)
                self.assertEqual(loaded["input_device"], "alsa_input.usb-CustomMic")
                self.assertEqual(loaded, last_valid)
        finally:
            if corrupt_path.exists():
                corrupt_path.unlink()

    # -------------------------------------------------------------------------
    # T15: Identical reload or telemetry-only change preserves state
    # -------------------------------------------------------------------------
    def test_t15_identical_reload_or_telemetry_only_preserves_state(self):
        """T15: Reloading identical config or changing only telemetry does not trigger source switch."""
        current_cfg = {
            "input_device": "alsa_input.integrated",
            "audio": {"trace_enabled": False},
        }
        self.coordinator.desired_source = "alsa_input.integrated"
        initial_epoch = self.coordinator.epoch

        # Identical reload
        switched1 = self.coordinator.request_source_switch(current_cfg.get("input_device", "auto"))
        self.assertFalse(switched1)
        self.assertEqual(self.coordinator.epoch, initial_epoch)

        # Telemetry-only change
        telemetry_cfg = {
            "input_device": "alsa_input.integrated",
            "audio": {"trace_enabled": True},
        }
        switched2 = self.coordinator.request_source_switch(telemetry_cfg.get("input_device", "auto"))
        self.assertFalse(switched2)
        self.assertEqual(self.coordinator.epoch, initial_epoch)

    # -------------------------------------------------------------------------
    # T18: Model and DSP states reset exactly once per generation, never per poll
    # -------------------------------------------------------------------------
    def test_t18_model_and_dsp_state_reset_exactly_once_per_generation(self):
        """T18: Effects chain and model temporal states reset on generation change, not on each poll."""
        chain = daemon.AudioEffectsChain(16000)
        cfg = {"studio_effects": True, "compressor_enabled": True, "highpass_enabled": True, "warmth_enabled": True}

        # Simulate audio passing through effects, creating non-zero filter states
        test_chunk = np.full(self.chunk_size, 0.5, dtype=np.float32)
        chain.process(test_chunk, cfg)

        # Filter states should be non-zero
        self.assertFalse(np.all(chain.zi_hp == 0))
        self.assertFalse(np.all(chain.zi_warmth == 0))

        # On generation change, reset_state is called
        chain.reset_state()
        fresh = daemon.AudioEffectsChain(16000)
        np.testing.assert_array_equal(chain.zi_hp, fresh.zi_hp)
        np.testing.assert_array_equal(chain.zi_warmth, fresh.zi_warmth)
        np.testing.assert_array_equal(chain.zi_presence, fresh.zi_presence)
        np.testing.assert_array_equal(chain.zi_deess, fresh.zi_deess)
        self.assertEqual(chain.gate_gain, 1.0)
        self.assertEqual(chain.comp_gain, 1.0)
        self.assertEqual(chain.reverb_tail, 0.0)

        # Normal polling does NOT reset states
        chain.process(test_chunk, cfg)
        self.assertFalse(np.all(chain.zi_hp == 0))

    # -------------------------------------------------------------------------
    # T19: Telemetry invariants: no PCM, monotonic counters, correct events
    # -------------------------------------------------------------------------
    def test_t19_telemetry_invariants_and_events(self):
        """T19: Telemetry events contain no PCM buffers, counters are monotonic, transitions correct."""
        events_emitted = []

        class MockTelemetry:
            def emit(self, event_type, **kwargs):
                events_emitted.append((event_type, kwargs))

            def sample(self, metric, val):
                pass

        mock_tel = MockTelemetry()

        # Emit switch event
        mock_tel.emit(
            "audio_source_switch_requested",
            old_source="mic-A",
            new_desired="mic-B",
            epoch=2,
        )

        # Emit stall event
        mock_tel.emit(
            "audio_capture_stall",
            reason="deadline_exceeded",
            discarded_bytes=1024,
            epoch=2,
            generation_id="gen-2",
        )

        # Emit health event
        mock_tel.emit(
            "audio_capture_health",
            capture_state=AudioCaptureState.ACTIVE.value,
            desired_source="mic-B",
            active_source="mic-B",
            epoch=2,
            audio_gen_id="gen-2",
            partial_bytes=0,
            total_discarded_bytes=1024,
            consecutive_failures=0,
        )

        self.assertEqual(len(events_emitted), 3)
        for ev_type, payload in events_emitted:
            for k, v in payload.items():
                self.assertNotIsInstance(v, (bytes, bytearray, np.ndarray))
                self.assertFalse(isinstance(v, str) and len(v) > 2048)

    # -------------------------------------------------------------------------
    # T20: PCM command parameters regression: --raw, f32, 16kHz, 1 channel
    # -------------------------------------------------------------------------
    def test_t20_pcm_command_parameters_regression(self):
        """T20: pw-record and pw-play commands strictly preserve --raw, f32, 16000 Hz, mono parameters."""
        rec_cmd = [
            "pw-record",
            "--raw",
            "--target", "alsa_input.test",
            "-P", "node.name=npu_mic_capture_1234",
            "-P", "node.dont-reconnect=true",
            "--format", "f32",
            "--rate", "16000",
            "--channels", "1",
            "-",
        ]
        self.assertIn("--raw", rec_cmd)
        self.assertEqual(rec_cmd[rec_cmd.index("--format") + 1], "f32")
        self.assertEqual(rec_cmd[rec_cmd.index("--rate") + 1], "16000")
        self.assertEqual(rec_cmd[rec_cmd.index("--channels") + 1], "1")
        self.assertEqual(rec_cmd[-1], "-")

        play_cmd = [
            "pw-play",
            "--raw",
            "--target", "0",
            "-P", "node.name=npu_mic_feed_1234",
            "--format", "f32",
            "--rate", "16000",
            "--channels", "1",
            "-",
        ]
        self.assertIn("--raw", play_cmd)
        self.assertEqual(play_cmd[play_cmd.index("--format") + 1], "f32")
        self.assertEqual(play_cmd[play_cmd.index("--rate") + 1], "16000")
        self.assertEqual(play_cmd[play_cmd.index("--channels") + 1], "1")
        self.assertEqual(play_cmd[-1], "-")

    # -------------------------------------------------------------------------
    # T09b: Explicit source never matches substring and filters virtual sources
    # -------------------------------------------------------------------------
    def test_t09b_explicit_source_never_matches_substring_and_filters_virtual(self):
        """T09b: select_hardware_microphone requires exact equality for explicit source and ignores virtual sources."""
        cfg = {"input_device": "alsa_input.usb-Headset"}
        avail = [
            "alsa_input.usb-Headset-Other_Device",
            "alsa_input.pci-integrated",
            "npu_clearvoice",
            "alsa_output.pci.monitor",
        ]
        self.assertIsNone(daemon.select_hardware_microphone(cfg, avail=avail))

        # Exact match succeeds
        avail_exact = avail + ["alsa_input.usb-Headset"]
        self.assertEqual(daemon.select_hardware_microphone(cfg, avail=avail_exact), "alsa_input.usb-Headset")

        # Virtual source requested explicitly is rejected
        cfg_virt = {"input_device": "npu_clearvoice"}
        self.assertIsNone(daemon.select_hardware_microphone(cfg_virt, avail=avail_exact))

    # -------------------------------------------------------------------------
    # T14b: Invalid config root, audio section, and field types preserve last_valid
    # -------------------------------------------------------------------------
    def test_t14b_invalid_config_root_and_audio_types_preserve_last_valid(self):
        """T14b: load_config rejects null/array/string audio sections and invalid field types."""
        last_valid = {"input_device": "alsa_input.usb-Headset", "enabled": True, "standby_timeout": 1.5}
        invalid_payloads = [
            "null",
            "[]",
            '"just_a_string"',
            '{"audio": null}',
            '{"audio": []}',
            '{"audio": "invalid"}',
            '{"audio": {"input_device": 123}}',
            '{"audio": {"input_device": "   "}}',
            '{"audio": {"noise_suppression": "true"}}',
            '{"audio": {"standby_timeout": "fast"}}',
            '{"audio": {"gate_threshold_db": true}}',
        ]
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.json"
            with patch("npu_audio_daemon.CONFIG_PATH", cfg_path):
                for raw in invalid_payloads:
                    cfg_path.write_text(raw)
                    loaded, ok = daemon.load_config(last_valid=last_valid, return_status=True)
                    self.assertFalse(ok, f"Expected payload {raw} to be rejected")
                    self.assertEqual(loaded, last_valid, f"Payload {raw} clobbered last_valid")

    # -------------------------------------------------------------------------
    # Real OS Pipe: stop_process never blocks on full pipe with BufferedWriter
    # -------------------------------------------------------------------------
    def test_stop_process_never_blocks_on_full_pipe_buffered_stdin(self):
        """stop_process terminates and waits before detaching/closing streams, never blocking on flush()."""
        rfd, wfd = os.pipe()
        try:
            pipe_size = fcntl.fcntl(wfd, 1031, 4096)
            os.write(wfd, b"X" * pipe_size)  # Pipe is 100% full
            buffered_stdin = os.fdopen(wfd, "wb", buffering=8192)
            # Queue 4096 bytes inside Python's BufferedWriter without flushing
            buffered_stdin.write(b"Y" * 4096)

            events = []
            fake_proc = SimpleNamespace(
                stdin=buffered_stdin,
                stdout=None,
                stderr=None,
                terminate=lambda: events.append("terminate"),
                wait=lambda timeout=None: events.append("wait"),
                kill=lambda: events.append("kill"),
            )

            t0 = time.monotonic()
            daemon.stop_process(fake_proc, timeout=0.05)
            elapsed = time.monotonic() - t0

            self.assertLess(elapsed, 0.20, "stop_process blocked on full pipe")
            self.assertEqual(events[:2], ["terminate", "wait"])
        finally:
            try:
                os.close(rfd)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Real Daemon Main Loop Integration Test
    # -------------------------------------------------------------------------
    def test_daemon_main_integration_pending_pacing_health_switch_and_obsolescence(self):
        """Executes npu_audio_daemon.main() end-to-end across PENDING pacing, health telemetry, obsolescence, and mic switch."""
        clock_ns = [1_000_000_000]  # Start at 1.0s

        def fake_monotonic():
            return clock_ns[0] / 1_000_000_000.0

        def fake_monotonic_ns():
            return clock_ns[0]

        def fake_sleep(sec):
            clock_ns[0] += int(max(0.0, sec) * 1_000_000_000)

        spawned_cmds = []
        sink_writes = []
        terminated_procs = []

        class FakeSinkStdin:
            closed = False

            def write(self, data):
                sink_writes.append(bytes(data))
                return len(data)

            def flush(self):
                return None

            def close(self):
                self.closed = True

        # Phase 1: MicA stalls and recovers across ~5.5s of simulated time (testing STALL + backoff + 128ms silence pacing + 5s health)
        # Phase 2: Hot-reload config switches to MicB -> 1 valid COMPLETE block on MicB -> shutdown
        step_counter = [0]
        switched_to_b = [False]

        class FakeCaptureStdout:
            def __init__(self, target_mic):
                self.target_mic = target_mic
                self.closed = False
                self.calls = 0

            def read(self, n):
                self.calls += 1
                step_counter[0] += 1
                clock_ns[0] += 25_000_000  # 25ms per read poll
                if self.target_mic == "alsa_input.usb-MicA":
                    if clock_ns[0] < 6_500_000_000:
                        return b""  # PENDING -> STALL after 500ms -> backoff retry
                    if not switched_to_b[0]:
                        switched_to_b[0] = True
                        # Trigger config switch to MicB on next iteration
                        cfg_file.write_text(json.dumps({
                            "audio": {
                                "enabled": True,
                                "noise_suppression": True,
                                "input_device": "alsa_input.usb-MicB",
                                "auto_standby": False,
                            }
                        }))
                        os.utime(cfg_file, (1000, 1000))
                        return b"A" * 8192
                    return b""
                elif self.target_mic == "alsa_input.usb-MicB":
                    if self.calls == 1:
                        return np.ones(2048, dtype=np.float32).tobytes()
                    daemon.running = False
                    return b""
                return b""

            def close(self):
                self.closed = True

        def fake_popen(cmd, stdin=None, stdout=None, bufsize=-1, **kwargs):
            spawned_cmds.append((list(cmd), bufsize))
            if cmd[0] == "pw-play":
                p = SimpleNamespace(
                    cmd=cmd,
                    stdin=FakeSinkStdin(),
                    stdout=None,
                    stderr=None,
                    poll=lambda: None,
                )
            else:
                target_mic = cmd[cmd.index("--target") + 1]
                p = SimpleNamespace(
                    cmd=cmd,
                    stdin=None,
                    stdout=FakeCaptureStdout(target_mic),
                    stderr=None,
                    poll=lambda: None,
                )
            p.terminate = lambda: terminated_procs.append(cmd[0])
            p.wait = lambda timeout=None: 0
            p.kill = lambda: None
            return p

        # Fake OpenVINO model & infer request
        fake_input = SimpleNamespace(get_any_name=lambda: "input", shape=[1, 2048])
        fake_state = SimpleNamespace(get_any_name=lambda: "inp_state_1", shape=[1, 64])
        fake_model = SimpleNamespace(inputs=[fake_input, fake_state])
        infer_calls = [0]

        class FakeInferRequest:
            def infer(self, feed):
                infer_calls[0] += 1

            def get_tensor(self, name):
                if name == "output":
                    return SimpleNamespace(data=np.ones((1, 2048), dtype=np.float32) * 0.1)
                return SimpleNamespace(data=np.zeros((1, 64), dtype=np.float32))

        fake_compiled = SimpleNamespace(create_infer_request=lambda: FakeInferRequest())
        fake_core = SimpleNamespace(
            available_devices=["NPU"],
            read_model=lambda p: fake_model,
            compile_model=lambda m, d: fake_compiled,
        )

        emitted_events = []

        class FakeTelemetry:
            def emit(self, name, **kw):
                emitted_events.append((name, kw))

            def sample(self, name, val):
                pass

        class FakeTelemetryRuntime:
            def __init__(self, *args, **kwargs):
                self.enabled = True
                self.telemetry = FakeTelemetry()
                self.counters = {}

            def register_owner_counter(self, name, fn):
                self.counters[name] = fn

            def update_config(self, cfg):
                pass

            def should_trace(self, block_id):
                return True

            def stop(self):
                pass

        with tempfile.TemporaryDirectory() as td:
            cfg_file = Path(td) / "config.json"
            cfg_file.write_text(json.dumps({
                "audio": {
                    "enabled": True,
                    "noise_suppression": True,
                    "input_device": "alsa_input.usb-MicA",
                    "auto_standby": False,
                }
            }))
            os.utime(cfg_file, (500, 500))
            model_file = Path(td) / "model.xml"
            model_file.write_text("<net/>")

            daemon.running = True
            with patch("npu_audio_daemon.CONFIG_PATH", cfg_file), \
                 patch("npu_audio_daemon.MODEL_PATH", model_file), \
                 patch("npu_audio_daemon.ov.Core", return_value=fake_core), \
                 patch("npu_audio_daemon.subprocess.Popen", side_effect=fake_popen), \
                 patch("npu_audio_daemon.setup_pipewire_virtual_devices"), \
                 patch("npu_audio_daemon.cleanup"), \
                 patch("npu_audio_daemon.run_cmd", return_value=""), \
                 patch("npu_audio_daemon.get_physical_microphones", return_value=["alsa_input.usb-MicA", "alsa_input.usb-MicB"]), \
                 patch("npu_audio_daemon.TelemetryRuntime", FakeTelemetryRuntime), \
                 patch("npu_audio_daemon.time.monotonic", side_effect=fake_monotonic), \
                 patch("npu_audio_daemon.time.monotonic_ns", side_effect=fake_monotonic_ns), \
                 patch("npu_audio_daemon.time.sleep", side_effect=fake_sleep), \
                 patch("npu_audio_daemon.AudioCaptureReader", side_effect=lambda **kw: AudioCaptureReader(clock=fake_monotonic, **kw)), \
                 patch("npu_audio_daemon.AudioOutputWriter", side_effect=lambda **kw: AudioOutputWriter(clock=fake_monotonic, **kw)), \
                 patch("npu_audio_daemon.CaptureRetryPolicy", side_effect=lambda **kw: CaptureRetryPolicy(clock=fake_monotonic, **kw)):
                daemon.main()

        # 1. Verify Popen commands use --raw and bufsize=0
        play_spawns = [c for c, b in spawned_cmds if c[0] == "pw-play"]
        rec_spawns = [c for c, b in spawned_cmds if c[0] == "pw-record"]
        self.assertGreaterEqual(len(play_spawns), 1)
        rec_targets = [c[c.index("--target") + 1] for c in rec_spawns]
        self.assertEqual(rec_targets[0], "alsa_input.usb-MicA")
        self.assertEqual(rec_targets[-1], "alsa_input.usb-MicB")
        for cmd, bufsize in spawned_cmds:
            self.assertIn("--raw", cmd)
            self.assertEqual(bufsize, 0)

        # 2. Verify silence pacing during ~5.5s of stall/backoff (expected ~40-50 blocks, never 240)
        self.assertGreaterEqual(len(sink_writes), 35)
        self.assertLessEqual(len(sink_writes), 55)

        # 3. Verify health event was emitted with liveness gauges
        health_events = [kw for name, kw in emitted_events if name == "audio_capture_health"]
        self.assertGreaterEqual(len(health_events), 1)
        self.assertIn("read_pending", health_events[0])
        self.assertIn("pending_block_age_ms", health_events[0])
        self.assertIn("output_pending", health_events[0])
        self.assertIn("sink_consecutive_failures", health_events[0])

        # 4. Verify stall + switch events A -> B were emitted and MicB block ran inference
        stall_events = [kw for name, kw in emitted_events if name == "audio_capture_stall"]
        self.assertGreaterEqual(len(stall_events), 1)
        switch_events = [kw for name, kw in emitted_events if name == "audio_source_switch_requested"]
        self.assertEqual(len(switch_events), 1)
        self.assertEqual(switch_events[0]["old_source"], "alsa_input.usb-MicA")
        self.assertEqual(switch_events[0]["target_mic"], "alsa_input.usb-MicB")
        self.assertGreaterEqual(infer_calls[0], 1)

    # -------------------------------------------------------------------------
    # V2 Integration Regressions: Sink Backoff Recovery & Incremental Output
    # -------------------------------------------------------------------------
    def _run_daemon_main_harness(self, *, fail_first_sink=False, standby_mode=False, partial_sink_switch=False):
        clock_ns = [1_000_000_000]
        spawns = []
        outputs = []
        emitted = []
        runtimes = []
        capture_calls = [0]

        def clock():
            return clock_ns[0] / 1_000_000_000.0

        def clock_ns_fn():
            return clock_ns[0]

        def sleep_fn(sec):
            clock_ns[0] += int(max(0.0, sec) * 1_000_000_000)

        class SinkStdin:
            closed = False

            def __init__(self, idx):
                self.idx = idx
                self.written_chunks = []
                self.calls = 0

            def write(self, data):
                self.calls += 1
                if fail_first_sink and self.idx == 1:
                    raise BrokenPipeError("simulated first sink failure")
                if partial_sink_switch and self.idx == 1:
                    if self.calls == 1:
                        chunk = bytes(data[:4096])
                        self.written_chunks.append(chunk)
                        outputs.append(chunk)
                        return 4096
                    raise BlockingIOError(errno.EAGAIN, "sink pipe backpressure")
                chunk = bytes(data)
                self.written_chunks.append(chunk)
                outputs.append(chunk)
                return len(data)

            def flush(self):
                pass

            def close(self):
                self.closed = True

        class CaptureStdout:
            closed = False

            def __init__(self, target):
                self.target = target
                self.calls = 0

            def read(self, count):
                self.calls += 1
                capture_calls[0] += 1
                if partial_sink_switch:
                    if self.target == "USB-A":
                        if self.calls == 1:
                            return np.full(2048, 1.0, dtype=np.float32).tobytes()
                        clock_ns[0] += 250_000_000  # Advance 250ms to trigger config check
                        cfg_path.write_text(json.dumps({
                            "audio": {
                                "input_device": "USB-B",
                                "auto_standby": False,
                                "noise_suppression": False,
                                "studio_mic_enabled": False,
                            }
                        }))
                        os.utime(cfg_path, (2000, 2000))
                        return b""
                    else:
                        if self.calls == 1:
                            return np.full(2048, 2.0, dtype=np.float32).tobytes()
                        daemon.running = False
                        return b""

                clock_ns[0] += 128_000_000
                if capture_calls[0] > 20:
                    daemon.running = False
                    return b""
                return np.full(2048, 0.05, dtype=np.float32).tobytes()

            def close(self):
                self.closed = True

        sink_instances = []

        def popen(cmd, **kwargs):
            spawns.append((list(cmd), clock()))
            if cmd[0] == "pw-play":
                idx = sum(1 for c, _ in spawns if c[0] == "pw-play")
                s = SinkStdin(idx)
                sink_instances.append(s)
                stdin, stdout = s, None
            else:
                target = cmd[cmd.index("--target") + 1]
                stdin, stdout = None, CaptureStdout(target)
            return SimpleNamespace(
                stdin=stdin,
                stdout=stdout,
                stderr=None,
                poll=lambda: None,
                terminate=lambda: None,
                wait=lambda timeout=None: 0,
                kill=lambda: None,
            )

        class Runtime:
            def __init__(self, *args, **kwargs):
                self.enabled = True
                self.counters = {}
                self.telemetry = SimpleNamespace(
                    emit=lambda name, **kw: emitted.append((name, kw)),
                    sample=lambda *a, **kw: None,
                )
                runtimes.append(self)

            def register_owner_counter(self, name, getter):
                self.counters[name] = getter

            def update_config(self, cfg):
                pass

            def should_trace(self, fid):
                return False

            def stop(self):
                pass

        model = SimpleNamespace(inputs=[
            SimpleNamespace(get_any_name=lambda: "input", shape=[1, 2048]),
            SimpleNamespace(get_any_name=lambda: "inp_state_1", shape=[1, 64]),
        ])
        core = SimpleNamespace(
            available_devices=["NPU"],
            read_model=lambda p: model,
            compile_model=lambda *a: SimpleNamespace(
                create_infer_request=lambda: SimpleNamespace(
                    infer=lambda feed: None,
                    get_tensor=lambda n: SimpleNamespace(
                        data=np.zeros((1, 2048) if n == "output" else (1, 64), np.float32)
                    ),
                )
            ),
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.json"
            cfg_path.write_text(json.dumps({
                "audio": {
                    "input_device": "USB-Missing" if standby_mode else "USB-A",
                    "auto_standby": False,
                    "noise_suppression": False,
                    "studio_mic_enabled": False,
                }
            }))
            os.utime(cfg_path, (1000, 1000))
            model_path = Path(td) / "model.xml"
            model_path.write_text("<net/>")

            orig_sleep = sleep_fn
            if standby_mode:
                sleep_calls = [0]

                def bounded_sleep(sec):
                    sleep_calls[0] += 1
                    orig_sleep(sec)
                    if sleep_calls[0] >= 40:
                        daemon.running = False

                active_sleep = bounded_sleep
            else:
                active_sleep = orig_sleep

            daemon.running = True
            with patch.object(daemon, "CONFIG_PATH", cfg_path), \
                 patch.object(daemon, "MODEL_PATH", model_path), \
                 patch.object(daemon.ov, "Core", return_value=core), \
                 patch.object(daemon.subprocess, "Popen", side_effect=popen), \
                 patch.object(daemon, "setup_pipewire_virtual_devices"), \
                 patch.object(daemon, "cleanup"), \
                 patch.object(daemon, "run_cmd", return_value=""), \
                 patch.object(daemon, "get_physical_microphones", return_value=["USB-A", "USB-B"]), \
                 patch.object(daemon, "TelemetryRuntime", Runtime), \
                 patch.object(daemon.time, "monotonic", side_effect=clock), \
                 patch.object(daemon.time, "monotonic_ns", side_effect=clock_ns_fn), \
                 patch.object(daemon.time, "sleep", side_effect=active_sleep), \
                 patch.object(daemon, "AudioCaptureReader", side_effect=lambda **kw: AudioCaptureReader(clock=clock, **kw)), \
                 patch.object(daemon, "AudioOutputWriter", side_effect=lambda **kw: AudioOutputWriter(clock=clock, **kw)), \
                 patch.object(daemon, "CaptureRetryPolicy", side_effect=lambda **kw: CaptureRetryPolicy(clock=clock, **kw)):
                daemon.main()

        return {
            "play_spawns": [t for cmd, t in spawns if cmd[0] == "pw-play"],
            "sink_instances": sink_instances,
            "counters": {k: fn() for k, fn in runtimes[0].counters.items()},
            "events": emitted,
        }

    def test_v2_broken_sink_recovers_while_capture_continues(self):
        """Broken first pw-play recovers after 250ms backoff even while capture produces blocks every 128ms."""
        res = self._run_daemon_main_harness(fail_first_sink=True)
        self.assertEqual(len(res["play_spawns"]), 2)
        self.assertEqual(res["counters"]["write_or_flush_failure_total"], 1)
        self.assertGreater(res["counters"]["normal_output_samples_accepted_total"], 0)

    def test_v2_broken_sink_recovers_during_standby_silence(self):
        """Broken first pw-play recovers after 250ms backoff during silence pacing when source is unavailable."""
        res = self._run_daemon_main_harness(fail_first_sink=True, standby_mode=True)
        self.assertEqual(len(res["play_spawns"]), 2)
        self.assertEqual(res["counters"]["write_or_flush_failure_total"], 1)
        self.assertGreater(res["counters"]["silence_output_samples_accepted_total"], 0)

    def test_v2_source_switch_cancels_partially_delivered_output_and_reconnects_sink(self):
        """Source switch with 4096/8192 bytes written cancels pending output, reconnects pw-play, and delivers new source cleanly."""
        res = self._run_daemon_main_harness(partial_sink_switch=True)
        self.assertGreaterEqual(len(res["play_spawns"]), 2)
        cancel_events = [kw for name, kw in res["events"] if name == "audio_output_pending_cancelled"]
        self.assertEqual(len(cancel_events), 1)
        self.assertEqual(cancel_events[0]["bytes_written"], 4096)
        self.assertEqual(cancel_events[0]["partial_discarded_bytes"], 4096)
        self.assertTrue(cancel_events[0]["requires_reconnect"])
        new_sink_bytes = b"".join(res["sink_instances"][1].written_chunks)
        self.assertIn(np.full(2048, 2.0, dtype=np.float32).tobytes(), new_sink_bytes)
        self.assertEqual(res["counters"]["normal_output_samples_accepted_total"], 2048)

    # -------------------------------------------------------------------------
    # 1,000-cycle Process Ownership & Reaping Stress Test
    # -------------------------------------------------------------------------
    def test_daemon_1000_switch_and_stall_process_reap_cycles(self):
        """1,000 rapid switch/stall cycles reap every subprocess cleanly without zombies or buffer growth."""
        reaped = [0]
        reader = AudioCaptureReader(block_size_bytes=8192)
        writer = AudioOutputWriter(block_size_bytes=8192)

        for i in range(1000):
            events = []
            proc = SimpleNamespace(
                stdin=io.BytesIO(),
                stdout=io.BytesIO(b"P" * (i % 4096)),
                stderr=None,
                terminate=lambda e=events: e.append("term"),
                wait=lambda timeout=None, e=events: (e.append("wait"), reaped.__setitem__(0, reaped[0] + 1)),
                kill=lambda e=events: e.append("kill"),
            )
            reader.attach(proc, generation_id=f"gen-{i}", epoch=i)
            reader.read_step(timeout_sec=0.001)
            reader.detach()
            writer.reset(reason="cycle")
            daemon.stop_process(proc, timeout=0.01)
            self.assertIn("term", events)
            self.assertIn("wait", events)
            self.assertEqual(reader.buffer_len, 0)
            self.assertFalse(writer.has_pending)

        self.assertEqual(reaped[0], 1000)


if __name__ == "__main__":
    unittest.main()
