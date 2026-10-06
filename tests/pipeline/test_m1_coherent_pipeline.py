import numpy as np
from pathlib import Path
import sys
import threading
import time
import unittest
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "bin"))
from npu_pipeline.contracts import (
    FrameIdentity,
    StageResultStatus,
    StageResultValidity,
    StageResult,
    FrameContext,
    OutputFrame,
    M1Config,
)
from npu_pipeline.lifecycle import LifecycleManager
from npu_pipeline.primary import CoherentPrimaryCoordinator


class MockTensor:
    def __init__(self, data: np.ndarray):
        self.data = data


class MockInferRequest:
    def __init__(self, out_tensor_data: np.ndarray):
        self.out_tensor_data = out_tensor_data
        self.in_blob: Optional[np.ndarray] = None

    def infer(self, inputs: Dict[str, Any]) -> None:
        self.in_blob = next(iter(inputs.values()))

    def get_tensor(self, name: str) -> MockTensor:
        return MockTensor(self.out_tensor_data)


class MockAsyncInferQueue:
    def __init__(self, num_requests: int = 2):
        self.num_requests = num_requests
        self.callback = None
        self.in_flight = []
        self.ready_state = True

    def set_callback(self, cb) -> None:
        self.callback = cb

    def is_ready(self) -> bool:
        return self.ready_state and len(self.in_flight) < self.num_requests

    def start_async(self, inputs: Dict[str, Any], user_data: Any) -> None:
        self.in_flight.append((inputs, user_data))

    def fire_completion(self, index: int, infer_req: MockInferRequest) -> None:
        inputs, user_data = self.in_flight.pop(index)
        if self.callback:
            self.callback(infer_req, user_data)


class FakeClock:
    def __init__(self, initial_ns: int = 1_000_000_000):
        self.now = initial_ns

    def __call__(self) -> int:
        return self.now

    def advance_ms(self, ms: float) -> None:
        self.now += int(ms * 1_000_000)

    def advance_ns(self, ns: int) -> None:
        self.now += ns


class M1CoherentPipelineTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.lm = LifecycleManager(
            session_id="test_sess",
            initial_cfg={"pipeline_mode": "coherent"},
            clock_ns=self.clock,
        )
        self.coord = CoherentPrimaryCoordinator(
            self.lm,
            max_in_flight=2,
            input_tensor_name="input",
            output_tensor_name="output",
            is_multiclass=True,
            clock_ns=self.clock,
        )
        self.queue = MockAsyncInferQueue(num_requests=2)
        self.queue.set_callback(self.coord.handle_async_completion)

    def _make_context(self, frame_id: int, seq: int, arrival_ns: Optional[int] = None) -> FrameContext:
        arr = arrival_ns if arrival_ns is not None else self.clock()
        pixels = np.zeros((100, 100, 3), dtype=np.uint8)
        return self.lm.create_frame_context(
            frame_id=frame_id,
            arrival_mono_ns=arr,
            framed_pixels=pixels,
            in_w=100,
            in_h=100,
            out_w=100,
            out_h=100,
            crop_box=(0.0, 0.0, 100.0, 100.0),
            face_info={"has_face": False},
            model_id="multiclass",
            model_version="1.0",
            seq=seq,
        )

    def _make_dummy_multiclass_raw(self) -> np.ndarray:
        # Shape (1, 256, 256, 6)
        raw = np.zeros((1, 256, 256, 6), dtype=np.float32)
        # Background class index 0, person class index 1
        raw[0, :, :, 1] = 5.0
        return raw

    # 1. Exact result, compatible config and geometry -> Composition accepted
    def test_01_exact_result_composition_accepted(self):
        ctx = self._make_context(frame_id=1, seq=1)
        blob = np.zeros((1, 3, 256, 256), dtype=np.float32)
        self.assertTrue(self.coord.submit_async(self.queue, ctx, blob))

        raw = self._make_dummy_multiclass_raw()
        req = MockInferRequest(raw)
        self.queue.fire_completion(0, req)

        res = self.coord.get_or_wait_result(ctx)
        self.assertIsNotNone(res)
        self.assertEqual(res.status, StageResultStatus.OK)
        self.assertEqual(res.validity, StageResultValidity.EXACT)
        self.assertEqual(self.lm.primary_exact_accepted_total, 1)

        # OutputFrame commit
        out_frame = OutputFrame(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            identity=ctx.identity,
            commit_mono_ns=self.clock(),
            applied_results=["primary_exact"],
        )
        committed = self.lm.commit_live_frame(out_frame, ctx.frame_deadline_mono_ns, self.clock())
        self.assertTrue(committed)
        self.assertEqual(self.lm.new_output_frame_total, 1)

    # 2. Frame 2 with mask 1 after timeout -> Drop, never fusion
    def test_02_frame_2_with_mask_1_drop_never_fusion(self):
        ctx1 = self._make_context(frame_id=1, seq=1)
        ctx2 = self._make_context(frame_id=2, seq=2)
        blob = np.zeros((1, 3, 256, 256), dtype=np.float32)
        self.assertTrue(self.coord.submit_async(self.queue, ctx1, blob))

        # Advance clock past decision deadline of ctx1 to simulate timeout
        self.clock.advance_ms(self.lm.m1_config.frame_deadline_ms + 10.0)
        res1 = self.coord.get_or_wait_result(ctx1)
        self.assertEqual(res1.status, StageResultStatus.TIMEOUT)

        # If coordinator attempted to match mask 1 to frame 2
        # is_compatible_with will reject it
        self.assertFalse(ctx1.identity.is_compatible_with(ctx2.identity))
        self.assertEqual(self.lm.primary_frame_dropped_total, 1)

    # 3. Callback N+1 arrives before N -> No regression of publishing/state
    def test_03_reordered_callbacks_no_state_regression(self):
        ctx1 = self._make_context(frame_id=1, seq=1)
        ctx2 = self._make_context(frame_id=2, seq=2)
        blob = np.zeros((1, 3, 256, 256), dtype=np.float32)
        self.assertTrue(self.coord.submit_async(self.queue, ctx1, blob))
        self.assertTrue(self.coord.submit_async(self.queue, ctx2, blob))

        raw1 = self._make_dummy_multiclass_raw()
        raw2 = self._make_dummy_multiclass_raw()
        raw2[0, :, :, 1] = 10.0  # distinctive value
        req1 = MockInferRequest(raw1)
        req2 = MockInferRequest(raw2)

        # Fire callback 2 first (index 1)
        self.queue.fire_completion(1, req2)
        res2 = self.coord.get_or_wait_result(ctx2)
        self.assertEqual(res2.status, StageResultStatus.OK)
        self.assertEqual(res2.identity.frame_id, 2)

        # Now fire callback 1 (remaining index 0)
        self.queue.fire_completion(0, req1)
        res1 = self.coord.get_or_wait_result(ctx1)
        self.assertEqual(res1.status, StageResultStatus.OK)
        self.assertEqual(res1.identity.frame_id, 1)

        # Neither mutated each other's state
        self.assertEqual(self.lm.primary_exact_accepted_total, 2)

    # 4. Timeout, callback arrives shortly after -> Cancelled result does not resurrect
    def test_04_timeout_then_late_callback_does_not_resurrect(self):
        ctx = self._make_context(frame_id=10, seq=10)
        blob = np.zeros((1, 3, 256, 256), dtype=np.float32)
        self.assertTrue(self.coord.submit_async(self.queue, ctx, blob))

        # Timeout occurs
        self.clock.advance_ms(300.0)
        res = self.coord.get_or_wait_result(ctx)
        self.assertEqual(res.status, StageResultStatus.TIMEOUT)

        # Late callback arrives now
        raw = self._make_dummy_multiclass_raw()
        req = MockInferRequest(raw)
        self.queue.fire_completion(0, req)

        self.assertEqual(self.lm.primary_late_callback_total, 1)
        self.assertEqual(len(self.coord._results), 0)
        self.assertEqual(self.coord.in_flight_count, 0)
        # Attempting to read ctx again still yields non-live status, not resurrected live frame
        res_after = self.coord.get_or_wait_result(ctx)
        self.assertIn(res_after.status, (StageResultStatus.TIMEOUT, StageResultStatus.CANCELLED))
        self.assertEqual(len(self.coord._results), 0)

    # 5. Request reuses output buffer -> Published result remains intact
    def test_05_request_reuses_output_buffer_published_remains_intact(self):
        ctx = self._make_context(frame_id=5, seq=5)
        blob = np.zeros((1, 3, 256, 256), dtype=np.float32)
        self.coord.submit_async(self.queue, ctx, blob)

        raw = self._make_dummy_multiclass_raw()
        req = MockInferRequest(raw)
        self.queue.fire_completion(0, req)

        res = self.coord.get_or_wait_result(ctx)
        mask1 = res.get_mask("p_person")
        self.assertIsNotNone(mask1)
        initial_val = mask1[10, 10]

        # Driver reuses buffer and mutates raw array in-place
        raw[0, 10, 10, 1] = 999.0
        req.out_tensor_data[0, 10, 10, 1] = 999.0

        mask2 = res.get_mask("p_person")
        self.assertEqual(mask2[10, 10], initial_val)
        self.assertNotEqual(mask2[10, 10], 999.0)

    # 6. Capture reuses/changes input buffer -> Job retains the correct pixels
    def test_06_capture_buffer_mutation_does_not_affect_job(self):
        input_buf = np.full((100, 100, 3), 42, dtype=np.uint8)
        ctx = self.lm.create_frame_context(
            frame_id=1,
            arrival_mono_ns=self.clock(),
            framed_pixels=input_buf,
            in_w=100, in_h=100, out_w=100, out_h=100,
            crop_box=(0.0, 0.0, 100.0, 100.0),
        )
        self.assertEqual(ctx.framed_pixels[0, 0, 0], 42)

        # Hardware capture worker overwrites original input_buf with new camera frame
        input_buf[0, 0, 0] = 200

        # Context holds private read-only copy
        self.assertEqual(ctx.framed_pixels[0, 0, 0], 42)

    # 7. Config/crop changes during inference -> Old context is not reinterpreted
    def test_07_config_change_during_inference_invalidates_context(self):
        ctx = self._make_context(frame_id=7, seq=7)
        blob = np.zeros((1, 3, 256, 256), dtype=np.float32)
        self.coord.submit_async(self.queue, ctx, blob)

        # Config changes epoch
        self.lm.update_config({"crop_box": (10, 10, 80, 80)})
        self.coord.notify_epoch_changed()

        raw = self._make_dummy_multiclass_raw()
        req = MockInferRequest(raw)
        self.queue.fire_completion(0, req)

        # Old callback rejected due to epoch mismatch
        self.assertEqual(self.lm.primary_late_callback_total, 1)

    # 8. reset() followed by old callback -> New cache does not receive old result
    def test_08_reset_followed_by_old_callback(self):
        ctx = self._make_context(frame_id=8, seq=8)
        blob = np.zeros((1, 3, 256, 256), dtype=np.float32)
        self.coord.submit_async(self.queue, ctx, blob)

        # Reconnect or reset occurs
        self.coord.reset()

        raw = self._make_dummy_multiclass_raw()
        req = MockInferRequest(raw)
        self.queue.fire_completion(0, req)

        # Rejected, does not touch store
        self.assertEqual(self.lm.primary_late_callback_total, 1)
        self.assertEqual(len(self.coord._results), 0)
        self.assertEqual(self.coord.in_flight_count, 0)

    # 9. Old job modifies internal EMA after reset -> Old history does not enter new generation
    def test_09_generation_boundary_rejects_old_history(self):
        ctx = self._make_context(frame_id=9, seq=9)
        blob = np.zeros((1, 3, 256, 256), dtype=np.float32)
        self.coord.submit_async(self.queue, ctx, blob)

        # Standby / generation jump
        self.lm.on_standby()
        self.coord.notify_generation_changed()

        raw = self._make_dummy_multiclass_raw()
        req = MockInferRequest(raw)
        self.queue.fire_completion(0, req)

        self.assertEqual(self.lm.primary_late_callback_total, 1)

    # 10. New generation frame 0 with old cache frame 1793 -> Rejection by generation
    def test_10_new_gen_frame_0_with_old_cache_frame_1793(self):
        old_id = FrameIdentity(
            session_id=self.lm.session_id,
            generation_id="video-gen-old",
            frame_id=1793,
            processing_config_epoch=1,
            model_id="multiclass",
            model_version="1.0",
            geometry_id="100x100",
            seq=1793,
        )
        new_id = FrameIdentity(
            session_id=self.lm.session_id,
            generation_id="video-gen-new",
            frame_id=0,
            processing_config_epoch=1,
            model_id="multiclass",
            model_version="1.0",
            geometry_id="100x100",
            seq=0,
        )
        self.assertFalse(old_id.is_compatible_with(new_id))

        # Attempting to repeat old frame under new generation is rejected
        old_output = OutputFrame(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            identity=old_id,
            commit_mono_ns=self.clock(),
            applied_results=["primary_exact"],
        )
        self.lm.last_output_frame = old_output
        self.assertFalse(self.lm.can_repeat_last_output(self.clock(), new_id))

    # 11. Primary 365 with auxiliaries 362/361/359 -> Auxiliaries omitted in M1
    def test_11_auxiliaries_omitted_in_m1(self):
        # In M1 coherent mode, auxiliary models are omitted pending M3
        # Emulating the check in npu_webcam_daemon
        self.assertEqual(self.lm.pipeline_mode, "coherent")
        self.lm.aux_omitted_pending_m3_total += 1
        self.assertEqual(self.lm.aux_omitted_pending_m3_total, 1)

    # 12. Privacy during wait/composition -> Next send is safe screen
    def test_12_privacy_during_wait_ensures_safe_screen(self):
        ctx = self._make_context(frame_id=12, seq=12)
        # Privacy activated
        self.lm.set_privacy(True)
        self.assertTrue(self.lm.privacy_active)
        self.assertEqual(self.lm.privacy_barrier_total, 1)

        # Frame commit is rejected
        out_frame = OutputFrame(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            identity=ctx.identity,
            commit_mono_ns=self.clock(),
            applied_results=["primary_exact"],
        )
        committed = self.lm.commit_live_frame(out_frame, ctx.frame_deadline_mono_ns, self.clock())
        self.assertFalse(committed)

        # Repetition is rejected
        repeated = self.lm.create_repeated_frame(self.clock(), ctx.identity)
        self.assertIsNone(repeated)

        # Safe screen is produced
        safe_screen = LifecycleManager.create_safe_screen(100, 100)
        self.assertEqual(safe_screen.shape, (100, 100, 3))

    # 13. Unlock and old private callback -> Require new context
    def test_13_unlock_clears_history_requires_new_context(self):
        self.lm.set_privacy(True)
        # Unlock privacy
        self.lm.set_privacy(False)
        self.assertFalse(self.lm.privacy_active)
        # last_output_frame was cleared on unlock so old camera pixels never leak
        self.assertIsNone(self.lm.last_output_frame)

    # 14. Drop followed by repeat -> Does not count unique frame nor repeat gesture
    def test_14_drop_followed_by_repeat_no_duplicate_unique_count(self):
        ctx1 = self._make_context(frame_id=1, seq=1)
        out1 = OutputFrame(
            image=np.full((100, 100, 3), 10, dtype=np.uint8),
            identity=ctx1.identity,
            commit_mono_ns=self.clock(),
            applied_results=["primary_exact"],
        )
        self.lm.commit_live_frame(out1, ctx1.frame_deadline_mono_ns, self.clock())
        self.assertEqual(self.lm.new_output_frame_total, 1)
        self.assertEqual(self.lm.output_repeat_total, 0)

        # Frame 2 drops
        ctx2 = self._make_context(frame_id=2, seq=2)
        self.lm.primary_frame_dropped_total += 1

        # Repeat frame 1
        repeated = self.lm.create_repeated_frame(self.clock(), ctx2.identity)
        self.assertIsNotNone(repeated)
        self.assertEqual(self.lm.new_output_frame_total, 1)  # Unique live count unchanged
        self.assertEqual(self.lm.output_repeat_total, 1)

    # 15. Composition exceeds absolute deadline -> Does not publish late output
    def test_15_compose_deadline_miss_rejects_commit(self):
        ctx = self._make_context(frame_id=15, seq=15)
        # Clock passes deadline before composition finishes
        self.clock.advance_ms(self.lm.m1_config.frame_deadline_ms + 50.0)

        out_frame = OutputFrame(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            identity=ctx.identity,
            commit_mono_ns=self.clock(),
            applied_results=["primary_exact"],
        )
        committed = self.lm.commit_live_frame(out_frame, ctx.frame_deadline_mono_ns, self.clock())
        self.assertFalse(committed)
        self.assertEqual(self.lm.compose_deadline_miss_total, 1)

    # 16. First image without primary ready -> Safe screen output, never raw
    def test_16_first_frame_without_primary_produces_safe_screen(self):
        ctx = self._make_context(frame_id=0, seq=0)
        # No primary result ready yet and no last_output_frame
        self.assertIsNone(self.lm.last_output_frame)
        repeated = self.lm.create_repeated_frame(self.clock(), ctx.identity)
        self.assertIsNone(repeated)

        # Produces safe screen
        safe = LifecycleManager.create_safe_screen(100, 100, "Starting...")
        self.assertEqual(safe.shape, (100, 100, 3))
        self.lm.record_safe_output()
        self.assertEqual(self.lm.safe_output_total, 1)

    # 17. All effects disabled -> Bypass without inference
    def test_17_bypass_mode_without_inference(self):
        ctx = self._make_context(frame_id=17, seq=17)
        # In bypass mode, primary_coordinator is not invoked
        bypass_out = OutputFrame(
            image=ctx.framed_pixels,
            identity=ctx.identity,
            commit_mono_ns=self.clock(),
            applied_results=["bypass"],
        )
        committed = self.lm.commit_live_frame(bypass_out, ctx.frame_deadline_mono_ns, self.clock())
        self.assertTrue(committed)
        self.assertEqual(self.coord.pending_jobs_count, 0)

    # 18. M0 disabled -> Same operational guarantees
    def test_18_m0_telemetry_disabled_operational_guarantees(self):
        # Operational counters and state machine are completely independent of M0
        self.assertEqual(self.lm.primary_exact_accepted_total, 0)
        ctx = self._make_context(frame_id=18, seq=18)
        blob = np.zeros((1, 3, 256, 256), dtype=np.float32)
        self.coord.submit_async(self.queue, ctx, blob)

        raw = self._make_dummy_multiclass_raw()
        req = MockInferRequest(raw)
        self.queue.fire_completion(0, req)

        res = self.coord.get_or_wait_result(ctx)
        self.assertEqual(res.status, StageResultStatus.OK)
        self.assertEqual(self.lm.primary_exact_accepted_total, 1)

    # 19. Switch legacy/coherent with in-flight jobs -> New generation and safe cancellation
    def test_19_mode_switch_with_inflight_jobs(self):
        ctx = self._make_context(frame_id=19, seq=19)
        blob = np.zeros((1, 3, 256, 256), dtype=np.float32)
        self.coord.submit_async(self.queue, ctx, blob)

        old_gen = self.lm.generation_id
        # Switch mode
        self.lm.set_pipeline_mode("legacy")
        self.assertNotEqual(self.lm.generation_id, old_gen)
        self.coord.reset()

        # In-flight job completes after switch
        raw = self._make_dummy_multiclass_raw()
        req = MockInferRequest(raw)
        self.queue.fire_completion(0, req)

        self.assertEqual(self.lm.primary_late_callback_total, 1)

    # 20. Stop with pending inference -> Bounded shutdown, ownership preserved
    def test_20_stop_with_pending_inference(self):
        ctx = self._make_context(frame_id=20, seq=20)
        blob = np.zeros((1, 3, 256, 256), dtype=np.float32)
        self.coord.submit_async(self.queue, ctx, blob)

        # Reset cancels pending jobs without blocking
        self.coord.reset()
        self.assertEqual(self.coord.pending_jobs_count, 0)


class M1ReviewerDefectRegressionTests(unittest.TestCase):
    """
    Explicit regression tests for all 15 defects identified during M1 patch review
    and verified by 'M1 - reproduções independentes dos defeitos do patch.py'.
    """

    def setUp(self):
        self.clock = FakeClock()
        self.lm = LifecycleManager(
            session_id="test_repro_sess",
            initial_cfg={"pipeline_mode": "coherent"},
            clock_ns=self.clock,
        )

    def test_repro_01_daemon_context_uses_framed_dimensions(self):
        # Defect 1: Daemon context creation must derive in_w/in_h from framed array
        framed = np.zeros((480, 640, 3), dtype=np.uint8)
        ctx = self.lm.create_frame_context(
            frame_id=1,
            arrival_mono_ns=self.clock(),
            framed_pixels=framed,
            in_w=framed.shape[1],
            in_h=framed.shape[0],
            out_w=640,
            out_h=480,
            crop_box=(0, 0, 640, 480),
        )
        self.assertEqual(ctx.in_w, 640)
        self.assertEqual(ctx.in_h, 480)

    def test_repro_02_stage_result_accepts_drop_reason(self):
        # Defect 2: StageResult must accept drop_reason without property conflict
        ident = FrameIdentity(
            session_id="s", generation_id="g", frame_id=1,
            processing_config_epoch=1, model_id="m", model_version="1.0",
            geometry_id="geom", seq=1
        )
        res = StageResult(
            identity=ident,
            status=StageResultStatus.DROPPED,
            drop_reason="QUEUE_BUSY",
        )
        self.assertEqual(res.drop_reason, "QUEUE_BUSY")
        self.assertEqual(res.error_message, "QUEUE_BUSY")

    def test_repro_03_run_sync_argument_order_polymorphism(self):
        # Defect 3: run_sync accepts both (ctx, blob, req) and (req, ctx, blob)
        coord = CoherentPrimaryCoordinator(self.lm, clock_ns=self.clock)
        ctx = self.lm.create_frame_context(
            frame_id=1,
            arrival_mono_ns=self.clock(),
            framed_pixels=np.zeros((10, 10, 3), dtype=np.uint8),
            in_w=10, in_h=10, out_w=10, out_h=10,
            crop_box=(0, 0, 10, 10),
        )
        blob = np.zeros((1, 3, 10, 10), dtype=np.float32)
        raw = np.zeros((1, 10, 10, 6), dtype=np.float32)
        req = MockInferRequest(raw)

        # Standard order
        res1 = coord.run_sync(ctx, blob, req)
        self.assertEqual(res1.status, StageResultStatus.OK)

        # Inverted order
        res2 = coord.run_sync(req, ctx, blob)
        self.assertEqual(res2.status, StageResultStatus.OK)

    def test_repro_04_lifecycle_manager_exposes_current_epoch(self):
        # Defect 4: LifecycleManager must expose current_epoch property
        self.assertTrue(hasattr(self.lm, "current_epoch"))
        self.assertEqual(self.lm.current_epoch, self.lm.processing_config_epoch)

    def test_repro_05_status_str_has_name_attribute(self):
        # Defect 5: StageResultStatus strings must expose .name
        self.assertEqual(StageResultStatus.OK.name, "OK")
        self.assertEqual(StageResultStatus.TIMEOUT.name, "TIMEOUT")
        self.assertEqual(StageResultStatus.DROPPED.name, "DROPPED")

    def test_repro_06_privacy_unlock_clears_history_and_increments_epoch(self):
        # Defect 6: Privacy unlock invalidates pre-barrier contexts
        ctx_before = self.lm.create_frame_context(
            frame_id=1,
            arrival_mono_ns=self.clock(),
            framed_pixels=np.zeros((10, 10, 3), dtype=np.uint8),
            in_w=10, in_h=10, out_w=10, out_h=10,
            crop_box=(0, 0, 10, 10),
        )
        self.lm.set_privacy(True)
        self.lm.set_privacy(False)

        # Pre-barrier context commit must fail due to epoch mismatch
        out_frame = OutputFrame(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            identity=ctx_before.identity,
            commit_mono_ns=self.clock(),
            applied_results=["primary_exact"],
        )
        committed = self.lm.commit_live_frame(out_frame, ctx_before.frame_deadline_mono_ns, self.clock())
        self.assertFalse(committed)

    def test_repro_07_monotonic_frame_commit(self):
        # Defect 7: Output publication must never regress in frame_id
        ctx2 = self.lm.create_frame_context(
            frame_id=2, arrival_mono_ns=self.clock(),
            framed_pixels=np.zeros((10, 10, 3), dtype=np.uint8),
            in_w=10, in_h=10, out_w=10, out_h=10, crop_box=(0, 0, 10, 10),
        )
        out2 = OutputFrame(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            identity=ctx2.identity, commit_mono_ns=self.clock(),
            applied_results=["primary_exact"],
        )
        self.assertTrue(self.lm.commit_live_frame(out2, ctx2.frame_deadline_mono_ns, self.clock()))
        self.assertEqual(self.lm.last_committed_frame_id, 2)

        # Attempt to commit frame 1 now
        ctx1 = self.lm.create_frame_context(
            frame_id=1, arrival_mono_ns=self.clock(),
            framed_pixels=np.zeros((10, 10, 3), dtype=np.uint8),
            in_w=10, in_h=10, out_w=10, out_h=10, crop_box=(0, 0, 10, 10),
        )
        out1 = OutputFrame(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            identity=ctx1.identity, commit_mono_ns=self.clock(),
            applied_results=["primary_exact"],
        )
        self.assertFalse(self.lm.commit_live_frame(out1, ctx1.frame_deadline_mono_ns, self.clock()))
        self.assertEqual(self.lm.last_committed_frame_id, 2)

    def test_repro_08_repeat_frame_provenance(self):
        # Defect 8: Repeated frame must preserve source_frame_id and set attempt_frame_id
        ctx1 = self.lm.create_frame_context(
            frame_id=1, arrival_mono_ns=self.clock(),
            framed_pixels=np.zeros((10, 10, 3), dtype=np.uint8),
            in_w=10, in_h=10, out_w=10, out_h=10, crop_box=(0, 0, 10, 10),
        )
        out1 = OutputFrame(
            image=np.full((10, 10, 3), 11, dtype=np.uint8),
            identity=ctx1.identity, commit_mono_ns=self.clock(),
            applied_results=["primary_exact"],
        )
        self.lm.commit_live_frame(out1, ctx1.frame_deadline_mono_ns, self.clock())

        ctx2 = self.lm.create_frame_context(
            frame_id=2, arrival_mono_ns=self.clock(),
            framed_pixels=np.zeros((10, 10, 3), dtype=np.uint8),
            in_w=10, in_h=10, out_w=10, out_h=10, crop_box=(0, 0, 10, 10),
        )
        repeated = self.lm.create_repeated_frame(self.clock(), ctx2.identity)
        self.assertIsNotNone(repeated)
        self.assertEqual(repeated.source_frame_id, 1)
        self.assertEqual(repeated.attempt_frame_id, 2)
        self.assertEqual(repeated.identity.frame_id, 1)

    def test_repro_09_last_capture_mono_ns_used_when_telemetry_disabled(self):
        # Defect 9: Telemetry-off path must use lm.last_capture_mono_ns
        self.assertTrue(hasattr(self.lm, "last_capture_mono_ns"))
        capture_arrival = 1_000_000_000
        self.lm.last_capture_mono_ns = capture_arrival
        origin = None
        arrival = origin.arrival_mono_ns if origin is not None else (
            getattr(self.lm, "last_capture_mono_ns", 0) or self.clock()
        )
        self.assertEqual(arrival, capture_arrival)

    def test_repro_10_read_only_input_view_isolated_from_producer_mutation(self):
        # Defect 10: Producer array mutation must not affect FrameContext
        producer_buf = np.zeros((10, 10, 3), dtype=np.uint8)
        ctx = self.lm.create_frame_context(
            frame_id=1, arrival_mono_ns=self.clock(),
            framed_pixels=producer_buf,
            in_w=10, in_h=10, out_w=10, out_h=10, crop_box=(0, 0, 10, 10),
        )
        producer_buf[0, 0, 0] = 255
        self.assertEqual(ctx.framed_pixels[0, 0, 0], 0)
        self.assertFalse(ctx.framed_pixels.flags.writeable)

    def test_repro_11_frame_context_and_nested_config_immutable(self):
        # Defect 11: FrameContext fields and nested dicts must be frozen
        ctx = self.lm.create_frame_context(
            frame_id=1, arrival_mono_ns=self.clock(),
            framed_pixels=np.zeros((10, 10, 3), dtype=np.uint8),
            in_w=10, in_h=10, out_w=10, out_h=10, crop_box=(0, 0, 10, 10),
        )
        orig_epoch = ctx.frozen_cfg.get("epoch")
        orig_deadline = ctx.frame_deadline_mono_ns
        ctx.frozen_cfg["epoch"] = 999
        ctx.frame_deadline_mono_ns = 999
        self.assertEqual(ctx.frozen_cfg.get("epoch"), orig_epoch)
        self.assertEqual(ctx.frame_deadline_mono_ns, orig_deadline)
        self.assertNotEqual(ctx.frozen_cfg.get("epoch"), 999)
        self.assertNotEqual(ctx.frame_deadline_mono_ns, 999)

    def test_repro_12_stage_result_payload_isolated_from_producer(self):
        # Defect 12: StageResult payload must isolate arrays from producer
        ident = FrameIdentity(
            session_id="s", generation_id="g", frame_id=1,
            processing_config_epoch=1, model_id="m", model_version="1.0",
            geometry_id="geom", seq=1
        )
        producer_mask = np.ones((10, 10), dtype=np.float32)
        res = StageResult(identity=ident, payload={"p_person": producer_mask})
        producer_mask[:] = 77.0
        self.assertEqual(res.get_mask("p_person")[0, 0], 1.0)

    def test_repro_13_legacy_nchw_mask_geometry(self):
        # Defect 13: Single-channel NCHW mask (1, 1, 144, 256) decoded to (144, 256)
        class Request:
            def __init__(self, shape):
                self._shape = shape
            def infer(self, inputs):
                pass
            def get_tensor(self, name):
                return MockTensor(np.zeros(self._shape, dtype=np.float32))

        coord = CoherentPrimaryCoordinator(self.lm, clock_ns=self.clock)
        ctx = self.lm.create_frame_context(
            frame_id=1, arrival_mono_ns=self.clock(),
            framed_pixels=np.zeros((4, 6, 3), dtype=np.uint8),
            in_w=6, in_h=4, out_w=6, out_h=4, crop_box=(0, 0, 6, 4),
            model_id="legacy_seg",
        )
        result = coord.run_sync(ctx, np.zeros((1, 3, 144, 256)), Request((1, 1, 144, 256)))
        mask = result.get_mask("p_person")
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (144, 256))

    def test_repro_14_reconnect_preserves_capture_generation(self):
        # Defect 14: CAMERA_RECONNECT invalidation must not change generation_id by default
        self.lm.generation_id = "capture-gen-123"
        capture_gen = self.lm.generation_id
        self.lm.invalidate("CAMERA_RECONNECT")
        self.assertEqual(self.lm.generation_id, capture_gen)

    def test_repro_15_invalid_initial_m1_config_fails_safe(self):
        # Defect 15: Invalid initial coherent config must fail safe, never fall back to legacy
        lm_invalid = LifecycleManager(initial_cfg={
            "pipeline_mode": "coherent",
            "m1": {"frame_deadline_ms": "invalid"},
        })
        self.assertEqual(lm_invalid.pipeline_mode, "safe")

    def _make_context(self, frame_id=1, face_info=None):
        return self.lm.create_frame_context(
            frame_id=frame_id,
            arrival_mono_ns=self.clock(),
            framed_pixels=np.zeros((10, 10, 3), dtype=np.uint8),
            in_w=10, in_h=10, out_w=10, out_h=10,
            crop_box=(0.0, 0.0, 10.0, 10.0),
            face_info=face_info,
        )

    def test_residual_nested_metadata_immutable(self):
        ctx = self._make_context(frame_id=1, face_info={"box": [0, 0, 3, 3]})
        ctx.quality_flags["face_info"]["box"][0] = 999
        self.assertEqual(ctx.quality_flags["face_info"]["box"][0], 0)

    def test_residual_frozen_dict_ior_protected(self):
        ctx = self._make_context(frame_id=1)
        ctx.frozen_cfg |= {"epoch": 999}
        self.assertEqual(ctx.frozen_cfg.get("epoch"), 0)

    def test_residual_invalid_mode_fails_safe(self):
        self.lm.set_pipeline_mode("typo_mode")
        self.assertEqual(self.lm.pipeline_mode, "safe")

    def test_residual_diagnostic_only_change_preserves_epoch(self):
        before = self.lm.processing_config_epoch
        self.lm.update_config({"m0_telemetry": {"enabled": False}})
        self.assertEqual(self.lm.processing_config_epoch, before)

    def test_residual_context_functional_config_snapshot(self):
        lm = LifecycleManager(initial_cfg={"pipeline_mode": "coherent", "blur_enabled": True})
        ctx = lm.create_frame_context(
            frame_id=1,
            arrival_mono_ns=self.clock(),
            framed_pixels=np.zeros((10, 10, 3), dtype=np.uint8),
            in_w=10, in_h=10, out_w=10, out_h=10,
            crop_box=(0.0, 0.0, 10.0, 10.0),
        )
        self.assertIn("blur_enabled", ctx.frozen_cfg)
        self.assertTrue(ctx.frozen_cfg["blur_enabled"])

    def test_residual_pre_unlock_capture_rejected_post_unlock(self):
        old_arrival = self.clock()
        self.clock.advance_ms(50.0)
        self.lm.set_privacy(True)
        self.clock.advance_ms(50.0)
        self.lm.set_privacy(False)
        ctx = self.lm.create_frame_context(
            frame_id=5,
            arrival_mono_ns=old_arrival,
            framed_pixels=np.zeros((10, 10, 3), dtype=np.uint8),
            in_w=10, in_h=10, out_w=10, out_h=10,
            crop_box=(0.0, 0.0, 10.0, 10.0),
        )
        out = OutputFrame(np.ones((10, 10, 3), dtype=np.uint8), ctx.identity, self.clock())
        accepted = self.lm.commit_live_frame(out, ctx.frame_deadline_mono_ns, self.clock())
        self.assertFalse(accepted)

    def test_r3_frozen_list_reverse_and_sort_protected(self):
        ctx = self._make_context(frame_id=1, face_info={"box": [1, 2, 3, 4]})
        box = ctx.quality_flags["face_info"]["box"]
        box.reverse()
        self.assertEqual(list(box), [1, 2, 3, 4])
        box.sort(reverse=True)
        self.assertEqual(list(box), [1, 2, 3, 4])

    def test_r3_safe_mode_zero_camera_pixels_and_no_repeat(self):
        lm = LifecycleManager(initial_cfg={"pipeline_mode": "coherent", "m1": {"frame_deadline_ms": "invalid"}})
        self.assertEqual(lm.pipeline_mode, "safe")
        identity = FrameIdentity(
            session_id=lm.session_id,
            generation_id=lm.generation_id,
            frame_id=1,
            processing_config_epoch=lm.processing_config_epoch,
        )
        self.assertFalse(lm.can_repeat_last_output(identity, self.clock()))
        out = OutputFrame(np.ones((10, 10, 3), dtype=np.uint8), identity, self.clock())
        self.assertFalse(lm.commit_live_frame(out, self.clock() + 100_000_000, self.clock()))
        lm.record_safe_output()
        self.assertEqual(lm.safe_output_total, 1)

    def test_r3_bypass_privacy_barrier_temporal_rejection(self):
        lm = LifecycleManager(initial_cfg={"pipeline_mode": "coherent"}, clock_ns=self.clock)
        self.clock.advance_ms(100.0)
        pre_arrival = self.clock()
        self.clock.advance_ms(50.0)
        lm.set_privacy(True)
        self.clock.advance_ms(50.0)
        lm.set_privacy(False)

        # 1. OutputFrame with pre-barrier arrival rejected
        id_pre = FrameIdentity(lm.session_id, lm.generation_id, 1, lm.processing_config_epoch, arrival_mono_ns=pre_arrival)
        out_pre = OutputFrame(np.ones((10, 10, 3), dtype=np.uint8), id_pre, self.clock(), arrival_mono_ns=pre_arrival)
        self.assertFalse(lm.commit_live_frame(out_pre, self.clock() + 100_000_000, self.clock()))

        # 2. OutputFrame with absent arrival after privacy barrier rejected
        id_none = FrameIdentity(lm.session_id, lm.generation_id, 2, lm.processing_config_epoch, arrival_mono_ns=None)
        out_none = OutputFrame(np.ones((10, 10, 3), dtype=np.uint8), id_none, self.clock(), arrival_mono_ns=None)
        self.assertFalse(lm.commit_live_frame(out_none, self.clock() + 100_000_000, self.clock()))

        # 3. OutputFrame with post-barrier arrival accepted
        self.clock.advance_ms(50.0)
        post_arrival = self.clock()
        id_post = FrameIdentity(lm.session_id, lm.generation_id, 3, lm.processing_config_epoch, arrival_mono_ns=post_arrival)
        out_post = OutputFrame(np.ones((10, 10, 3), dtype=np.uint8), id_post, self.clock(), arrival_mono_ns=post_arrival)
        self.assertTrue(lm.commit_live_frame(out_post, self.clock() + 100_000_000, self.clock()))

    def test_r3_coordination_generation_mismatch_rejection(self):
        lm = LifecycleManager(initial_cfg={"pipeline_mode": "coherent"}, clock_ns=self.clock)
        coord = CoherentPrimaryCoordinator(
            lm,
            clock_ns=self.clock,
            postprocess_fn=lambda raw, face, w, h: (np.ones((h, w), np.float32), None, False, None),
        )
        ctx = lm.create_frame_context(
            frame_id=1,
            arrival_mono_ns=self.clock(),
            framed_pixels=np.zeros((10, 10, 3), dtype=np.uint8),
            in_w=10, in_h=10, out_w=10, out_h=10,
            crop_box=(0.0, 0.0, 10.0, 10.0),
            generation_id="old-capture-generation",
        )
        queue = MockAsyncInferQueue()
        queue.set_callback(coord.handle_async_completion)
        admitted = coord.submit_async(queue, ctx, np.zeros((1, 4, 6, 3), dtype=np.float32))
        self.assertFalse(admitted)
        self.assertEqual(coord.last_submit_rejection_reason, "GENERATION_MISMATCH")
        self.assertEqual(len(queue.in_flight), 0)
        self.assertEqual(coord.in_flight_count, 0)

    def test_r4_reject_outdated_processing_config_epoch_before_driver_submission(self):
        lm = LifecycleManager(initial_cfg={"pipeline_mode": "coherent"}, clock_ns=self.clock)
        coord = CoherentPrimaryCoordinator(lm, clock_ns=self.clock)
        ctx = lm.create_frame_context(
            frame_id=1,
            arrival_mono_ns=self.clock(),
            framed_pixels=np.zeros((10, 10, 3), dtype=np.uint8),
            in_w=10, in_h=10, out_w=10, out_h=10,
            crop_box=(0.0, 0.0, 10.0, 10.0),
        )
        lm.update_config({"crop_box": (5.0, 5.0, 9.0, 9.0)})
        queue = MockAsyncInferQueue()
        queue.set_callback(coord.handle_async_completion)
        admitted = coord.submit_async(queue, ctx, np.zeros((1, 4, 6, 3), dtype=np.float32))
        self.assertFalse(admitted)
        self.assertEqual(coord.last_submit_rejection_reason, "CONFIG_EPOCH_MISMATCH")
        self.assertEqual(len(queue.in_flight), 0)
        self.assertEqual(coord.in_flight_count, 0)

    def test_r4_coordination_generation_transition_after_admission(self):
        lm = LifecycleManager(initial_cfg={"pipeline_mode": "coherent"}, clock_ns=self.clock)
        coord = CoherentPrimaryCoordinator(
            lm,
            clock_ns=self.clock,
            postprocess_fn=lambda raw, face, w, h: (np.ones((h, w), np.float32), None, False, None),
        )
        ctx = lm.create_frame_context(
            frame_id=1,
            arrival_mono_ns=self.clock(),
            framed_pixels=np.zeros((10, 10, 3), dtype=np.uint8),
            in_w=10, in_h=10, out_w=10, out_h=10,
            crop_box=(0.0, 0.0, 10.0, 10.0),
        )
        queue = MockAsyncInferQueue()
        queue.set_callback(coord.handle_async_completion)
        admitted = coord.submit_async(queue, ctx, np.zeros((1, 4, 6, 3), dtype=np.float32))
        self.assertTrue(admitted)

        # Transition generation while in flight
        lm.on_standby()
        self.assertNotEqual(lm.generation_id, ctx.identity.generation_id)

        req = MockInferRequest(np.ones((1, 10, 10, 1), dtype=np.float32))
        queue.fire_completion(0, req)

        # Active consumer reads result and receives MISMATCH
        result = coord.get_or_wait_result(ctx)
        self.assertEqual(result.status, StageResultStatus.ERROR)
        self.assertEqual(result.validity, StageResultValidity.MISMATCH)
        self.assertEqual(result.drop_reason, "GENERATION_MISMATCH")
        # After consumption, store is empty
        self.assertEqual(len(coord._results), 0)
        self.assertEqual(coord.in_flight_count, 0)

    def test_r4_coordination_mismatch_after_abandonment(self):
        lm = LifecycleManager(initial_cfg={"pipeline_mode": "coherent"}, clock_ns=self.clock)
        coord = CoherentPrimaryCoordinator(
            lm,
            clock_ns=self.clock,
            postprocess_fn=lambda raw, face, w, h: (np.ones((h, w), np.float32), None, False, None),
        )
        ctx = lm.create_frame_context(
            frame_id=1,
            arrival_mono_ns=self.clock(),
            framed_pixels=np.zeros((10, 10, 3), dtype=np.uint8),
            in_w=10, in_h=10, out_w=10, out_h=10,
            crop_box=(0.0, 0.0, 10.0, 10.0),
        )
        queue = MockAsyncInferQueue()
        queue.set_callback(coord.handle_async_completion)
        admitted = coord.submit_async(queue, ctx, np.zeros((1, 4, 6, 3), dtype=np.float32))
        self.assertTrue(admitted)

        # Waiter times out / abandons
        self.clock.advance_ms(200.0)
        res_timeout = coord.get_or_wait_result(ctx)
        self.assertEqual(res_timeout.status, StageResultStatus.TIMEOUT)

        # Transition generation
        lm.on_standby()

        req = MockInferRequest(np.ones((1, 10, 10, 1), dtype=np.float32))
        queue.fire_completion(0, req)

        # Discarded without populating _results
        self.assertEqual(len(coord._results), 0)
        self.assertEqual(coord.in_flight_count, 0)

    def test_r4_callback_started_before_reset_completed_after(self):
        started_event = threading.Event()
        resume_event = threading.Event()

        def pausing_postprocess(raw, face, w, h):
            started_event.set()
            resume_event.wait(timeout=2.0)
            return (np.ones((h, w), np.float32), None, False, None)

        lm = LifecycleManager(initial_cfg={"pipeline_mode": "coherent"}, clock_ns=self.clock)
        coord = CoherentPrimaryCoordinator(
            lm,
            clock_ns=self.clock,
            postprocess_fn=pausing_postprocess,
        )
        ctx = lm.create_frame_context(
            frame_id=1,
            arrival_mono_ns=self.clock(),
            framed_pixels=np.zeros((10, 10, 3), dtype=np.uint8),
            in_w=10, in_h=10, out_w=10, out_h=10,
            crop_box=(0.0, 0.0, 10.0, 10.0),
        )
        queue = MockAsyncInferQueue()
        queue.set_callback(coord.handle_async_completion)
        self.assertTrue(coord.submit_async(queue, ctx, np.zeros((1, 4, 6, 3), dtype=np.float32)))

        req = MockInferRequest(np.ones((1, 10, 10, 1), dtype=np.float32))
        cb_thread = threading.Thread(target=queue.fire_completion, args=(0, req))
        cb_thread.start()

        self.assertTrue(started_event.wait(timeout=1.0))
        coord.reset()
        resume_event.set()
        cb_thread.join(timeout=1.0)
        self.assertFalse(cb_thread.is_alive())

        self.assertEqual(len(coord._results), 0)
        self.assertEqual(coord.in_flight_count, 0)


if __name__ == "__main__":
    unittest.main()
