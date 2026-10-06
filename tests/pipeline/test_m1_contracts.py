import numpy as np
from pathlib import Path
import sys
import unittest
from dataclasses import FrozenInstanceError

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


class M1ContractsTests(unittest.TestCase):
    def test_frame_identity_immutability(self):
        ident = FrameIdentity(
            session_id="sess_1",
            generation_id="gen_1",
            frame_id=10,
            processing_config_epoch=1,
            model_id="seg_v1",
            model_version="1.0",
            geometry_id="1280x720",
            seq=5,
        )
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            ident.frame_id = 11

    def test_frame_identity_compatibility(self):
        base = FrameIdentity(
            session_id="sess_1",
            generation_id="gen_1",
            frame_id=10,
            processing_config_epoch=1,
            model_id="seg_v1",
            model_version="1.0",
            geometry_id="1280x720",
            seq=5,
        )
        # Exact match
        same = FrameIdentity(
            session_id="sess_1",
            generation_id="gen_1",
            frame_id=10,
            processing_config_epoch=1,
            model_id="seg_v1",
            model_version="1.0",
            geometry_id="1280x720",
            seq=5,
        )
        self.assertTrue(base.is_compatible_with(same))

        # Session mismatch
        diff_sess = FrameIdentity(
            session_id="sess_2",
            generation_id="gen_1",
            frame_id=10,
            processing_config_epoch=1,
            model_id="seg_v1",
            model_version="1.0",
            geometry_id="1280x720",
            seq=5,
        )
        self.assertFalse(base.is_compatible_with(diff_sess))

        # Generation mismatch
        diff_gen = FrameIdentity(
            session_id="sess_1",
            generation_id="gen_2",
            frame_id=10,
            processing_config_epoch=1,
            model_id="seg_v1",
            model_version="1.0",
            geometry_id="1280x720",
            seq=5,
        )
        self.assertFalse(base.is_compatible_with(diff_gen))

        # Epoch mismatch
        diff_epoch = FrameIdentity(
            session_id="sess_1",
            generation_id="gen_1",
            frame_id=10,
            processing_config_epoch=2,
            model_id="seg_v1",
            model_version="1.0",
            geometry_id="1280x720",
            seq=5,
        )
        self.assertFalse(base.is_compatible_with(diff_epoch))

        # Model mismatch
        diff_model = FrameIdentity(
            session_id="sess_1",
            generation_id="gen_1",
            frame_id=10,
            processing_config_epoch=1,
            model_id="seg_v2",
            model_version="1.0",
            geometry_id="1280x720",
            seq=5,
        )
        self.assertFalse(base.is_compatible_with(diff_model))

        # Geometry mismatch
        diff_geom = FrameIdentity(
            session_id="sess_1",
            generation_id="gen_1",
            frame_id=10,
            processing_config_epoch=1,
            model_id="seg_v1",
            model_version="1.0",
            geometry_id="1920x1080",
            seq=5,
        )
        self.assertFalse(base.is_compatible_with(diff_geom))

        # Frame ID mismatch
        diff_frame = FrameIdentity(
            session_id="sess_1",
            generation_id="gen_1",
            frame_id=11,
            processing_config_epoch=1,
            model_id="seg_v1",
            model_version="1.0",
            geometry_id="1280x720",
            seq=5,
        )
        self.assertFalse(base.is_compatible_with(diff_frame))

    def test_stage_result_isolated_memory(self):
        ident = FrameIdentity(
            session_id="s1", generation_id="g1", frame_id=1,
            processing_config_epoch=1, model_id="m", model_version="1.0",
            geometry_id="geom", seq=1
        )
        raw_mask = np.ones((10, 10), dtype=np.float32)
        res = StageResult(
            identity=ident,
            status=StageResultStatus.OK,
            validity=StageResultValidity.EXACT,
            t_submit_mono_ns=1000,
            t_complete_mono_ns=2000,
            payload={"p_person": raw_mask, "has_hands": True},
        )
        self.assertTrue(res.has_hands)
        self.assertIsNone(res.drop_reason)

        # Verify get_mask() returns a copy
        mask_copy1 = res.get_mask("p_person")
        self.assertTrue(np.all(mask_copy1 == 1.0))
        mask_copy1[0, 0] = 999.0

        mask_copy2 = res.get_mask("p_person")
        self.assertEqual(mask_copy2[0, 0], 1.0)
        self.assertNotEqual(mask_copy1[0, 0], mask_copy2[0, 0])

    def test_frame_context_immutability_and_deadlines(self):
        lm = LifecycleManager("sess_test", initial_cfg={"pipeline_mode": "coherent"})
        lm.m1_config = M1Config(frame_deadline_ms=200.0, compose_reserve_ms=50.0, safety_margin_ms=10.0)

        pixels = np.zeros((100, 100, 3), dtype=np.uint8)
        arr_ns = 1_000_000_000
        ctx = lm.create_frame_context(
            frame_id=42,
            arrival_mono_ns=arr_ns,
            framed_pixels=pixels,
            in_w=100,
            in_h=100,
            out_w=100,
            out_h=100,
            crop_box=(0.0, 0.0, 100.0, 100.0),
            face_info={"has_face": True},
            model_id="mod1",
            model_version="1.0",
            seq=42,
        )

        # Read-only framed_pixels buffer check
        with self.assertRaises(ValueError):
            ctx.framed_pixels[0, 0, 0] = 255

        # Absolute deadlines check
        self.assertEqual(ctx.frame_deadline_mono_ns, arr_ns + 200_000_000)
        self.assertEqual(ctx.primary_decision_deadline_mono_ns, arr_ns + (200_000_000 - 60_000_000))
        self.assertTrue(ctx.primary_decision_deadline_mono_ns < ctx.frame_deadline_mono_ns)

    def test_output_frame_constraints(self):
        ident = FrameIdentity(
            session_id="s1", generation_id="g1", frame_id=1,
            processing_config_epoch=1, model_id="m", model_version="1.0",
            geometry_id="geom", seq=1
        )
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        out = OutputFrame(
            image=img,
            identity=ident,
            commit_mono_ns=500,
            output_type="LIVE",
            applied_results=["primary_exact"],
        )

        # Pre-M5 requirement: presentation_timestamp_ns must be None
        self.assertIsNone(out.presentation_timestamp_ns)

        # Image must be write-protected
        self.assertFalse(out.image.flags.writeable)
        with self.assertRaises(ValueError):
            out.image[0, 0, 0] = 128

    def test_m1_config_validation(self):
        # Valid default
        cfg = M1Config()
        self.assertEqual(cfg.frame_deadline_ms, 250.0)
        self.assertEqual(cfg.compose_reserve_ms, 90.0)
        self.assertEqual(cfg.safety_margin_ms, 10.0)

        # Valid custom
        cfg_custom = M1Config(frame_deadline_ms=100.0, compose_reserve_ms=30.0, safety_margin_ms=5.0)
        self.assertEqual(cfg_custom.frame_deadline_ms, 100.0)

        # Frame deadline out of bounds
        with self.assertRaises(ValueError):
            M1Config(frame_deadline_ms=40.0)
        with self.assertRaises(ValueError):
            M1Config(frame_deadline_ms=1050.0)

        # Compose reserve out of bounds
        with self.assertRaises(ValueError):
            M1Config(compose_reserve_ms=0.5)
        with self.assertRaises(ValueError):
            M1Config(compose_reserve_ms=600.0)

        # Safety margin out of bounds
        with self.assertRaises(ValueError):
            M1Config(safety_margin_ms=-1.0)
        with self.assertRaises(ValueError):
            M1Config(safety_margin_ms=120.0)

        # Sum exceeds deadline
        with self.assertRaises(ValueError):
            M1Config(frame_deadline_ms=100.0, compose_reserve_ms=80.0, safety_margin_ms=25.0)

        # Non-numeric
        with self.assertRaises(ValueError):
            M1Config(frame_deadline_ms="fast")


if __name__ == "__main__":
    unittest.main()
