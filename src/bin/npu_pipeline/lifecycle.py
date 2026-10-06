"""Lifecycle, epoch, cancellation, invalidation, and safety barriers (AV-M1-04, AV-M1-05).

Centralizes generation transitions, privacy barrier guarantees, output authority,
and authoritative telemetry counters across all pipeline components.
"""
import copy
from dataclasses import dataclass
import logging
import math
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple
import uuid
import cv2
import numpy as np

from .contracts import (
    FrameContext,
    FrameIdentity,
    M1Config,
    OutputFrame,
)


VALID_PIPELINE_MODES = ("legacy", "coherent", "safe")
NON_FUNCTIONAL_KEYS = {"m0_telemetry", "telemetry", "diagnostics", "perf_stats", "debug", "log_level", "logging"}


def _extract_functional_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in cfg.items() if k not in NON_FUNCTIONAL_KEYS}


class LifecycleManager:
    """Central authority for pipeline lifecycle, epochs, safety barriers, and output repetition."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        initial_cfg: Optional[Dict[str, Any]] = None,
        max_repeat_age_ms: float = 500.0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ):
        self._lock = threading.RLock()
        self.clock_ns = clock_ns
        self.session_id = session_id or str(uuid.uuid4())
        self.generation_id = f"video-gen-{uuid.uuid4().hex[:8]}"
        self.epoch = 0
        self.processing_config_epoch = 1
        self.config_version = 1
        self.max_repeat_age_ms = max_repeat_age_ms
        self.last_capture_mono_ns = self.clock_ns()
        self.last_committed_frame_id = -1
        self._frame_arrivals: Dict[int, int] = {}

        # Pipeline mode & M1 config validation
        cfg = initial_cfg or {}
        raw_mode = cfg.get("pipeline_mode", "legacy")
        if raw_mode not in VALID_PIPELINE_MODES:
            logging.warning(f"[M1-CONFIG] Invalid pipeline mode {raw_mode!r}; defaulting to safe mode")
            requested_mode = "safe"
        else:
            requested_mode = raw_mode

        m1_valid = True
        try:
            self.m1_config = M1Config.from_dict(cfg.get("m1", {}))
        except Exception as e:
            logging.warning(f"[M1-CONFIG] Initial M1 configuration invalid: {e}")
            self.m1_config = M1Config()
            m1_valid = False
        self.last_valid_m1_config = self.m1_config

        if requested_mode == "coherent" and not m1_valid:
            logging.warning("[M1-CONFIG] Failing safe due to invalid initial coherent configuration")
            self.pipeline_mode = "safe"
        else:
            self.pipeline_mode = requested_mode

        self.functional_cfg = _extract_functional_cfg(cfg)
        self.functional_cfg["pipeline_mode"] = self.pipeline_mode

        # Privacy state & temporal watermark
        self.privacy_active = False
        self.privacy_epoch = 0
        self.privacy_watermark = 0
        self.privacy_unlock_mono_ns = 0
        self.privacy_muted = False

        # Output frame tracking
        self.last_output_frame: Optional[OutputFrame] = None

        # Authoritative counters (AV-M1-06)
        self.primary_exact_accepted_total = 0
        self.primary_frame_dropped_total = 0
        self.primary_late_callback_total = 0
        self.primary_identity_rejected_total = 0
        self.generation_rejected_total = 0
        self.aux_omitted_pending_m3_total = 0
        self.compose_deadline_miss_total = 0
        self.new_output_frame_total = 0
        self.output_repeat_total = 0
        self.safe_output_total = 0
        self.privacy_barrier_total = 0

    @property
    def current_epoch(self) -> int:
        return self.processing_config_epoch

    def invalidate(self, reason: str, new_generation: bool = False) -> int:
        """Central invalidation barrier.

        Increments epoch and invalidates in-flight jobs and cached output frames.
        """
        with self._lock:
            self.epoch += 1
            if new_generation:
                self.generation_id = f"video-gen-{uuid.uuid4().hex[:8]}"
                self._frame_arrivals.clear()
            self.last_output_frame = None
            self.last_committed_frame_id = -1
            logging.info(
                f"[M1-LIFECYCLE] Invalidation: reason={reason} epoch={self.epoch} "
                f"gen={self.generation_id} mode={self.pipeline_mode}"
            )
            return self.epoch

    def on_standby(self) -> None:
        self.invalidate("standby", new_generation=True)

    def on_resume(self) -> None:
        self.invalidate("resume", new_generation=True)

    def on_reconnect(self) -> None:
        self.invalidate("reconnect", new_generation=False)

    def set_pipeline_mode(self, mode: str) -> None:
        with self._lock:
            if mode not in VALID_PIPELINE_MODES:
                logging.warning(f"[M1-LIFECYCLE] Invalid pipeline mode {mode!r}; defaulting to safe mode")
                mode = "safe"
            if mode != self.pipeline_mode:
                old_mode = self.pipeline_mode
                self.pipeline_mode = mode
                self.epoch += 1
                self.generation_id = f"video-gen-{uuid.uuid4().hex[:8]}"
                self.last_output_frame = None
                self.last_committed_frame_id = -1
                self.functional_cfg["pipeline_mode"] = mode
                self.processing_config_epoch += 1
                logging.info(f"[M1-LIFECYCLE] Mode switched: {old_mode} -> {mode}")

    def update_config(self, cfg: Dict[str, Any]) -> None:
        """Updates lifecycle configuration with validation and epoch management."""
        with self._lock:
            self.config_version += 1
            raw_mode = cfg.get("pipeline_mode", self.pipeline_mode)
            if raw_mode not in VALID_PIPELINE_MODES:
                logging.warning(f"[M1-CONFIG] Invalid pipeline mode {raw_mode!r}; failing safe")
                new_mode = "safe"
            else:
                new_mode = raw_mode

            # Validate M1 config if present
            if "m1" in cfg:
                try:
                    parsed = M1Config.from_dict(cfg["m1"])
                    self.m1_config = parsed
                    self.last_valid_m1_config = parsed
                except Exception as e:
                    logging.warning(
                        f"[M1-CONFIG] Invalid M1 configuration rejected: {e}. "
                        f"Retaining last valid config: {self.last_valid_m1_config}"
                    )
                    self.m1_config = self.last_valid_m1_config

            if new_mode != self.pipeline_mode:
                old_mode = self.pipeline_mode
                self.pipeline_mode = new_mode
                self.epoch += 1
                self.generation_id = f"video-gen-{uuid.uuid4().hex[:8]}"
                self.last_output_frame = None
                self.last_committed_frame_id = -1
                logging.info(f"[M1-LIFECYCLE] Mode switched: {old_mode} -> {new_mode}")

            # Functional config epoch only increments when functional settings change
            new_functional = _extract_functional_cfg(cfg)
            new_functional["pipeline_mode"] = self.pipeline_mode
            if new_functional != self.functional_cfg:
                self.functional_cfg = new_functional
                self.processing_config_epoch += 1

    def set_privacy(self, enable: bool, set_audio_mute_fn: Optional[Callable[[bool], None]] = None) -> bool:
        """Enforces the privacy barrier (AV-M1-05).

        When enabled, establishes watermark/epoch, clears last output frame
        so camera pixels are never repeated or restored, and mutes audio idempotently.
        """
        with self._lock:
            self.epoch += 1
            self.processing_config_epoch += 1
            self.privacy_epoch += 1
            self.privacy_watermark += 1
            self.last_output_frame = None  # Clear immediately: no leaks
            self.last_committed_frame_id = -1
            self._frame_arrivals.clear()

            if enable:
                self.privacy_active = True
                self.privacy_barrier_total += 1
                self.privacy_unlock_mono_ns = float("inf")
                if set_audio_mute_fn and not self.privacy_muted:
                    try:
                        set_audio_mute_fn(True)
                        self.privacy_muted = True
                    except Exception as e:
                        logging.warning(f"[M1-PRIVACY] Failed to mute audio: {e}")
                        self.privacy_muted = False
                return True
            else:
                self.privacy_active = False
                self.privacy_unlock_mono_ns = self.clock_ns()
                if set_audio_mute_fn and self.privacy_muted:
                    try:
                        set_audio_mute_fn(False)
                        self.privacy_muted = False
                    except Exception as e:
                        logging.warning(f"[M1-PRIVACY] Failed to unmute audio: {e}")
                        self.privacy_muted = True
                return False

    def can_repeat_last_output(self, now_mono_ns: int, current_identity: FrameIdentity) -> bool:
        """Determines if the last completed OutputFrame is eligible for repetition."""
        with self._lock:
            if self.pipeline_mode != "coherent":
                return False
            if self.privacy_active:
                return False
            if self.last_output_frame is None:
                return False
            if self.privacy_unlock_mono_ns > 0:
                last_gen = self.last_output_frame.identity.generation_id
                last_fid = self.last_output_frame.identity.frame_id
                last_arr = (
                    getattr(self.last_output_frame, "arrival_mono_ns", None)
                    or getattr(self.last_output_frame.identity, "arrival_mono_ns", None)
                    or self._frame_arrivals.get((last_gen, last_fid))
                    or self._frame_arrivals.get(last_fid)
                )
                if last_arr is None or last_arr < self.privacy_unlock_mono_ns:
                    return False
            last_id = self.last_output_frame.identity
            if last_id.generation_id != current_identity.generation_id:
                return False
            if last_id.processing_config_epoch != current_identity.processing_config_epoch:
                return False
            if last_id.geometry_id != current_identity.geometry_id:
                return False
            age_ms = (now_mono_ns - self.last_output_frame.commit_mono_ns) / 1_000_000
            if age_ms < 0 or age_ms > self.max_repeat_age_ms:
                return False
            return True

    def create_repeated_frame(self, now_mono_ns: int, current_identity: FrameIdentity) -> Optional[OutputFrame]:
        """Creates a repeated OutputFrame without incrementing unique frame counts."""
        with self._lock:
            if not self.can_repeat_last_output(now_mono_ns, current_identity):
                return None
            self.output_repeat_total += 1
            src_frame = self.last_output_frame
            src_arr = (
                getattr(src_frame, "arrival_mono_ns", None)
                or getattr(src_frame.identity, "arrival_mono_ns", None)
                or self._frame_arrivals.get((src_frame.identity.generation_id, src_frame.identity.frame_id))
                or self._frame_arrivals.get(src_frame.identity.frame_id)
            )
            return OutputFrame(
                image=src_frame.image,
                identity=src_frame.identity,
                commit_mono_ns=now_mono_ns,
                output_type="REPEATED",
                applied_results=["repeated"],
                quality_flags=dict(src_frame.quality_flags),
                source_frame_id=getattr(src_frame, "source_frame_id", None) or src_frame.identity.frame_id,
                source_generation_id=getattr(src_frame, "source_generation_id", None) or src_frame.identity.generation_id,
                attempt_frame_id=current_identity.frame_id,
                arrival_mono_ns=src_arr,
            )

    def commit_live_frame(self, output_frame: OutputFrame, deadline_mono_ns: int, now_mono_ns: int) -> bool:
        """Commits a completed live output frame after verifying all safety barriers."""
        with self._lock:
            if self.pipeline_mode != "coherent":
                return False
            if self.privacy_active:
                return False
            gen_id = output_frame.identity.generation_id
            fid = output_frame.identity.frame_id
            arr = (
                getattr(output_frame, "arrival_mono_ns", None)
                or getattr(output_frame.identity, "arrival_mono_ns", None)
                or self._frame_arrivals.get((gen_id, fid))
                or self._frame_arrivals.get(fid)
            )
            # Enforce privacy temporal barrier:
            # 1. Any frame captured prior to unlock is rejected.
            # 2. Any camera output with unknown/missing arrival timestamp post-unlock is rejected.
            if self.privacy_unlock_mono_ns > 0:
                if arr is None or arr < self.privacy_unlock_mono_ns:
                    self.privacy_barrier_total += 1
                    return False
            if now_mono_ns > deadline_mono_ns:
                self.compose_deadline_miss_total += 1
                return False
            if output_frame.identity.generation_id != self.generation_id:
                self.generation_rejected_total += 1
                return False
            if output_frame.identity.processing_config_epoch != self.processing_config_epoch:
                return False
            if output_frame.identity.frame_id <= self.last_committed_frame_id:
                return False

            self.last_committed_frame_id = output_frame.identity.frame_id
            self.last_output_frame = output_frame
            self.new_output_frame_total += 1
            if arr is not None:
                self._frame_arrivals[(gen_id, fid)] = arr
                self._frame_arrivals[fid] = arr
                if len(self._frame_arrivals) > 400:
                    for k in list(self._frame_arrivals.keys())[:200]:
                        self._frame_arrivals.pop(k, None)
            return True

    def get_frame_arrival(self, frame_id: Optional[int], generation_id: Optional[str] = None) -> Optional[int]:
        if frame_id is None:
            return None
        with self._lock:
            if generation_id is not None and (generation_id, frame_id) in self._frame_arrivals:
                return self._frame_arrivals[(generation_id, frame_id)]
            return self._frame_arrivals.get(frame_id)

    def record_safe_output(self) -> None:
        with self._lock:
            self.safe_output_total += 1

    def make_identity(
        self,
        frame_id: int,
        in_w: int,
        in_h: int,
        out_w: int,
        out_h: int,
        crop_box: Tuple[float, float, float, float],
        model_id: str = "multiclass",
        model_version: str = "1.0",
        seq: Optional[int] = None,
        generation_id: Optional[str] = None,
        arrival_mono_ns: Optional[int] = None,
    ) -> FrameIdentity:
        """Factory for a frozen FrameIdentity matching the current lifecycle state."""
        with self._lock:
            effective_gen = generation_id if generation_id is not None else self.generation_id
            geom_id = f"{in_w}x{in_h}->{crop_box[0]:.1f},{crop_box[1]:.1f},{crop_box[2]:.1f},{crop_box[3]:.1f}->{out_w}x{out_h}"
            return FrameIdentity(
                session_id=self.session_id,
                generation_id=effective_gen,
                frame_id=frame_id,
                processing_config_epoch=self.processing_config_epoch,
                model_id=model_id,
                model_version=model_version,
                geometry_id=geom_id,
                seq=seq,
                arrival_mono_ns=arrival_mono_ns,
            )

    def create_frame_context(
        self,
        frame_id: int,
        arrival_mono_ns: int,
        framed_pixels: np.ndarray,
        in_w: int,
        in_h: int,
        out_w: int,
        out_h: int,
        crop_box: Tuple[float, float, float, float],
        face_info: Optional[Dict[str, Any]] = None,
        model_id: str = "multiclass",
        model_version: str = "1.0",
        seq: Optional[int] = None,
        pts_ns: Optional[int] = None,
        timestamp_quality: str = "ARRIVAL_ESTIMATE",
        generation_id: Optional[str] = None,
    ) -> FrameContext:
        """Constructs a frozen FrameContext with exact deadlines and isolated ownership."""
        identity = self.make_identity(
            frame_id=frame_id,
            in_w=in_w,
            in_h=in_h,
            out_w=out_w,
            out_h=out_h,
            crop_box=crop_box,
            model_id=model_id,
            model_version=model_version,
            seq=seq,
            generation_id=generation_id,
            arrival_mono_ns=arrival_mono_ns,
        )
        with self._lock:
            fd_ns = arrival_mono_ns + int(self.m1_config.frame_deadline_ms * 1_000_000)
            pdd_ns = fd_ns - int((self.m1_config.compose_reserve_ms + self.m1_config.safety_margin_ms) * 1_000_000)
            cur_mode = self.pipeline_mode
            cur_epoch = self.epoch
            snapshot_cfg = dict(self.functional_cfg)
            snapshot_cfg["pipeline_mode"] = cur_mode
            snapshot_cfg["epoch"] = cur_epoch
            self._frame_arrivals[(identity.generation_id, frame_id)] = arrival_mono_ns
            self._frame_arrivals[frame_id] = arrival_mono_ns
            if len(self._frame_arrivals) > 400:
                for k in list(self._frame_arrivals.keys())[:200]:
                    self._frame_arrivals.pop(k, None)

        quality = {}
        if face_info is not None:
            quality["face_info"] = copy.deepcopy(face_info)

        return FrameContext(
            arrival_mono_ns=arrival_mono_ns,
            frame_deadline_mono_ns=fd_ns,
            primary_decision_deadline_mono_ns=pdd_ns,
            identity=identity,
            frozen_cfg=snapshot_cfg,
            in_w=in_w,
            in_h=in_h,
            out_w=out_w,
            out_h=out_h,
            crop_box=crop_box,
            framed_pixels=framed_pixels,
            pts_ns=pts_ns,
            timestamp_quality=timestamp_quality,
            quality_flags=quality,
        )

    @classmethod
    def create_safe_screen(cls, width: int, height: int, text: str = "NPU AI Webcam - Standing by") -> np.ndarray:
        """Generates a neutral, dark safe placeholder frame."""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :] = (24, 24, 24)
        cv2.putText(
            canvas,
            text,
            (max(20, width // 2 - 220), height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
        return canvas
