"""Coherent primary segmentation coordinator and adapter (AV-M1-01, AV-M1-02, AV-M1-03).

Provides bounded asynchronous and synchronous execution for primary neural
segmentation, enforcing:
- Single submission authority with pool limit (jobs=2)
- Exact FrameIdentity indexing and validation
- Finite deadline budgets (primary_decision_deadline)
- Input tensor retention until physical inference completion
- Output tensor copy before driver request reuse
- Immediate rejection and accounting of late or mismatched callbacks
- Strict exact_or_drop semantics: never fall back to older masks
"""
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

from .contracts import (
    FrameContext,
    FrameIdentity,
    StageResult,
    StageResultStatus,
    StageResultValidity,
)
from .lifecycle import LifecycleManager


def _extract_fallback_mask(raw: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Extracts a 2D mask from an arbitrary-shaped inference tensor."""
    if raw is None:
        return None
    arr = raw
    if arr.ndim == 4:
        # e.g. (1, 256, 256, 6) or (1, 1, 144, 256)
        arr = arr[0]
    if arr.ndim == 3:
        c_first, c_last = arr.shape[0], arr.shape[2]
        # Check single-channel masks first: (1, H, W) -> arr[0] of shape (H, W)
        if c_first == 1 and c_last > 1:
            return arr[0]
        # (H, W, 1) -> arr[:, :, 0] of shape (H, W)
        elif c_last == 1 and c_first > 1:
            return arr[:, :, 0]
        # Multi-class channels last: (H, W, C) where C <= 16
        elif 1 < c_last <= 16 and c_first > c_last:
            return arr[:, :, 1]
        # Multi-class channels first: (C, H, W) where C <= 16
        elif 1 < c_first <= 16 and c_last > c_first:
            return arr[1, :, :]
        elif c_first == 1:
            return arr[0]
        elif c_last == 1:
            return arr[:, :, 0]
    return arr


class CoherentPrimaryCoordinator:
    """Coordinates primary segmentation inferences under coherent exact_or_drop policy."""

    def __init__(
        self,
        lifecycle: LifecycleManager,
        max_in_flight: int = 2,
        postprocess_fn: Optional[Callable[..., Tuple[np.ndarray, Optional[np.ndarray], bool, Optional[np.ndarray]]]] = None,
        telemetry: Any = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        seg_inp_name: str = "input_tensor",
        seg_out_name: str = "output_tensor",
        input_tensor_name: Optional[str] = None,
        output_tensor_name: Optional[str] = None,
        is_multiclass: bool = True,
    ):
        self.lifecycle = lifecycle
        self.max_in_flight = max_in_flight
        self.postprocess_fn = postprocess_fn
        self.telemetry = telemetry
        self.clock_ns = clock_ns
        self.seg_inp_name = input_tensor_name or seg_inp_name
        self.seg_out_name = output_tensor_name or seg_out_name
        self.is_multiclass = is_multiclass

        self._lock = threading.RLock()
        self._completion_event = threading.Event()
        # In-flight jobs indexed by FrameIdentity
        self._jobs: Dict[FrameIdentity, Dict[str, Any]] = {}
        # Completed StageResults indexed by FrameIdentity (bounded by max_in_flight)
        self._results: Dict[FrameIdentity, StageResult] = {}
        self.last_submit_rejection_reason: Optional[str] = None

    @property
    def in_flight_count(self) -> int:
        with self._lock:
            return sum(
                1 for j in self._jobs.values()
                if not j.get("cancelled", False) and not j.get("abandoned", False)
            )

    @property
    def pending_jobs_count(self) -> int:
        return self.in_flight_count

    def notify_epoch_changed(self) -> None:
        self.reset()

    def notify_generation_changed(self) -> None:
        self.reset()

    def can_submit(self, queue: Any = None) -> bool:
        """Checks if a new primary inference job can be admitted."""
        with self._lock:
            active_jobs = sum(
                1 for j in self._jobs.values()
                if not j.get("cancelled", False) and not j.get("abandoned", False)
            )
            if active_jobs >= self.max_in_flight:
                return False
        if queue is not None:
            if hasattr(queue, "is_ready") and not queue.is_ready():
                return False
        return True

    def _store_result_locked(self, identity: FrameIdentity, result: StageResult) -> None:
        """Stores a StageResult enforcing bounded capacity for unconsumed entries."""
        while len(self._results) >= self.max_in_flight:
            old_id, _ = next(iter(self._results.items()))
            self._results.pop(old_id, None)
            self.lifecycle.primary_frame_dropped_total += 1
        self._results[identity] = result

    def submit_async(
        self,
        arg1: Any,
        arg2: Any,
        arg3: Any,
        seg_inp_name: Optional[str] = None,
    ) -> bool:
        """Submits an inference job to the AsyncInferQueue under exact identity.

        Supports both (queue, frame_ctx, blob) and (frame_ctx, blob, queue) signatures.
        Retains input blob in job dictionary until physical callback finishes.
        """
        if isinstance(arg1, FrameContext):
            frame_ctx, blob, queue = arg1, arg2, arg3
        elif isinstance(arg2, FrameContext):
            queue, frame_ctx, blob = arg1, arg2, arg3
        else:
            raise TypeError("submit_async requires FrameContext, blob np.ndarray, and queue")

        inp_name = seg_inp_name or self.seg_inp_name
        identity = frame_ctx.identity

        with self._lock:
            active_jobs = sum(
                1 for j in self._jobs.values()
                if not j.get("cancelled", False) and not j.get("abandoned", False)
            )
            if active_jobs >= self.max_in_flight:
                self.last_submit_rejection_reason = "MAX_IN_FLIGHT"
                return False
            if hasattr(queue, "is_ready") and not queue.is_ready():
                self.last_submit_rejection_reason = "QUEUE_BUSY"
                return False

            now_ns = self.clock_ns()
            if now_ns >= frame_ctx.primary_decision_deadline_mono_ns:
                # Deadline already expired prior to submission
                self.last_submit_rejection_reason = "DEADLINE_EXPIRED"
                return False

            # Snapshot lifecycle state atomically under lock
            cur_epoch = self.lifecycle.epoch
            cur_config_epoch = self.lifecycle.processing_config_epoch
            cur_gen = self.lifecycle.generation_id

            if identity.generation_id != cur_gen:
                self.lifecycle.generation_rejected_total += 1
                self.last_submit_rejection_reason = "GENERATION_MISMATCH"
                return False

            if identity.processing_config_epoch != cur_config_epoch:
                self.last_submit_rejection_reason = "CONFIG_EPOCH_MISMATCH"
                return False

            self.last_submit_rejection_reason = None
            job_entry = {
                "identity": identity,
                "frame_ctx": frame_ctx,
                "blob": blob,  # Retain reference until physical conclusion
                "epoch": cur_epoch,
                "processing_config_epoch": cur_config_epoch,
                "generation_id": cur_gen,
                "submit_mono_ns": now_ns,
                "cancelled": False,
                "abandoned": False,
                "completed": False,
            }
            self._jobs[identity] = job_entry

            userdata = (
                identity,
                cur_epoch,
                cur_gen,
                frame_ctx,
                blob,
            )

        try:
            queue.start_async({inp_name: blob}, userdata)
            return True
        except Exception as e:
            with self._lock:
                self._jobs.pop(identity, None)
            logging.warning(f"[M1-PRIMARY] Async submission failed: {e}")
            return False

    def handle_async_completion(self, request: Any, userdata: Any) -> None:
        """Callback handler executed when an asynchronous inference finishes.

        Must:
        1. Copy raw output tensor before driver request reuse.
        2. Post-process using frozen FrameContext (never global state).
        3. Check epoch and generation validity.
        4. Check decision deadline.
        5. Atomically decide publication vs abandonment under lock.
        6. Release retained input blob reference upon physical conclusion.
        """
        t_complete_mono_ns = self.clock_ns()

        if not isinstance(userdata, tuple) or len(userdata) < 5:
            logging.error(f"[M1-PRIMARY] Malformed userdata received in callback: {userdata}")
            return

        identity, job_epoch, job_gen_id, frame_ctx, blob_ref = userdata[:5]

        # 1. Output copy: request buffer will be reused by driver immediately after callback returns
        raw_copy = None
        try:
            raw_tensor = request.get_tensor(self.seg_out_name)
            # data[0].copy() ensures private array memory
            raw_copy = raw_tensor.data[0].copy()
        except Exception as e:
            logging.warning(f"[M1-PRIMARY] Failed to extract raw tensor: {e}")

        # 2. Check early cancellation / abandonment / mismatch under lock
        with self._lock:
            cur_epoch = self.lifecycle.epoch
            cur_config_epoch = self.lifecycle.processing_config_epoch
            cur_gen = self.lifecycle.generation_id

            job = self._jobs.get(identity)
            # If job is missing, cancelled, or already abandoned by consumer:
            if job is None or job.get("cancelled", False) or job.get("abandoned", False):
                self._jobs.pop(identity, None)  # Physical completion done, release blob
                self.lifecycle.primary_late_callback_total += 1
                self._completion_event.set()
                return

            has_mismatch = (
                job_epoch != cur_epoch
                or job_gen_id != cur_gen
                or identity.generation_id != cur_gen
                or identity.processing_config_epoch != cur_config_epoch
            )
            if has_mismatch:
                self._jobs.pop(identity, None)  # Physical completion done, release blob
                self.lifecycle.primary_late_callback_total += 1
                self.lifecycle.generation_rejected_total += 1
                # Active consumer is waiting; publish mismatch so get_or_wait_result can consume it
                self._store_result_locked(
                    identity,
                    StageResult(
                        identity=identity,
                        status=StageResultStatus.ERROR,
                        validity=StageResultValidity.MISMATCH,
                        drop_reason="GENERATION_MISMATCH",
                        error_message=f"Generation mismatch: expected {cur_gen}, got {identity.generation_id}",
                    ),
                )
                self._completion_event.set()
                return

        # 3. Postprocess using frozen context (executed outside lock)
        p_person, hand_skin, has_hands, bg_prob = None, None, False, None
        if raw_copy is not None:
            if self.postprocess_fn is not None and frame_ctx.identity.model_id == "multiclass":
                try:
                    face_info = frame_ctx.quality_flags.get("face_info", {"has_face": False})
                    p_person, hand_skin, has_hands, bg_prob = self.postprocess_fn(
                        raw_copy,
                        face_info,
                        frame_ctx.out_w,
                        frame_ctx.out_h,
                    )
                except Exception as e:
                    logging.warning(f"[M1-PRIMARY] Postprocess error: {e}")
            else:
                p_person = _extract_fallback_mask(raw_copy)

        # 4. Check deadline
        is_late = (t_complete_mono_ns > frame_ctx.primary_decision_deadline_mono_ns)

        # 5. Atomic publication under lock
        with self._lock:
            # Physical inference is done: release input blob by popping job
            job = self._jobs.pop(identity, None)
            if job is None or job.get("cancelled", False) or job.get("abandoned", False):
                # Consumer timed out, reset was called, or job abandoned during postprocessing
                self.lifecycle.primary_late_callback_total += 1
                self._completion_event.set()
                return

            cur_epoch = self.lifecycle.epoch
            cur_config_epoch = self.lifecycle.processing_config_epoch
            cur_gen = self.lifecycle.generation_id

            if (
                job_epoch != cur_epoch
                or job_gen_id != cur_gen
                or identity.generation_id != cur_gen
                or identity.processing_config_epoch != cur_config_epoch
            ):
                self.lifecycle.primary_late_callback_total += 1
                self.lifecycle.generation_rejected_total += 1
                self._store_result_locked(
                    identity,
                    StageResult(
                        identity=identity,
                        status=StageResultStatus.ERROR,
                        validity=StageResultValidity.MISMATCH,
                        drop_reason="GENERATION_MISMATCH",
                        error_message=f"Generation mismatch: expected {cur_gen}, got {identity.generation_id}",
                    ),
                )
                self._completion_event.set()
                return

            if is_late:
                self.lifecycle.primary_late_callback_total += 1
                stage_result = StageResult(
                    identity=identity,
                    status=StageResultStatus.TIMEOUT,
                    validity=StageResultValidity.LATE,
                    t_submit_mono_ns=job["submit_mono_ns"],
                    t_complete_mono_ns=t_complete_mono_ns,
                    error_message="Callback completed after primary decision deadline",
                )
            elif p_person is None:
                stage_result = StageResult(
                    identity=identity,
                    status=StageResultStatus.ERROR,
                    validity=StageResultValidity.INVALID,
                    t_submit_mono_ns=job["submit_mono_ns"],
                    t_complete_mono_ns=t_complete_mono_ns,
                    error_message="Inference or postprocess produced empty result",
                )
            else:
                stage_result = StageResult(
                    identity=identity,
                    status=StageResultStatus.OK,
                    validity=StageResultValidity.EXACT,
                    payload={
                        "p_person": p_person,
                        "hand_skin": hand_skin,
                        "has_hands": has_hands,
                        "bg_prob": bg_prob,
                    },
                    t_submit_mono_ns=job["submit_mono_ns"],
                    t_complete_mono_ns=t_complete_mono_ns,
                )

            self._store_result_locked(identity, stage_result)

        self._completion_event.set()

    def get_or_wait_result(
        self,
        frame_ctx: FrameContext,
        timeout_ms: Optional[float] = None,
    ) -> StageResult:
        """Waits for and retrieves the exact StageResult for frame_ctx.

        Under exact_or_drop:
        - If result is missing, late, mismatched, or timed out: returns drop/timeout result.
        - Never returns an older frame's mask.
        """
        identity = frame_ctx.identity
        deadline_mono_ns = frame_ctx.primary_decision_deadline_mono_ns

        stage_res = None
        while True:
            with self._lock:
                if identity in self._results:
                    stage_res = self._results.pop(identity)
                    break
                job = self._jobs.get(identity)
                if job is None or job.get("cancelled", False) or job.get("abandoned", False):
                    # Job was never submitted or was cancelled/abandoned
                    stage_res = None
                    break

            now_mono_ns = self.clock_ns()
            remaining_ns = deadline_mono_ns - now_mono_ns
            if remaining_ns <= 0:
                stage_res = None
                break

            wait_timeout_s = remaining_ns / 1_000_000_000.0
            if timeout_ms is not None:
                wait_timeout_s = min(wait_timeout_s, timeout_ms / 1000.0)

            signaled = self._completion_event.wait(timeout=wait_timeout_s)
            self._completion_event.clear()
            if not signaled:
                stage_res = None
                break

        # Process outcome
        decision_now_mono_ns = self.clock_ns()

        if stage_res is None:
            # Deadline expired or job missing / cancelled / abandoned
            with self._lock:
                # Check if result arrived right before acquiring lock
                if identity in self._results:
                    stage_res = self._results.pop(identity)
                else:
                    job = self._jobs.get(identity)
                    if job is not None:
                        job["abandoned"] = True
                        job["cancelled"] = True
                    self.lifecycle.primary_frame_dropped_total += 1
                    return StageResult(
                        identity=identity,
                        status=StageResultStatus.TIMEOUT,
                        validity=StageResultValidity.LATE,
                        error_message="Primary decision deadline reached without result",
                    )

        # Validate result against decision deadline and compatibility
        if (
            identity.generation_id != self.lifecycle.generation_id
            or identity.processing_config_epoch != self.lifecycle.processing_config_epoch
        ):
            with self._lock:
                self.lifecycle.primary_frame_dropped_total += 1
                self.lifecycle.generation_rejected_total += 1
            return StageResult(
                identity=identity,
                status=StageResultStatus.ERROR,
                validity=StageResultValidity.MISMATCH,
                drop_reason="GENERATION_MISMATCH",
                error_message=f"Generation mismatch: expected {self.lifecycle.generation_id}, got {identity.generation_id}",
            )

        if decision_now_mono_ns > deadline_mono_ns:
            with self._lock:
                self.lifecycle.primary_frame_dropped_total += 1
                self.lifecycle.primary_late_callback_total += 1
            return StageResult(
                identity=identity,
                status=StageResultStatus.TIMEOUT,
                validity=StageResultValidity.LATE,
                error_message="Primary result observed after decision deadline",
            )

        if not stage_res.identity.is_compatible_with(identity):
            with self._lock:
                self.lifecycle.primary_frame_dropped_total += 1
                self.lifecycle.primary_identity_rejected_total += 1
            return StageResult(
                identity=identity,
                status=StageResultStatus.ERROR,
                validity=StageResultValidity.MISMATCH,
                error_message=f"Identity mismatch: expected {identity}, got {stage_res.identity}",
            )

        if stage_res.status != StageResultStatus.OK or stage_res.validity != StageResultValidity.EXACT:
            with self._lock:
                self.lifecycle.primary_frame_dropped_total += 1
            return stage_res

        # Accepted exact result!
        with self._lock:
            self.lifecycle.primary_exact_accepted_total += 1

        return stage_res

    def run_sync(
        self,
        arg1: Any,
        arg2: Any,
        arg3: Any,
        seg_inp_name: Optional[str] = None,
        seg_out_name: Optional[str] = None,
    ) -> StageResult:
        """Synchronous inference path (warmup or when AsyncInferQueue is unavailable).

        Enforces the same finite deadline budget and output copying protections.
        Supports both (frame_ctx, blob, infer_request) and (infer_request, frame_ctx, blob).
        """
        if isinstance(arg1, FrameContext) or hasattr(arg1, "identity"):
            frame_ctx, blob, infer_request = arg1, arg2, arg3
        elif isinstance(arg2, FrameContext) or hasattr(arg2, "identity"):
            infer_request, frame_ctx, blob = arg1, arg2, arg3
        else:
            raise TypeError("run_sync requires FrameContext, blob np.ndarray, and infer_request")

        inp_name = seg_inp_name or self.seg_inp_name
        out_name = seg_out_name or self.seg_out_name
        identity = frame_ctx.identity
        deadline_mono_ns = frame_ctx.primary_decision_deadline_mono_ns

        if (
            frame_ctx.identity.generation_id != self.lifecycle.generation_id
            or frame_ctx.identity.processing_config_epoch != self.lifecycle.processing_config_epoch
        ):
            with self._lock:
                self.lifecycle.primary_frame_dropped_total += 1
                self.lifecycle.generation_rejected_total += 1
            return StageResult(
                identity=identity,
                status=StageResultStatus.ERROR,
                validity=StageResultValidity.MISMATCH,
                drop_reason="GENERATION_MISMATCH",
                error_message=f"Generation mismatch: expected {self.lifecycle.generation_id}, got {frame_ctx.identity.generation_id}",
            )

        t_start = self.clock_ns()
        if t_start >= deadline_mono_ns:
            with self._lock:
                self.lifecycle.primary_frame_dropped_total += 1
            return StageResult(
                identity=identity,
                status=StageResultStatus.TIMEOUT,
                validity=StageResultValidity.LATE,
                error_message="Synchronous inference skipped: deadline already expired",
            )

        try:
            infer_request.infer({inp_name: blob})
            raw_copy = infer_request.get_tensor(out_name).data[0].copy()
        except Exception as e:
            with self._lock:
                self.lifecycle.primary_frame_dropped_total += 1
            return StageResult(
                identity=identity,
                status=StageResultStatus.ERROR,
                validity=StageResultValidity.INVALID,
                error_message=f"Synchronous infer failed: {e}",
            )

        t_inf_end = self.clock_ns()
        if t_inf_end >= deadline_mono_ns:
            with self._lock:
                self.lifecycle.primary_frame_dropped_total += 1
                self.lifecycle.primary_late_callback_total += 1
            return StageResult(
                identity=identity,
                status=StageResultStatus.TIMEOUT,
                validity=StageResultValidity.LATE,
                error_message="Synchronous infer completed after primary decision deadline",
            )

        p_person, hand_skin, has_hands, bg_prob = None, None, False, None
        if self.postprocess_fn is not None and frame_ctx.identity.model_id == "multiclass":
            try:
                face_info = frame_ctx.quality_flags.get("face_info", {"has_face": False})
                p_person, hand_skin, has_hands, bg_prob = self.postprocess_fn(
                    raw_copy,
                    face_info,
                    frame_ctx.out_w,
                    frame_ctx.out_h,
                )
            except Exception as e:
                with self._lock:
                    self.lifecycle.primary_frame_dropped_total += 1
                return StageResult(
                    identity=identity,
                    status=StageResultStatus.ERROR,
                    validity=StageResultValidity.INVALID,
                    error_message=f"Postprocess failed: {e}",
                )
        else:
            p_person = _extract_fallback_mask(raw_copy)

        t_complete = self.clock_ns()
        if t_complete > deadline_mono_ns:
            with self._lock:
                self.lifecycle.primary_frame_dropped_total += 1
                self.lifecycle.primary_late_callback_total += 1
            return StageResult(
                identity=identity,
                status=StageResultStatus.TIMEOUT,
                validity=StageResultValidity.LATE,
                error_message="Synchronous postprocess completed after primary decision deadline",
            )

        with self._lock:
            self.lifecycle.primary_exact_accepted_total += 1

        return StageResult(
            identity=identity,
            status=StageResultStatus.OK,
            validity=StageResultValidity.EXACT,
            payload={
                "p_person": p_person,
                "hand_skin": hand_skin,
                "has_hands": has_hands,
                "bg_prob": bg_prob,
            },
            t_submit_mono_ns=t_start,
            t_complete_mono_ns=t_complete,
        )

    def reset(self) -> None:
        """Invalidates all pending jobs and results on lifecycle boundary.

        Pending jobs are marked cancelled and abandoned so late driver callbacks are rejected.
        Input blob references remain pinned until their callbacks fire.
        """
        with self._lock:
            for job in self._jobs.values():
                job["cancelled"] = True
                job["abandoned"] = True
            self._results.clear()
        self._completion_event.set()
