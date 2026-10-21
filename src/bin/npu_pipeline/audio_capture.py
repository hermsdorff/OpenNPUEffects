"""Non-blocking, deadline-bounded audio capture and output infrastructure for OpenNPUEffects.

Provides bounded incremental I/O, stall detection, generation invalidation barriers,
explicit state management, and backoff policies for dynamic microphone switching.
"""
from dataclasses import dataclass
import enum
import io
import os
import select
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


def _default_monotonic() -> float:
    return time.monotonic()


class CaptureReadStatus(str, enum.Enum):
    """Result of an incremental capture read step."""
    PENDING = "PENDING"      # Incomplete block; returned to control loop without blocking
    COMPLETE = "COMPLETE"    # Exactly block_size_bytes from active generation available
    EOF = "EOF"              # Producer closed pipe; partial discarded, no inference
    ERROR = "ERROR"          # Descriptor or OS failure
    STALL = "STALL"          # Absolute formation deadline exceeded while producer alive
    CANCELLED = "CANCELLED"  # Epoch/source change or shutdown invalidated assembly


class AudioCaptureState(str, enum.Enum):
    """Explicit lifecycle state of the audio capture subsystem."""
    STANDBY = "STANDBY"        # Physical mic released; silence emitted to virtual source
    CONNECTING = "CONNECTING"  # Opening physical capture process
    ACTIVE = "ACTIVE"          # Capturing and processing live physical audio
    SWITCHING = "SWITCHING"    # Transitioning between microphones / generations
    RECOVERING = "RECOVERING"  # Handling stall / disconnect / unavailable source with backoff
    STOPPING = "STOPPING"      # Graceful shutdown


@dataclass
class CaptureBlockResult:
    """Outcome of a single capture read step."""
    status: CaptureReadStatus
    data: Optional[bytes] = None
    discarded_bytes: int = 0
    reason: Optional[str] = None
    generation_id: Optional[str] = None
    epoch: int = 0
    first_byte_time: Optional[float] = None


@dataclass
class WriteResult:
    """Outcome of a bounded output write step."""
    success: bool
    samples_written: int = 0
    bytes_written: int = 0
    pending: bool = False
    partial_discarded_bytes: int = 0
    original_block_bytes: int = 0
    requires_reconnect: bool = False
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AudioCaptureReader:
    """Incremental non-blocking capture reader with absolute deadline detection.

    Guarantees:
    - Never blocks the caller indefinitely.
    - Limits internal buffer to at most 1 block (block_size_bytes).
    - Tracks absolute deadline from block start; byte trickling cannot prolong indefinitely.
    - Tracks discarded bytes on EOF, STALL, and CANCELLED.
    """

    def __init__(
        self,
        block_size_bytes: int = 8192,
        poll_timeout_sec: float = 0.025,
        block_deadline_sec: float = 0.750,
        initial_tolerance_sec: float = 1.500,
        clock: Callable[[], float] = _default_monotonic,
    ):
        self.block_size_bytes = block_size_bytes
        self.poll_timeout_sec = poll_timeout_sec
        self.block_deadline_sec = block_deadline_sec
        self.initial_tolerance_sec = initial_tolerance_sec
        self._clock = clock

        self._buffer = bytearray()
        self._epoch: int = 0
        self._generation_id: str = ""
        self._first_block: bool = True
        self._block_start_time: Optional[float] = None
        self._first_byte_time: Optional[float] = None

        self._proc: Any = None
        self._stream: Any = None
        self._fd: Optional[int] = None
        self._total_discarded_bytes: int = 0

    @property
    def buffer_len(self) -> int:
        return len(self._buffer)

    @property
    def total_discarded_bytes(self) -> int:
        return self._total_discarded_bytes

    @property
    def is_pending(self) -> bool:
        return self._block_start_time is not None or len(self._buffer) > 0

    @property
    def first_byte_time(self) -> Optional[float]:
        return self._first_byte_time

    def pending_age_ms(self) -> float:
        if self._block_start_time is None:
            return 0.0
        return max(0.0, (self._clock() - self._block_start_time) * 1000.0)

    def attach(
        self,
        proc_or_stream: Any,
        generation_id: str = "",
        epoch: int = 0,
        is_first_block: bool = True,
    ) -> int:
        """Attaches a new capture process or stream. Returns any previously discarded bytes."""
        discarded = self.cancel(reason="attach_new_source")
        self._epoch = epoch
        self._generation_id = generation_id
        self._first_block = is_first_block
        self._block_start_time = None
        self._first_byte_time = None

        if hasattr(proc_or_stream, "stdout"):
            self._proc = proc_or_stream
            self._stream = proc_or_stream.stdout
        else:
            self._proc = None
            self._stream = proc_or_stream

        self._fd = None
        if self._stream is not None and hasattr(self._stream, "fileno"):
            try:
                fd = self._stream.fileno()
                if isinstance(fd, int) and fd >= 0:
                    try:
                        os.set_blocking(fd, False)
                        self._fd = fd
                    except Exception:
                        self._fd = None
            except Exception:
                self._fd = None

        return discarded

    def detach(self) -> int:
        """Detaches current source and returns any discarded partial bytes."""
        discarded = self.cancel(reason="detach")
        self._proc = None
        self._stream = None
        self._fd = None
        return discarded

    def cancel(self, reason: str = "cancelled") -> int:
        """Cancels current block assembly and returns number of discarded partial bytes."""
        discarded = len(self._buffer)
        if discarded > 0:
            self._total_discarded_bytes += discarded
            self._buffer.clear()
        self._block_start_time = None
        self._first_byte_time = None
        return discarded

    def set_epoch(self, epoch: int) -> int:
        """Updates epoch. If changed, cancels active assembly and returns discarded bytes."""
        if epoch != self._epoch:
            self._epoch = epoch
            return self.cancel(reason="epoch_change")
        return 0

    def read_step(self, timeout_sec: Optional[float] = None) -> CaptureBlockResult:
        """Performs a bounded incremental read step."""
        if self._stream is None:
            return CaptureBlockResult(
                status=CaptureReadStatus.ERROR,
                reason="no_source_attached",
                epoch=self._epoch,
                generation_id=self._generation_id,
            )

        now = self._clock()
        if self._block_start_time is None:
            self._block_start_time = now

        # Absolute formation deadline check
        deadline = self.initial_tolerance_sec if self._first_block else self.block_deadline_sec
        elapsed = now - self._block_start_time
        if elapsed >= deadline:
            discarded = len(self._buffer)
            self._total_discarded_bytes += discarded
            self._buffer.clear()
            self._block_start_time = None
            self._first_byte_time = None
            return CaptureBlockResult(
                status=CaptureReadStatus.STALL,
                discarded_bytes=discarded,
                reason="deadline_exceeded",
                epoch=self._epoch,
                generation_id=self._generation_id,
            )

        max_poll = self.poll_timeout_sec if timeout_sec is None else timeout_sec
        time_left = max(0.0, deadline - elapsed)
        wait_sec = min(max_poll, time_left)

        remaining = self.block_size_bytes - len(self._buffer)

        # Path 1: Real OS file descriptor (non-blocking)
        if self._fd is not None:
            try:
                rlist, _, _ = select.select([self._fd], [], [], wait_sec)
            except (ValueError, OSError) as e:
                return CaptureBlockResult(
                    status=CaptureReadStatus.ERROR,
                    reason=f"select_failed: {e}",
                    epoch=self._epoch,
                    generation_id=self._generation_id,
                )

            if not rlist:
                # Deadline re-check after wait
                if self._clock() - self._block_start_time >= deadline:
                    discarded = len(self._buffer)
                    self._total_discarded_bytes += discarded
                    self._buffer.clear()
                    self._block_start_time = None
                    self._first_byte_time = None
                    return CaptureBlockResult(
                        status=CaptureReadStatus.STALL,
                        discarded_bytes=discarded,
                        reason="deadline_exceeded",
                        epoch=self._epoch,
                        generation_id=self._generation_id,
                    )
                return CaptureBlockResult(
                    status=CaptureReadStatus.PENDING,
                    epoch=self._epoch,
                    generation_id=self._generation_id,
                )

            try:
                chunk = os.read(self._fd, remaining)
            except (BlockingIOError, InterruptedError):
                return CaptureBlockResult(
                    status=CaptureReadStatus.PENDING,
                    epoch=self._epoch,
                    generation_id=self._generation_id,
                )
            except OSError as e:
                discarded = len(self._buffer)
                self._total_discarded_bytes += discarded
                self._buffer.clear()
                self._block_start_time = None
                self._first_byte_time = None
                return CaptureBlockResult(
                    status=CaptureReadStatus.ERROR,
                    discarded_bytes=discarded,
                    reason=f"read_os_error: {e}",
                    epoch=self._epoch,
                    generation_id=self._generation_id,
                )

            if chunk == b"":
                discarded = len(self._buffer)
                self._total_discarded_bytes += discarded
                self._buffer.clear()
                self._block_start_time = None
                self._first_byte_time = None
                return CaptureBlockResult(
                    status=CaptureReadStatus.EOF,
                    discarded_bytes=discarded,
                    reason="eof_zero_bytes",
                    epoch=self._epoch,
                    generation_id=self._generation_id,
                )

            if len(self._buffer) == 0:
                self._first_byte_time = self._clock()
            self._buffer.extend(chunk)
            if len(self._buffer) >= self.block_size_bytes:
                full_block = bytes(self._buffer[:self.block_size_bytes])
                fb_time = self._first_byte_time
                self._buffer = self._buffer[self.block_size_bytes:]
                self._block_start_time = None
                self._first_byte_time = self._clock() if len(self._buffer) > 0 else None
                self._first_block = False
                return CaptureBlockResult(
                    status=CaptureReadStatus.COMPLETE,
                    data=full_block,
                    epoch=self._epoch,
                    generation_id=self._generation_id,
                    first_byte_time=fb_time,
                )

            return CaptureBlockResult(
                status=CaptureReadStatus.PENDING,
                epoch=self._epoch,
                generation_id=self._generation_id,
            )

        # Path 2: Test double / in-memory stream without OS descriptor
        try:
            chunk = self._stream.read(remaining)
        except Exception as e:
            discarded = len(self._buffer)
            self._total_discarded_bytes += discarded
            self._buffer.clear()
            self._block_start_time = None
            self._first_byte_time = None
            return CaptureBlockResult(
                status=CaptureReadStatus.ERROR,
                discarded_bytes=discarded,
                reason=f"stream_read_error: {e}",
                epoch=self._epoch,
                generation_id=self._generation_id,
            )

        if chunk is None or chunk == b"":
            is_eof = False
            if self._proc is not None:
                if hasattr(self._proc, "poll") and self._proc.poll() is not None:
                    is_eof = True
                elif getattr(self._proc, "terminated", False):
                    is_eof = True
            elif getattr(self._stream, "closed", False) or getattr(self._stream, "eof", False):
                is_eof = True
            elif isinstance(self._stream, io.BytesIO):
                is_eof = True

            if is_eof:
                discarded = len(self._buffer)
                self._total_discarded_bytes += discarded
                self._buffer.clear()
                self._block_start_time = None
                self._first_byte_time = None
                return CaptureBlockResult(
                    status=CaptureReadStatus.EOF,
                    discarded_bytes=discarded,
                    reason="eof_stream_ended",
                    epoch=self._epoch,
                    generation_id=self._generation_id,
                )
            return CaptureBlockResult(
                status=CaptureReadStatus.PENDING,
                epoch=self._epoch,
                generation_id=self._generation_id,
            )

        if len(self._buffer) == 0:
            self._first_byte_time = self._clock()
        self._buffer.extend(chunk)
        if len(self._buffer) >= self.block_size_bytes:
            full_block = bytes(self._buffer[:self.block_size_bytes])
            fb_time = self._first_byte_time
            self._buffer = self._buffer[self.block_size_bytes:]
            self._block_start_time = None
            self._first_byte_time = self._clock() if len(self._buffer) > 0 else None
            self._first_block = False
            return CaptureBlockResult(
                status=CaptureReadStatus.COMPLETE,
                data=full_block,
                epoch=self._epoch,
                generation_id=self._generation_id,
                first_byte_time=fb_time,
            )

        return CaptureBlockResult(
            status=CaptureReadStatus.PENDING,
            epoch=self._epoch,
            generation_id=self._generation_id,
        )


class AudioOutputWriter:
    """Bounded, non-blocking, deadline-checked incremental writer to PipeWire playback.

    Guarantees:
    - Uses os.set_blocking(fd, False) and os.write(fd, view[offset:]) on real OS descriptors.
    - Never blocks in BufferedWriter.write() or BufferedWriter.flush() when a pipe has partial capacity.
    - Tracks pending byte offset across steps up to write_deadline_sec.
    - Caps internal pending buffer to 1 block (block_size_bytes).
    - Never reports success=True or samples_written > 0 unless all bytes of the block are accepted.
    """

    def __init__(
        self,
        block_size_bytes: int = 8192,
        poll_timeout_sec: float = 0.025,
        write_deadline_sec: float = 0.500,
        clock: Callable[[], float] = _default_monotonic,
    ):
        self.block_size_bytes = block_size_bytes
        self.poll_timeout_sec = poll_timeout_sec
        self.write_deadline_sec = write_deadline_sec
        self._clock = clock

        self._pending_buf = bytearray()
        self._offset: int = 0
        self._write_start_time: Optional[float] = None
        self._pending_meta: Optional[Dict[str, Any]] = None
        self._proc: Any = None
        self._fd: Optional[int] = None
        self._fd_is_real_os: bool = False
        self._total_discarded_bytes: int = 0

    @property
    def has_pending(self) -> bool:
        return len(self._pending_buf) > 0 and self._offset < len(self._pending_buf)

    @property
    def pending_offset(self) -> int:
        return self._offset

    @property
    def pending_remaining_bytes(self) -> int:
        if not self.has_pending:
            return 0
        return max(0, len(self._pending_buf) - self._offset)

    @property
    def pending_original_bytes(self) -> int:
        return len(self._pending_buf)

    @property
    def pending_metadata(self) -> Optional[Dict[str, Any]]:
        return self._pending_meta

    def pending_age_ms(self) -> float:
        if not self.has_pending or self._write_start_time is None:
            return 0.0
        return max(0.0, (self._clock() - self._write_start_time) * 1000.0)

    @property
    def total_discarded_bytes(self) -> int:
        return self._total_discarded_bytes

    def cancel_pending(self, reason: str = "cancelled") -> WriteResult:
        """Explicitly cancels any pending output block.

        Separates already delivered bytes (bytes_written), remaining undelivered
        bytes discarded (partial_discarded_bytes), and original_block_bytes.
        If any prefix was already delivered to the sink (bytes_written > 0),
        requires_reconnect is True so no subsequent block splices onto a partial prefix.
        """
        meta = self._pending_meta
        if not self.has_pending:
            self._pending_buf.clear()
            self._offset = 0
            self._write_start_time = None
            self._pending_meta = None
            return WriteResult(
                success=False,
                samples_written=0,
                bytes_written=0,
                pending=False,
                partial_discarded_bytes=0,
                original_block_bytes=0,
                requires_reconnect=False,
                error=reason,
                metadata=meta,
            )

        delivered = self._offset
        original = len(self._pending_buf)
        remaining = max(0, original - delivered)
        if remaining > 0:
            self._total_discarded_bytes += remaining
        self._pending_buf.clear()
        self._offset = 0
        self._write_start_time = None
        self._pending_meta = None
        return WriteResult(
            success=False,
            samples_written=0,
            bytes_written=delivered,
            pending=False,
            partial_discarded_bytes=remaining,
            original_block_bytes=original,
            requires_reconnect=(delivered > 0),
            error=reason,
            metadata=meta,
        )

    def _abort_pending(self, error: str) -> WriteResult:
        """Aborts an active pending block due to sink failure or timeout."""
        delivered = self._offset
        original = len(self._pending_buf)
        remaining = max(0, original - delivered)
        meta = self._pending_meta
        if remaining > 0:
            self._total_discarded_bytes += remaining
        self._pending_buf.clear()
        self._offset = 0
        self._write_start_time = None
        self._pending_meta = None
        return WriteResult(
            success=False,
            samples_written=0,
            bytes_written=delivered,
            pending=False,
            partial_discarded_bytes=remaining,
            original_block_bytes=original,
            requires_reconnect=True,
            error=error,
            metadata=meta,
        )

    def reset(self, reason: str = "reset") -> int:
        """Discards any pending partial output block and resets state."""
        res = self.cancel_pending(reason=reason)
        self._proc = None
        self._fd = None
        self._fd_is_real_os = False
        return res.partial_discarded_bytes

    def _attach_sink(self, proc: Any) -> None:
        if proc is self._proc:
            return
        self.reset(reason="sink_changed")
        self._proc = proc
        stdin = getattr(proc, "stdin", None)
        self._fd = None
        self._fd_is_real_os = False
        if stdin is not None and hasattr(stdin, "fileno"):
            try:
                fd = stdin.fileno()
                if isinstance(fd, int) and fd >= 0:
                    self._fd = fd
                    try:
                        os.set_blocking(fd, False)
                        self._fd_is_real_os = True
                    except OSError:
                        self._fd_is_real_os = False
            except Exception:
                self._fd = None
                self._fd_is_real_os = False

    def write_step(
        self,
        proc: Any,
        data: Optional[bytes] = None,
        timeout_sec: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WriteResult:
        """Performs a single bounded non-blocking write step."""
        if proc is None or getattr(proc, "stdin", None) is None:
            meta = self._pending_meta if metadata is None else dict(metadata)
            cancel_res = self.cancel_pending(reason="no_proc_or_stdin")
            self.reset(reason="no_proc_or_stdin")
            return WriteResult(
                success=False,
                samples_written=0,
                bytes_written=cancel_res.bytes_written,
                partial_discarded_bytes=cancel_res.partial_discarded_bytes,
                original_block_bytes=cancel_res.original_block_bytes,
                requires_reconnect=True,
                error="no_proc_or_stdin",
                metadata=meta,
            )

        stdin = proc.stdin
        if getattr(stdin, "closed", False):
            meta = self._pending_meta if metadata is None else dict(metadata)
            cancel_res = self.cancel_pending(reason="stdin_closed")
            self.reset(reason="stdin_closed")
            return WriteResult(
                success=False,
                samples_written=0,
                bytes_written=cancel_res.bytes_written,
                partial_discarded_bytes=cancel_res.partial_discarded_bytes,
                original_block_bytes=cancel_res.original_block_bytes,
                requires_reconnect=True,
                error="stdin_closed",
                metadata=meta,
            )

        self._attach_sink(proc)

        now = self._clock()
        if data is not None:
            if len(data) == 0:
                return WriteResult(success=True, samples_written=0, bytes_written=0)
            if self.has_pending:
                # Never silently replace an in-flight block with new data.
                # Preserve existing buffer/offset and report the overlap violation.
                return WriteResult(
                    success=False,
                    samples_written=0,
                    bytes_written=self._offset,
                    pending=True,
                    partial_discarded_bytes=0,
                    original_block_bytes=len(self._pending_buf),
                    requires_reconnect=(self._offset > 0),
                    error="pending_write_already_in_progress",
                    metadata=self._pending_meta,
                )
            self._pending_buf = bytearray(data[: self.block_size_bytes])
            self._offset = 0
            self._write_start_time = now
            self._pending_meta = dict(metadata) if metadata is not None else None

        if not self.has_pending:
            return WriteResult(success=True, samples_written=0, bytes_written=0)

        if self._write_start_time is None:
            self._write_start_time = now

        elapsed = now - self._write_start_time
        if elapsed >= self.write_deadline_sec:
            return self._abort_pending("write_timeout_sink_stalled")

        max_poll = self.poll_timeout_sec if timeout_sec is None else timeout_sec
        time_left = max(0.0, self.write_deadline_sec - elapsed)
        wait_sec = min(max_poll, time_left)

        if self._fd is not None:
            try:
                _, wlist, _ = select.select([], [self._fd], [], wait_sec)
            except Exception as e:
                return self._abort_pending(f"select_write_error: {e}")

            if not wlist:
                if not self._fd_is_real_os or (self._clock() - self._write_start_time) >= self.write_deadline_sec:
                    return self._abort_pending("write_timeout_sink_stalled")
                return WriteResult(
                    success=False,
                    samples_written=0,
                    bytes_written=self._offset,
                    pending=True,
                    partial_discarded_bytes=0,
                    original_block_bytes=len(self._pending_buf),
                    metadata=self._pending_meta,
                )

        # Path 1: Real OS non-blocking file descriptor
        if self._fd_is_real_os and self._fd is not None:
            while self._offset < len(self._pending_buf):
                try:
                    n = os.write(self._fd, bytes(self._pending_buf[self._offset :]))
                except (BlockingIOError, InterruptedError):
                    break
                except (BrokenPipeError, OSError) as e:
                    return self._abort_pending(str(e))
                if n <= 0:
                    break
                self._offset += n

            if self._offset >= len(self._pending_buf):
                total_bytes = len(self._pending_buf)
                samples = total_bytes // 4
                meta = self._pending_meta
                self._pending_buf.clear()
                self._offset = 0
                self._write_start_time = None
                self._pending_meta = None
                return WriteResult(
                    success=True,
                    samples_written=samples,
                    bytes_written=total_bytes,
                    original_block_bytes=total_bytes,
                    metadata=meta,
                )

            if (self._clock() - self._write_start_time) >= self.write_deadline_sec:
                return self._abort_pending("write_timeout_sink_stalled")

            return WriteResult(
                success=False,
                samples_written=0,
                bytes_written=self._offset,
                pending=True,
                partial_discarded_bytes=0,
                original_block_bytes=len(self._pending_buf),
                metadata=self._pending_meta,
            )

        # Path 2: Test double / stream without real OS non-blocking descriptor
        try:
            chunk = bytes(self._pending_buf[self._offset :])
            written_ret = stdin.write(chunk)
            if hasattr(stdin, "flush"):
                stdin.flush()
            written = len(chunk) if written_ret is None else int(written_ret)
        except (BlockingIOError, InterruptedError):
            if (self._clock() - self._write_start_time) >= self.write_deadline_sec:
                return self._abort_pending("write_timeout_sink_stalled")
            return WriteResult(
                success=False,
                samples_written=0,
                bytes_written=self._offset,
                pending=True,
                partial_discarded_bytes=0,
                original_block_bytes=len(self._pending_buf),
                metadata=self._pending_meta,
            )
        except (BrokenPipeError, OSError) as e:
            return self._abort_pending(str(e))

        if written <= 0:
            return self._abort_pending("zero_write_to_sink")

        self._offset += written
        if self._offset >= len(self._pending_buf):
            total_bytes = len(self._pending_buf)
            samples = total_bytes // 4
            meta = self._pending_meta
            self._pending_buf.clear()
            self._offset = 0
            self._write_start_time = None
            self._pending_meta = None
            return WriteResult(
                success=True,
                samples_written=samples,
                bytes_written=total_bytes,
                original_block_bytes=total_bytes,
                metadata=meta,
            )

        total_expected = len(self._pending_buf)
        return WriteResult(
            success=False,
            samples_written=0,
            bytes_written=self._offset,
            pending=True,
            partial_discarded_bytes=0,
            original_block_bytes=total_expected,
            error=f"short_write_{self._offset}_of_{total_expected}",
            metadata=self._pending_meta,
        )

    def write_block(
        self,
        proc: Any,
        data: Optional[bytes] = None,
        wait_for_completion: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WriteResult:
        """Bounded output write operation.

        By default (wait_for_completion=False), executes at most one bounded
        poll_timeout_sec step and yields control back to the caller if the OS pipe
        is still pending, while immediately rejecting short writes on non-OS test doubles.
        If wait_for_completion=True, steps until completion or write_deadline_sec.
        """
        res = self.write_step(proc, data=data, metadata=metadata)
        if not res.pending:
            return res

        # If stream is a synchronous test double without a real OS non-blocking fd,
        # a partial write cannot be completed asynchronously; reject immediately.
        if not self._fd_is_real_os:
            return self._abort_pending(res.error or f"short_write_{self._offset}_of_{len(self._pending_buf)}")

        if wait_for_completion:
            while res.pending:
                res = self.write_step(proc, data=None)
        return res


class CaptureRetryPolicy:
    """Backoff policy for capture/sink reconnections.

    Steps: 250ms, 500ms, 1000ms, 2000ms (capped at 2000ms).
    Resets backoff after 10 consecutive stable complete blocks.
    """

    def __init__(
        self,
        backoff_steps: Tuple[float, ...] = (0.250, 0.500, 1.000, 2.000),
        stable_blocks_for_reset: int = 10,
        clock: Callable[[], float] = _default_monotonic,
    ):
        self.backoff_steps = backoff_steps
        self.stable_blocks_for_reset = stable_blocks_for_reset
        self._clock = clock

        self._consecutive_failures: int = 0
        self._consecutive_successes: int = 0
        self._last_failure_time: float = 0.0
        self._current_backoff: float = self.backoff_steps[0]

    @property
    def current_backoff(self) -> float:
        return self._current_backoff

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def consecutive_successes(self) -> int:
        return self._consecutive_successes

    def record_success(self) -> None:
        """Records a successful block. Resets backoff after reaching threshold."""
        self._consecutive_successes += 1
        if self._consecutive_successes >= self.stable_blocks_for_reset:
            self._consecutive_failures = 0
            self._current_backoff = self.backoff_steps[0]

    def record_failure(self) -> float:
        """Records a failure and advances to next backoff step."""
        self._consecutive_successes = 0
        self._last_failure_time = self._clock()
        idx = min(self._consecutive_failures, len(self.backoff_steps) - 1)
        self._current_backoff = self.backoff_steps[idx]
        self._consecutive_failures += 1
        return self._current_backoff

    def can_retry(self) -> bool:
        """Returns True if the backoff duration since last failure has elapsed."""
        if self._consecutive_failures == 0:
            return True
        return (self._clock() - self._last_failure_time) >= self._current_backoff

    def time_until_retry(self) -> float:
        """Returns seconds remaining before retry is allowed."""
        if self._consecutive_failures == 0:
            return 0.0
        elapsed = self._clock() - self._last_failure_time
        return max(0.0, self._current_backoff - elapsed)


class AudioSwitchCoordinator:
    """Coordinates atomic input source switching, epoch barrier, and generation identity."""

    def __init__(
        self,
        initial_desired_source: str = "auto",
        initial_active_source: Optional[str] = None,
        clock: Callable[[], float] = _default_monotonic,
    ):
        self.desired_source: str = initial_desired_source
        self.active_source: Optional[str] = initial_active_source
        self.state: AudioCaptureState = AudioCaptureState.STANDBY
        self.epoch: int = 1
        self.audio_gen_id: str = "audio-gen-init"
        self.first_sample_index: int = 0
        self._clock = clock

    def request_source_switch(self, new_desired_source: str) -> bool:
        """Registers a new input source intention. Advances epoch immediately.

        Returns True if the intention changed.
        """
        if new_desired_source == self.desired_source:
            return False

        self.desired_source = new_desired_source
        self.epoch += 1
        self.state = AudioCaptureState.SWITCHING
        return True

    def notify_source_change(self) -> None:
        """Advances epoch and marks state as SWITCHING when physical mic changes."""
        self.epoch += 1
        self.state = AudioCaptureState.SWITCHING

    def begin_connecting(self, target_mic: str, new_gen_id: str) -> None:
        """Updates identity when starting a new capture stream."""
        self.active_source = target_mic
        self.audio_gen_id = new_gen_id
        self.first_sample_index = 0
        self.state = AudioCaptureState.CONNECTING

    def mark_active(self) -> None:
        """Marks capture as active upon arrival of first valid block."""
        self.state = AudioCaptureState.ACTIVE

    def mark_recovering(self) -> None:
        """Marks capture as recovering after stall, disconnect, or unavailable source."""
        self.active_source = None
        self.state = AudioCaptureState.RECOVERING

    def mark_standby(self) -> None:
        """Marks capture as standby when no consumers exist."""
        self.active_source = None
        self.state = AudioCaptureState.STANDBY

    def mark_stopping(self) -> None:
        """Marks capture as stopping during daemon shutdown."""
        self.active_source = None
        self.state = AudioCaptureState.STOPPING

    def is_block_valid(self, result: CaptureBlockResult) -> bool:
        """Validates that a block matches active epoch and generation."""
        if result.status != CaptureReadStatus.COMPLETE:
            return False
        return result.epoch == self.epoch and result.generation_id == self.audio_gen_id
