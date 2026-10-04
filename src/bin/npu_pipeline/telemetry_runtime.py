"""Telemetry runtime, config validation, background consumer, and diagnostics file management for M0."""
import copy
import logging
import math
import os
from pathlib import Path
import stat
import threading
import time
from typing import Any, Callable, Dict, Optional
import uuid

from npu_pipeline.telemetry import (
    BoundedTelemetry,
    NullTelemetry,
    ObservationOrigin,
    SCHEMA_VERSION,
    duration_ms,
    event_json,
    provenance_fields,
)

logger = logging.getLogger(__name__)

DEFAULT_M0_CONFIG = {
    "enabled": False,
    "trace_enabled": False,
    "report_interval_seconds": 5.0,
    "event_capacity": 2048,
    "samples_per_metric": 512,
    "trace_every_n": 30,
    "trace_max_bytes": 10485760,
    "trace_backup_count": 2,
}

_LAST_CONFIG_WARN_TIME = 0.0


def _warn_rate_limited(msg: str, interval: float = 10.0):
    global _LAST_CONFIG_WARN_TIME
    now = time.monotonic()
    if now - _LAST_CONFIG_WARN_TIME >= interval:
        _LAST_CONFIG_WARN_TIME = now
        logger.warning(f"M0-CONFIG: {msg}")


def validate_m0_config(raw_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate and normalize M0 telemetry configuration with explicit boundaries.

    Invalid fields fall back to safe defaults without corrupting operational config.
    Guards against integer overflow, invalid types, and extreme numeric values.
    """
    if raw_cfg is None or not isinstance(raw_cfg, dict):
        return copy.deepcopy(DEFAULT_M0_CONFIG)

    section = raw_cfg.get("m0_telemetry")
    if section is None:
        return copy.deepcopy(DEFAULT_M0_CONFIG)
    if not isinstance(section, dict):
        _warn_rate_limited("m0_telemetry must be a dictionary; using disabled defaults")
        return copy.deepcopy(DEFAULT_M0_CONFIG)

    validated = copy.deepcopy(DEFAULT_M0_CONFIG)

    # enabled: strict boolean, reject strings
    if "enabled" in section:
        val = section["enabled"]
        if isinstance(val, bool):
            validated["enabled"] = val
        else:
            _warn_rate_limited(f"invalid 'enabled' value {val!r}; expected boolean")

    if "trace_enabled" in section:
        val = section["trace_enabled"]
        if isinstance(val, bool):
            validated["trace_enabled"] = val
        else:
            _warn_rate_limited(f"invalid 'trace_enabled' value {val!r}; expected boolean")

    if "report_interval_seconds" in section:
        val = section["report_interval_seconds"]
        try:
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if isinstance(val, int) and (val > 100_000 or val < 0):
                    _warn_rate_limited(f"report_interval_seconds out of sane bounds")
                else:
                    fval = float(val)
                    if math.isfinite(fval) and 1.0 <= fval <= 60.0:
                        validated["report_interval_seconds"] = fval
                    else:
                        _warn_rate_limited(f"report_interval_seconds {val!r} out of range [1.0, 60.0]")
            else:
                _warn_rate_limited(f"report_interval_seconds {val!r} invalid type; expected float/int")
        except (OverflowError, ValueError, TypeError):
            _warn_rate_limited("report_interval_seconds conversion overflow/error")

    if "event_capacity" in section:
        val = section["event_capacity"]
        try:
            if isinstance(val, int) and not isinstance(val, bool) and 16 <= val <= 16384:
                validated["event_capacity"] = val
            else:
                _warn_rate_limited(f"event_capacity {val!r} out of range [16, 16384]")
        except (OverflowError, ValueError, TypeError):
            _warn_rate_limited("event_capacity conversion overflow/error")

    if "samples_per_metric" in section:
        val = section["samples_per_metric"]
        try:
            if isinstance(val, int) and not isinstance(val, bool) and 8 <= val <= 4096:
                validated["samples_per_metric"] = val
            else:
                _warn_rate_limited(f"samples_per_metric {val!r} out of range [8, 4096]")
        except (OverflowError, ValueError, TypeError):
            _warn_rate_limited("samples_per_metric conversion overflow/error")

    if "trace_every_n" in section:
        val = section["trace_every_n"]
        try:
            if isinstance(val, int) and not isinstance(val, bool) and 1 <= val <= 10000:
                validated["trace_every_n"] = val
            else:
                _warn_rate_limited(f"trace_every_n {val!r} out of range [1, 10000]")
        except (OverflowError, ValueError, TypeError):
            _warn_rate_limited("trace_every_n conversion overflow/error")

    if "trace_max_bytes" in section:
        val = section["trace_max_bytes"]
        try:
            if isinstance(val, int) and not isinstance(val, bool) and 1048576 <= val <= 67108864:
                validated["trace_max_bytes"] = val
            else:
                _warn_rate_limited(f"trace_max_bytes {val!r} out of range [1048576, 67108864]")
        except (OverflowError, ValueError, TypeError):
            _warn_rate_limited("trace_max_bytes conversion overflow/error")

    if "trace_backup_count" in section:
        val = section["trace_backup_count"]
        try:
            if isinstance(val, int) and not isinstance(val, bool) and 1 <= val <= 3:
                validated["trace_backup_count"] = val
            else:
                _warn_rate_limited(f"trace_backup_count {val!r} out of range [1, 3]")
        except (OverflowError, ValueError, TypeError):
            _warn_rate_limited("trace_backup_count conversion overflow/error")

    return validated


def get_private_diagnostics_dir(custom_base: Optional[str] = None) -> Optional[Path]:
    """Ensure secure private directory for diagnostics (mode 0700, owned by current UID)."""
    try:
        if custom_base:
            diag_dir = Path(custom_base) / "npu-effects" / "diagnostics"
        else:
            runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
            if runtime_dir:
                diag_dir = Path(runtime_dir) / "npu-effects" / "diagnostics"
            else:
                diag_dir = Path(f"/tmp/npu-effects-{os.getuid()}") / "diagnostics"

        # Check existing path security
        if diag_dir.exists():
            if diag_dir.is_symlink():
                logger.error(f"Diagnostics path {diag_dir} is an unexpected symlink; disabling file trace")
                return None
            st = os.stat(diag_dir)
            if st.st_uid != os.getuid():
                logger.error(f"Diagnostics path {diag_dir} owned by UID {st.st_uid}, expected {os.getuid()}")
                return None
            try:
                os.chmod(diag_dir, 0o700)
            except OSError:
                pass
        else:
            diag_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                os.chmod(diag_dir, 0o700)
            except OSError:
                pass
        return diag_dir
    except Exception as e:
        logger.error(f"Failed to initialize diagnostics directory: {e}")
        return None


class DiagnosticsTraceWriter:
    """Rotational, opt-in JSONL trace file writer in secure runtime directory."""

    def __init__(self, stream: str, max_bytes: int = 10485760, backup_count: int = 2,
                 custom_dir: Optional[Path] = None):
        self.stream = stream
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.dir_path = custom_dir or get_private_diagnostics_dir()
        self.file_path = (self.dir_path / f"{stream}_trace.jsonl") if self.dir_path else None
        self._disabled = (self.file_path is None)
        self._last_warn_time = 0.0

    def _warn_writer(self, msg: str):
        now = time.monotonic()
        if now - self._last_warn_time >= 10.0:
            self._last_warn_time = now
            logger.warning(f"M0-TRACE: {msg}")

    def write_event(self, record: dict) -> bool:
        if self._disabled or not self.file_path:
            return False
        try:
            line = event_json(record) + "\n"
            line_bytes = line.encode("utf-8")

            # Validate path not tampered with
            if self.file_path.is_symlink():
                self._disabled = True
                self._warn_writer(f"Trace file {self.file_path} is a symlink; disabling writer")
                return False

            if self.file_path.exists():
                st = os.stat(self.file_path)
                if not stat.S_ISREG(st.st_mode):
                    self._disabled = True
                    self._warn_writer(f"Trace file {self.file_path} is not a regular file; disabling writer")
                    return False
                if st.st_uid != os.getuid():
                    self._disabled = True
                    self._warn_writer(f"Trace file {self.file_path} UID mismatch; disabling writer")
                    return False
                if st.st_size + len(line_bytes) > self.max_bytes:
                    self._rotate()

            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC

            fd = os.open(self.file_path, flags, 0o600)
            try:
                fst = os.fstat(fd)
                if not stat.S_ISREG(fst.st_mode) or fst.st_uid != os.getuid():
                    self._disabled = True
                    return False
                os.write(fd, line_bytes)
                return True
            finally:
                os.close(fd)
        except Exception as e:
            self._warn_writer(f"DiagnosticsTraceWriter write error: {e}")
            return False

    def _rotate(self):
        try:
            for i in range(self.backup_count, 0, -1):
                src = self.file_path if i == 1 else self.file_path.with_name(f"{self.file_path.name}.{i-1}")
                dst = self.file_path.with_name(f"{self.file_path.name}.{i}")
                if src.exists() and not src.is_symlink():
                    if dst.exists() and not dst.is_symlink():
                        try:
                            dst.unlink()
                        except OSError:
                            pass
                    os.replace(src, dst)
                    try:
                        os.chmod(dst, 0o600)
                    except OSError:
                        pass
        except Exception as e:
            self._warn_writer(f"Trace rotation failed: {e}")


class _ConsumerWorker:
    """Isolated background worker thread for telemetry draining, periodic summaries, and file tracing."""

    def __init__(
        self,
        stream: str,
        session_id: str,
        telemetry: Any,
        trace_writer: Optional[DiagnosticsTraceWriter],
        trace_enabled: bool,
        report_interval_seconds: float,
        clock_ns: Callable[[], int],
        owner_counters_provider: Callable[[], Dict[str, int]],
    ):
        self.stream = stream
        self.session_id = session_id
        self.telemetry = telemetry
        self.trace_writer = trace_writer
        self.trace_enabled = trace_enabled
        self.report_interval_seconds = report_interval_seconds
        self.clock_ns = clock_ns
        self.owner_counters_provider = owner_counters_provider
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self._last_report_mono_ns = clock_ns()
        self._last_owner_snapshots: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._pending_collectors: list = []
        self._dropped_events_total: int = 0

    @property
    def dropped_events_total(self) -> int:
        with self._lock:
            return self._dropped_events_total

    def update_config(
        self,
        telemetry: Any,
        trace_writer: Optional[DiagnosticsTraceWriter],
        trace_enabled: bool,
        report_interval_seconds: float,
    ):
        """Update telemetry collector and trace settings in-place without spawning new threads."""
        with self._lock:
            was_enabled = getattr(self.telemetry, "enabled", False)
            is_enabled = getattr(telemetry, "enabled", False)

            if telemetry is not self.telemetry:
                if getattr(self.telemetry, "events_queued", 0) > 0 or was_enabled:
                    self._pending_collectors.append(self.telemetry)
                    while len(self._pending_collectors) > 4:
                        evicted = self._pending_collectors.pop(0)
                        discarded = getattr(evicted, "events_queued", 0)
                        if discarded > 0:
                            self._dropped_events_total += discarded
                            logger.warning(
                                f"M0-PENDING-DROP: stream={self.stream} evicted pending collector with {discarded} events"
                            )

            self.telemetry = telemetry
            self.trace_writer = trace_writer
            self.trace_enabled = trace_enabled
            self.report_interval_seconds = report_interval_seconds

            # Re-anchor reporting window and owner counters when transitioning from OFF to ON
            if not was_enabled and is_enabled:
                self._last_report_mono_ns = self.clock_ns()
                curr_owners = self.owner_counters_provider()
                self._last_owner_snapshots = dict(curr_owners)

    def start(self):
        self.thread = threading.Thread(
            target=self._run,
            name=f"npu-m0-{self.stream}",
            daemon=True,
        )
        self.thread.start()

    def _drain_batch(self):
        with self._lock:
            trace_enabled = self.trace_enabled
            trace_writer = self.trace_writer
            collectors = list(self._pending_collectors) + [self.telemetry]

        for col in collectors:
            capacity = getattr(col, "event_capacity", 2048)
            batch_limit = min(256, capacity) if capacity > 0 else 256
            try:
                records = col.drain(limit=batch_limit)
            except Exception as e:
                logger.warning(f"Error draining telemetry events: {e}")
                records = []
            if trace_enabled and trace_writer:
                for record in records:
                    try:
                        trace_writer.write_event(record)
                    except Exception:
                        pass

        with self._lock:
            self._pending_collectors = [
                c for c in self._pending_collectors if getattr(c, "events_queued", 0) > 0
            ]

    def _run(self):
        while not self.stop_event.is_set():
            self._drain_batch()
            now_ns = self.clock_ns()
            with self._lock:
                interval_sec = self.report_interval_seconds
                is_enabled = getattr(self.telemetry, "enabled", False)
                pending_count = len(self._pending_collectors)

            if is_enabled:
                report_interval_ns = int(interval_sec * 1_000_000_000)
                if now_ns - self._last_report_mono_ns >= report_interval_ns:
                    self._generate_summary(now_ns)
                self.stop_event.wait(0.05)
            else:
                # Parked state when disabled: sleep longer if no pending collectors
                if pending_count == 0:
                    self.stop_event.wait(0.2)
                else:
                    self.stop_event.wait(0.05)

        # On graceful stop, drain remaining events if trace writer is active
        with self._lock:
            trace_enabled = self.trace_enabled
            trace_writer = self.trace_writer
        if trace_enabled and trace_writer:
            self._drain_batch()

        # Account for any events still queued when shutting down (truncation)
        with self._lock:
            remaining_dropped = sum(
                getattr(c, "events_queued", 0)
                for c in list(self._pending_collectors) + [self.telemetry]
            )
            if remaining_dropped > 0:
                self._dropped_events_total += remaining_dropped
                logger.warning(
                    f"M0-SHUTDOWN-DROP: stream={self.stream} {remaining_dropped} events truncated on shutdown"
                )

    def _generate_summary(self, now_ns: int):
        with self._lock:
            cur_telemetry = self.telemetry
            if not getattr(cur_telemetry, "enabled", False):
                return
            elapsed_ns = now_ns - self._last_report_mono_ns
            self._last_report_mono_ns = now_ns
            window_sec = max(0.001, elapsed_ns / 1_000_000_000)
            pending_dropped = self._dropped_events_total
            pending_queued = sum(
                getattr(c, "events_queued", 0) for c in self._pending_collectors
            )

        rep = cur_telemetry.report()
        curr_owners = self.owner_counters_provider()
        owner_deltas = {}
        for name, curr in curr_owners.items():
            prev = self._last_owner_snapshots.get(name, curr)
            delta = max(0, curr - prev)
            self._last_owner_snapshots[name] = curr
            owner_deltas[name] = {
                "total": curr,
                "delta": delta,
                "rate_per_sec": round(delta / window_sec, 2),
            }

        summary = {
            "schema_version": SCHEMA_VERSION,
            "type": "summary",
            "stream": self.stream,
            "session_id": self.session_id,
            "window_duration_sec": round(window_sec, 3),
            "owner_counters": owner_deltas,
            "metrics": rep.get("metrics", {}),
            "events_queue_dropped": rep.get("events_queue_dropped", 0) + pending_dropped,
            "events_queued": rep.get("events_queued", 0) + pending_queued,
        }
        logger.info(f"M0-SUMMARY: {event_json(summary)}")


class TelemetryRuntime:
    """Manages M0 telemetry lifecycle, consumer thread, periodic summaries, and tracing."""

    def __init__(
        self,
        stream: str,
        session_id: Optional[str] = None,
        config: Optional[dict] = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        custom_diag_dir: Optional[Path] = None,
    ):
        if stream not in ("video", "audio"):
            raise ValueError("stream must be video or audio")
        self.stream = stream
        self.session_id = session_id or str(uuid.uuid4())
        self.clock_ns = clock_ns
        self.custom_diag_dir = custom_diag_dir

        self.config = validate_m0_config(config)
        self.enabled = self.config["enabled"]
        self.trace_enabled = self.config["trace_enabled"]
        self.trace_every_n = self.config["trace_every_n"]

        self._lock = threading.RLock()
        self._worker: Optional[_ConsumerWorker] = None
        self._trace_writer: Optional[DiagnosticsTraceWriter] = None
        self._owner_counters: Dict[str, Callable[[], int]] = {}

        if self.enabled:
            self._start_worker_locked()
        else:
            self.telemetry = NullTelemetry()

    @property
    def _running(self) -> bool:
        with self._lock:
            return self._worker is not None and self._worker.thread is not None and self._worker.thread.is_alive()

    def _get_owner_snapshot(self) -> Dict[str, int]:
        with self._lock:
            res = {}
            for name, getter in self._owner_counters.items():
                try:
                    res[name] = int(getter())
                except Exception:
                    res[name] = 0
            return res

    def _start_worker_locked(self):
        self.telemetry = BoundedTelemetry(
            stream=self.stream,
            session_id=self.session_id,
            event_capacity=self.config["event_capacity"],
            samples_per_metric=self.config["samples_per_metric"],
            clock_ns=self.clock_ns,
        )
        if self.trace_enabled:
            self._trace_writer = DiagnosticsTraceWriter(
                stream=self.stream,
                max_bytes=self.config["trace_max_bytes"],
                backup_count=self.config["trace_backup_count"],
                custom_dir=self.custom_diag_dir,
            )
        else:
            self._trace_writer = None

        self._worker = _ConsumerWorker(
            stream=self.stream,
            session_id=self.session_id,
            telemetry=self.telemetry,
            trace_writer=self._trace_writer,
            trace_enabled=self.trace_enabled,
            report_interval_seconds=self.config["report_interval_seconds"],
            clock_ns=self.clock_ns,
            owner_counters_provider=self._get_owner_snapshot,
        )
        self._worker.start()

    def register_owner_counter(self, name: str, getter: Callable[[], int]):
        """Register an authoritative owner counter for interval rate and delta tracking."""
        with self._lock:
            self._owner_counters[name] = getter

    def update_config(self, raw_cfg: Optional[Dict[str, Any]]):
        """Hot reload M0 telemetry configuration without restarting media pipelines.

        Does not perform synchronous trace writing on the caller thread.
        Never spawns multiple consumer threads or writers for the same stream.
        """
        new_config = validate_m0_config(raw_cfg)
        with self._lock:
            was_enabled = self.enabled
            is_enabled = new_config["enabled"]
            self.config = new_config
            self.enabled = is_enabled
            self.trace_enabled = new_config["trace_enabled"]
            self.trace_every_n = new_config["trace_every_n"]

            if was_enabled and not is_enabled:
                self.telemetry = NullTelemetry()
                if self._worker:
                    self._worker.update_config(
                        telemetry=self.telemetry,
                        trace_writer=None,
                        trace_enabled=False,
                        report_interval_seconds=new_config["report_interval_seconds"],
                    )
            elif not was_enabled and is_enabled:
                self.telemetry = BoundedTelemetry(
                    stream=self.stream,
                    session_id=self.session_id,
                    event_capacity=new_config["event_capacity"],
                    samples_per_metric=new_config["samples_per_metric"],
                    clock_ns=self.clock_ns,
                )
                if self.trace_enabled and self._trace_writer is None:
                    self._trace_writer = DiagnosticsTraceWriter(
                        stream=self.stream,
                        max_bytes=new_config["trace_max_bytes"],
                        backup_count=new_config["trace_backup_count"],
                        custom_dir=self.custom_diag_dir,
                    )
                if self._worker is None or not self._worker.thread or not self._worker.thread.is_alive():
                    self._start_worker_locked()
                else:
                    self._worker.update_config(
                        telemetry=self.telemetry,
                        trace_writer=self._trace_writer if self.trace_enabled else None,
                        trace_enabled=self.trace_enabled,
                        report_interval_seconds=new_config["report_interval_seconds"],
                    )
            elif was_enabled and is_enabled:
                capacity_changed = (
                    getattr(self.telemetry, "event_capacity", 2048) != new_config["event_capacity"]
                    or getattr(self.telemetry, "samples_per_metric", 512) != new_config["samples_per_metric"]
                )
                if capacity_changed:
                    self.telemetry = BoundedTelemetry(
                        stream=self.stream,
                        session_id=self.session_id,
                        event_capacity=new_config["event_capacity"],
                        samples_per_metric=new_config["samples_per_metric"],
                        clock_ns=self.clock_ns,
                    )
                if self.trace_enabled:
                    if self._trace_writer is None:
                        self._trace_writer = DiagnosticsTraceWriter(
                            stream=self.stream,
                            max_bytes=new_config["trace_max_bytes"],
                            backup_count=new_config["trace_backup_count"],
                            custom_dir=self.custom_diag_dir,
                        )
                    else:
                        self._trace_writer.max_bytes = new_config["trace_max_bytes"]
                        self._trace_writer.backup_count = new_config["trace_backup_count"]
                else:
                    self._trace_writer = None

                if self._worker is None or not self._worker.thread or not self._worker.thread.is_alive():
                    self._start_worker_locked()
                else:
                    self._worker.update_config(
                        telemetry=self.telemetry,
                        trace_writer=self._trace_writer if self.trace_enabled else None,
                        trace_enabled=self.trace_enabled,
                        report_interval_seconds=new_config["report_interval_seconds"],
                    )

    def should_trace(self, frame_or_block_count: int, is_anomaly: bool = False) -> bool:
        """Determines if an event should be traced based on sampling policy.

        Anomalies are always emitted when telemetry is enabled.
        """
        if not self.enabled:
            return False
        if is_anomaly:
            return True
        if not self.trace_enabled:
            return False
        return (frame_or_block_count % self.trace_every_n) == 0

    def _drain_to_trace(self, limit: Optional[int] = None):
        """Drain batch of events. Always drains collector; writes to file if trace is active."""
        capacity = getattr(self.telemetry, "event_capacity", 2048)
        batch_limit = min(256 if limit is None else limit, capacity)
        try:
            records = self.telemetry.drain(limit=batch_limit)
        except Exception as e:
            logger.warning(f"Error draining telemetry events: {e}")
            return
        if self._trace_writer and self.trace_enabled:
            for record in records:
                try:
                    self._trace_writer.write_event(record)
                except Exception:
                    pass

    def stop(self, timeout: float = 1.0):
        """Clean shutdown of diagnostic thread and files with bounded timeout.

        Never performs synchronous trace writing on the caller thread.
        """
        worker = None
        with self._lock:
            self.enabled = False
            self.telemetry = NullTelemetry()
            worker = self._worker
            self._worker = None

        if worker:
            worker.stop_event.set()
            if worker.thread and worker.thread.is_alive():
                worker.thread.join(timeout=timeout)
