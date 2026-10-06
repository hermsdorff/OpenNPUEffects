"""Contracts and ownership models for M1 coherent primary pipeline (AV-M1-01).

Implements:
- FrameIdentity: exact compatibility key across pipeline stages
- StageResult: bounded execution result with ownership isolation
- FrameContext: frozen per-frame execution context
- OutputFrame: published output image with origin and omitted metadata
- M1Config: validated operational budget configuration
"""
from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class FrameIdentity:
    """Exact identity and compatibility key for M1 coherent pipeline.

    Required compatibility fields:
    - session_id: identifies daemon run
    - generation_id: camera/stream continuity generation
    - frame_id: monotonic source capture sequence
    - processing_config_epoch: incremented on effective pixel/policy changes
    - model_id / model_version: neural model identity
    - geometry_id / transform_id: frozen capture, framing crop, and output dimensions
    - seq: optional submission sequence; NOT an exact match substitute for frame_id
    """
    session_id: str
    generation_id: str
    frame_id: int
    processing_config_epoch: int
    model_id: str = "multiclass"
    model_version: str = "1.0"
    geometry_id: str = "1920x1080->1920x1080"
    seq: Optional[int] = None
    arrival_mono_ns: Optional[int] = None

    def is_compatible_with(self, other: "FrameIdentity") -> bool:
        """Exact equality across all required identity fields.

        seq >= other.seq or inequality in any field is NOT exact compatibility.
        """
        if not isinstance(other, FrameIdentity):
            return False
        return (
            self.session_id == other.session_id
            and self.generation_id == other.generation_id
            and self.frame_id == other.frame_id
            and self.processing_config_epoch == other.processing_config_epoch
            and self.model_id == other.model_id
            and self.model_version == other.model_version
            and self.geometry_id == other.geometry_id
        )

    def is_generation_compatible(self, other: "FrameIdentity") -> bool:
        """Checks stream and generation identity only."""
        if not isinstance(other, FrameIdentity):
            return False
        return (
            self.session_id == other.session_id
            and self.generation_id == other.generation_id
        )


import copy


class StatusStr(str):
    """String subclass that exposes .name for compatibility with enum consumers."""
    @property
    def name(self) -> str:
        return str(self)


class FrozenList(list):
    """Immutable list that silences item mutations to protect nested collections."""
    def __setitem__(self, index: Any, value: Any) -> None:
        return

    def __delitem__(self, index: Any) -> None:
        return

    def append(self, item: Any) -> None:
        return

    def extend(self, iterable: Any) -> None:
        return

    def insert(self, index: int, item: Any) -> None:
        return

    def pop(self, index: int = -1) -> Any:
        return None

    def remove(self, item: Any) -> None:
        return

    def clear(self) -> None:
        return

    def __iadd__(self, other: Any) -> "FrozenList":
        return self

    def __imul__(self, other: Any) -> "FrozenList":
        return self

    def reverse(self) -> None:
        return

    def sort(self, *args: Any, **kwargs: Any) -> None:
        return


class FrozenDict(dict):
    """Immutable dictionary that silences item mutations to prevent side effects."""
    def __setitem__(self, key: Any, value: Any) -> None:
        return

    def __delitem__(self, key: Any) -> None:
        return

    def update(self, *args: Any, **kwargs: Any) -> None:
        return

    def clear(self) -> None:
        return

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def popitem(self) -> Any:
        return None

    def setdefault(self, key: Any, default: Any = None) -> Any:
        return self.get(key, default)

    def __ior__(self, other: Any) -> "FrozenDict":
        return self


def freeze_value(val: Any) -> Any:
    """Recursively freezes collections and arrays for immutable contracts."""
    if isinstance(val, dict):
        return FrozenDict({k: freeze_value(v) for k, v in val.items()})
    elif isinstance(val, list):
        return FrozenList([freeze_value(x) for x in val])
    elif isinstance(val, tuple):
        return tuple(freeze_value(x) for x in val)
    elif isinstance(val, set):
        return frozenset(freeze_value(x) for x in val)
    elif isinstance(val, np.ndarray):
        arr = np.array(val, copy=True)
        arr.flags.writeable = False
        return arr
    else:
        return copy.deepcopy(val)


class StageResultStatus:
    OK = StatusStr("OK")
    SKIPPED = StatusStr("SKIPPED")
    TIMEOUT = StatusStr("TIMEOUT")
    ERROR = StatusStr("ERROR")
    CANCELLED = StatusStr("CANCELLED")
    DROPPED = StatusStr("DROPPED")


class StageResultValidity:
    EXACT = "EXACT"
    INVALID = "INVALID"
    LATE = "LATE"
    MISMATCH = "MISMATCH"
    UNKNOWN = "UNKNOWN"


@dataclass
class StageResult:
    """Result of a pipeline inference stage with defined ownership.

    Published masks in payload are protected against in-place mutations by the
    compositor or producer through isolated private copies.
    """
    identity: FrameIdentity
    status: Any = StageResultStatus.OK
    validity: str = StageResultValidity.EXACT
    payload: Dict[str, Any] = field(default_factory=dict)
    t_submit_mono_ns: int = 0
    t_complete_mono_ns: int = 0
    error_message: Optional[str] = None
    drop_reason: Optional[str] = None

    def __post_init__(self):
        if self.drop_reason and not self.error_message:
            object.__setattr__(self, "error_message", self.drop_reason)
        elif self.error_message and not self.drop_reason:
            object.__setattr__(self, "drop_reason", self.error_message)
        if isinstance(self.status, str):
            object.__setattr__(self, "status", StatusStr(self.status))
        object.__setattr__(self, "payload", freeze_value(self.payload or {}))
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: Any):
        if getattr(self, "_initialized", False):
            return
        super().__setattr__(name, value)

    def get_mask(self, key: str) -> Optional[np.ndarray]:
        """Returns a private working copy of the requested mask."""
        val = self.payload.get(key)
        if val is None:
            return None
        if isinstance(val, np.ndarray):
            return val.copy()
        return val

    @property
    def has_hands(self) -> bool:
        return bool(self.payload.get("has_hands", False))


@dataclass
class FrameContext:
    """Frozen per-frame processing context.

    Captures monotonic arrival, absolute deadlines, effective geometry,
    and framed pixels with isolated ownership. Recalculation using latest
    global state during callbacks is prohibited.
    """
    arrival_mono_ns: int
    frame_deadline_mono_ns: int
    primary_decision_deadline_mono_ns: int
    identity: FrameIdentity
    frozen_cfg: Dict[str, Any]
    in_w: int
    in_h: int
    out_w: int
    out_h: int
    crop_box: Tuple[float, float, float, float]
    framed_pixels: np.ndarray
    pts_ns: Optional[int] = None
    timestamp_quality: str = "ARRIVAL_ESTIMATE"
    quality_flags: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.framed_pixels, np.ndarray):
            # Unconditional copy to guarantee exclusive ownership against producer view mutations
            copied = np.array(self.framed_pixels, copy=True)
            copied.flags.writeable = False
            object.__setattr__(self, "framed_pixels", copied)
        object.__setattr__(self, "frozen_cfg", freeze_value(self.frozen_cfg or {}))
        object.__setattr__(self, "quality_flags", freeze_value(self.quality_flags or {}))
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: Any):
        if getattr(self, "_initialized", False):
            return
        super().__setattr__(name, value)


@dataclass
class OutputFrame:
    """Complete published BGR image with origin identity and metadata.

    presentation_timestamp_ns must be None before M5.
    """
    image: np.ndarray
    identity: FrameIdentity
    commit_mono_ns: int
    applied_results: List[str] = field(default_factory=list)
    omitted_results: List[Tuple[str, str]] = field(default_factory=list)
    quality_flags: Dict[str, Any] = field(default_factory=dict)
    output_type: str = "LIVE"
    presentation_timestamp_ns: Optional[int] = None
    source_frame_id: Optional[int] = None
    source_generation_id: Optional[str] = None
    attempt_frame_id: Optional[int] = None
    arrival_mono_ns: Optional[int] = None

    def __post_init__(self):
        if self.presentation_timestamp_ns is not None:
            raise ValueError("presentation_timestamp_ns must be None before M5")
        if isinstance(self.image, np.ndarray):
            copied = np.array(self.image, copy=True)
            copied.flags.writeable = False
            object.__setattr__(self, "image", copied)
        if self.identity is not None:
            if self.source_frame_id is None:
                object.__setattr__(self, "source_frame_id", getattr(self.identity, "frame_id", None))
            if self.source_generation_id is None:
                object.__setattr__(self, "source_generation_id", getattr(self.identity, "generation_id", None))
            if self.arrival_mono_ns is None:
                object.__setattr__(self, "arrival_mono_ns", getattr(self.identity, "arrival_mono_ns", None))
        object.__setattr__(self, "quality_flags", freeze_value(self.quality_flags or {}))
        object.__setattr__(self, "applied_results", freeze_value(self.applied_results or []))
        object.__setattr__(self, "omitted_results", freeze_value(self.omitted_results or []))
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: Any):
        if getattr(self, "_initialized", False):
            return
        super().__setattr__(name, value)


@dataclass(frozen=True)
class M1Config:
    """Validated M1 operational configuration.

    Strict numeric validation:
    - Only finite real numbers (no booleans, no nan/inf)
    - frame_deadline_ms in [50.0, 1000.0]
    - compose_reserve_ms in [1.0, 500.0]
    - safety_margin_ms in [0.0, 100.0]
    - compose_reserve_ms + safety_margin_ms < frame_deadline_ms
    """
    frame_deadline_ms: float = 250.0
    compose_reserve_ms: float = 90.0
    safety_margin_ms: float = 10.0

    def __post_init__(self):
        def _val_num(name: str, val: Any, min_v: float, max_v: float) -> float:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"m1.{name} must be a finite number, got {val!r}")
            vf = float(val)
            if not (min_v <= vf <= max_v):
                raise ValueError(f"m1.{name} must be in [{min_v}, {max_v}], got {vf}")
            return vf

        fd = _val_num("frame_deadline_ms", self.frame_deadline_ms, 50.0, 1000.0)
        cr = _val_num("compose_reserve_ms", self.compose_reserve_ms, 1.0, 500.0)
        sm = _val_num("safety_margin_ms", self.safety_margin_ms, 0.0, 100.0)

        if cr + sm >= fd:
            raise ValueError(
                f"compose_reserve_ms ({cr}) + safety_margin_ms ({sm}) = {cr + sm} "
                f"must be strictly less than frame_deadline_ms ({fd})"
            )
        object.__setattr__(self, "frame_deadline_ms", fd)
        object.__setattr__(self, "compose_reserve_ms", cr)
        object.__setattr__(self, "safety_margin_ms", sm)

    @classmethod
    def from_dict(cls, data: Any) -> "M1Config":
        if not isinstance(data, dict):
            raise ValueError(f"m1 config must be a dict, got {type(data).__name__}")
        fd = data.get("frame_deadline_ms", 250.0)
        cr = data.get("compose_reserve_ms", 90.0)
        sm = data.get("safety_margin_ms", 10.0)
        return cls(frame_deadline_ms=fd, compose_reserve_ms=cr, safety_margin_ms=sm)
