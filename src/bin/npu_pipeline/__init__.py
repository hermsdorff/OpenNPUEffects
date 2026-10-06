"""OpenNPUEffects Pipeline Infrastructure and Contracts (M0 & M1)."""

from .contracts import (
    FrameIdentity,
    StageResultStatus,
    StageResultValidity,
    StageResult,
    FrameContext,
    OutputFrame,
    M1Config,
)
from .lifecycle import LifecycleManager
from .primary import CoherentPrimaryCoordinator
from .telemetry import BoundedTelemetry, ObservationOrigin, NullTelemetry
from .telemetry_runtime import TelemetryRuntime

__all__ = [
    "FrameIdentity",
    "StageResultStatus",
    "StageResultValidity",
    "StageResult",
    "FrameContext",
    "OutputFrame",
    "M1Config",
    "LifecycleManager",
    "CoherentPrimaryCoordinator",
    "BoundedTelemetry",
    "ObservationOrigin",
    "NullTelemetry",
    "TelemetryRuntime",
]
