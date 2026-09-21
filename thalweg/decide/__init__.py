"""Safe bound to allocation decisions and congestion alerts. No ML here."""

from thalweg.decide.admission import AdmissionResult, admit_sessions, evaluate_admission
from thalweg.decide.congestion import CongestionResult, detect_congestion
from thalweg.decide.flows import (
    CriticalityScorer,
    FlowClass,
    FlowObservation,
    FlowSpec,
    ScorerConfig,
)
from thalweg.decide.protect import (
    AllocationResult,
    FlowDemand,
    ProtectionConfig,
    ProtectionController,
    allocate_protected,
)

__all__ = [
    "AdmissionResult", "AllocationResult", "CongestionResult", "CriticalityScorer",
    "FlowClass", "FlowDemand", "FlowObservation", "FlowSpec", "ProtectionConfig",
    "ProtectionController", "ScorerConfig",
    "admit_sessions", "allocate_protected", "detect_congestion", "evaluate_admission",
]
