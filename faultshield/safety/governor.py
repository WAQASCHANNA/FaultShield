"""
Independent Safety Governor.
Operates as the last line of defense in the control hierarchy:
- Enforces Time-To-Live (TTL) on Nemotron recovery decisions (rejects stale decisions)
- Validates actions against an explicit safety allowlist
- Hard-clamps velocities and prevents moves into critical obstacle envelopes
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from faultshield.config import (
    GOVERNOR_MAX_DECISION_TTL_SEC,
    GOVERNOR_CRITICAL_STOP_DISTANCE,
    GOVERNOR_RECOVERY_SPEED_LIMIT,
    PERMITTED_RECOVERY_ACTIONS,
)
from faultshield.agent.nemotron import NemotronDecision
from faultshield.core.sensors import TelemetryPacket


@dataclass
class GovernorDecision:
    status: str  # "ACCEPTED", "REJECTED_STALE", "REJECTED_DISALLOWED", "CRITICAL_OVERRIDE", "NORMAL_PASS"
    executed_v: float
    executed_omega: float
    is_safe: bool
    stale_age_sec: float
    reason: str


class SafetyGovernor:
    def __init__(
        self,
        max_ttl_sec: float = GOVERNOR_MAX_DECISION_TTL_SEC,
        critical_stop_dist: float = GOVERNOR_CRITICAL_STOP_DISTANCE,
        recovery_speed_limit: float = GOVERNOR_RECOVERY_SPEED_LIMIT,
    ):
        self.max_ttl_sec = max_ttl_sec
        self.critical_stop_dist = critical_stop_dist
        self.recovery_speed_limit = recovery_speed_limit

        self.total_evaluations = 0
        self.stale_rejections_count = 0
        self.disallowed_rejections_count = 0
        self.emergency_stops_count = 0

    def reset(self):
        self.total_evaluations = 0
        self.stale_rejections_count = 0
        self.disallowed_rejections_count = 0
        self.emergency_stops_count = 0

    def validate_and_govern(
        self,
        command_v: float,
        command_omega: float,
        telemetry: TelemetryPacket,
        current_sim_time: float,
        ai_decision: Optional[NemotronDecision] = None,
    ) -> GovernorDecision:
        """
        Validate proposed motion command or AI recovery recommendation.
        Enforces strict fail-safe rules and TTL expiration.
        """
        self.total_evaluations += 1

        # Check physical obstacle proximity from telemetry
        min_observed_dist = telemetry.lidar.min_distance

        # 1. Evaluate AI Decision if one is provided for execution
        if ai_decision is not None:
            age = current_sim_time - ai_decision.dispatch_timestamp

            # Rule A: Check Staleness / TTL Expiration
            if age > self.max_ttl_sec:
                self.stale_rejections_count += 1
                return GovernorDecision(
                    status="REJECTED_STALE",
                    executed_v=0.0,
                    executed_omega=0.0,
                    is_safe=False,
                    stale_age_sec=age,
                    reason=f"Nemotron decision rejected: age ({age:.3f}s) exceeded safety TTL ({self.max_ttl_sec}s). Executed immediate safe halt.",
                )

            # Rule B: Allowlist Validation
            if ai_decision.recommended_action not in PERMITTED_RECOVERY_ACTIONS:
                self.disallowed_rejections_count += 1
                return GovernorDecision(
                    status="REJECTED_DISALLOWED",
                    executed_v=0.0,
                    executed_omega=0.0,
                    is_safe=False,
                    stale_age_sec=age,
                    reason=f"Action '{ai_decision.recommended_action}' not on permitted safety allowlist. Defaulted to safe stop.",
                )

            # Rule C: Action approved - Clamp speeds to recovery envelope
            v_clamped = max(-self.recovery_speed_limit, min(ai_decision.target_linear_velocity, self.recovery_speed_limit))
            omega_clamped = max(-1.0, min(ai_decision.target_angular_velocity, 1.0))

            # Critical obstacle proximity guard
            if min_observed_dist < self.critical_stop_dist and v_clamped > 0.0:
                v_clamped = 0.0  # Disallow forward motion towards critical obstacle

            return GovernorDecision(
                status="ACCEPTED",
                executed_v=v_clamped,
                executed_omega=omega_clamped,
                is_safe=True,
                stale_age_sec=age,
                reason=f"Nemotron action '{ai_decision.recommended_action}' validated within safety envelope (age: {age:.3f}s).",
            )

        # 2. Standard Pass-through for normal navigation commands
        # Hardware proximity override check
        if min_observed_dist < self.critical_stop_dist and command_v > 0.0:
            self.emergency_stops_count += 1
            return GovernorDecision(
                status="CRITICAL_OVERRIDE",
                executed_v=0.0,
                executed_omega=command_omega * 0.5,
                is_safe=True,
                stale_age_sec=0.0,
                reason=f"Obstacle at {min_observed_dist:.2f}m triggered hard proximity override. Clamped forward velocity.",
            )

        return GovernorDecision(
            status="NORMAL_PASS",
            executed_v=command_v,
            executed_omega=command_omega,
            is_safe=True,
            stale_age_sec=0.0,
            reason="Standard navigation command passed validation.",
        )
