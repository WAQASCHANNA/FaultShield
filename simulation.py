"""Deterministic 2D robot simulation for FaultShield.

This is an educational simulation, not a safety-certified robotics controller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import math


@dataclass
class SimResult:
    label: str
    guarded: bool
    scenario: str
    status: str
    reached_goal: bool
    collision: bool
    safe_stop: bool
    steps: int
    distance_travelled: float
    path: list[dict[str, float]]
    events: list[str]
    accepted_ai_action: str
    rejected_ai_action: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ALLOWED_ACTIONS = {"BYPASS_LEFT", "BYPASS_RIGHT", "SAFE_STOP", "CONTINUE"}


def run_simulation(
    scenario: str = "Healthy sensors",
    guarded: bool = True,
    ai_action: str = "BYPASS_LEFT",
    max_steps: int = 180,
) -> SimResult:
    """Run one reproducible scenario.

    Scenarios:
      - Healthy sensors
      - Stale LiDAR
      - Stale LiDAR + delayed AI response

    In the combined scenario, model-response delay is simulated in the virtual
    timeline. It is not a measurement of actual Token Factory latency.
    """
    valid_scenarios = {
        "Healthy sensors",
        "Stale LiDAR",
        "Stale LiDAR + delayed AI response",
    }
    if scenario not in valid_scenarios:
        raise ValueError(f"Unknown scenario: {scenario}")

    action = ai_action.strip().upper()
    if action not in ALLOWED_ACTIONS:
        action = "BYPASS_LEFT"

    # World and robot parameters (metres, seconds).
    x, y = 0.0, 0.0
    goal_x = 10.0
    obstacle_x, obstacle_y = 5.0, 0.0
    obstacle_radius = 0.58
    robot_radius = 0.22
    dt = 0.1
    forward_step = 0.15
    lane_speed = 0.22
    detect_distance = 2.4
    stale_start_x = 2.25
    stale_threshold_s = 0.25
    response_delay_steps = 8

    path: list[dict[str, float]] = [{"x": x, "y": y, "time": 0.0}]
    events: list[str] = []
    collision = False
    reached_goal = False
    safe_stop = False
    bypass_direction = 0  # +1 = left/up, -1 = right/down
    bypass_triggered = False
    frozen_range: float | None = None
    freeze_step: int | None = None
    accepted_ai_action = "NOT_USED"
    rejected_ai_action: str | None = None
    distance_travelled = 0.0
    status = "Step limit reached"

    is_stale = scenario != "Healthy sensors"
    has_delayed_response = scenario == "Stale LiDAR + delayed AI response"

    if not guarded:
        events.append("Baseline controller: no sensor-integrity monitor or AI-action governor.")
    else:
        events.append("FaultShield enabled: sensor freshness monitor and constrained recovery policy active.")

    for step in range(1, max_steps + 1):
        now = step * dt
        previous_x, previous_y = x, y

        # Inject a frozen range stream after the robot starts approaching the obstacle.
        true_range = max(0.0, obstacle_x - x) if x < obstacle_x else 9.9
        if abs(y - obstacle_y) > 1.25:
            true_range = 9.9  # obstacle is outside the forward corridor

        if is_stale and x >= stale_start_x:
            if frozen_range is None:
                frozen_range = true_range
                freeze_step = step
                events.append(f"t={now:.1f}s: LiDAR reading froze at {frozen_range:.2f} m.")
            measured_range = frozen_range
            age_steps = max(0, step - (freeze_step or step))
            lidar_age_s = age_steps * dt
        else:
            measured_range = true_range
            lidar_age_s = 0.0

        # Conventional obstacle response works when readings are fresh.
        if not bypass_triggered and measured_range <= detect_distance:
            bypass_triggered = True
            bypass_direction = 1
            events.append(f"t={now:.1f}s: obstacle detected at {measured_range:.2f} m; route detour started.")

        # FaultShield can detect a frozen sensor using its timestamp/age.
        if guarded and is_stale and lidar_age_s > stale_threshold_s and not bypass_triggered:
            events.append(f"t={now:.1f}s: stale LiDAR detected (age={lidar_age_s:.1f}s); local fallback engaged.")
            if has_delayed_response:
                events.append(
                    f"t={now:.1f}s: remote AI recommendation is simulated as delayed by {response_delay_steps * dt:.1f}s; robot will not wait for it."
                )
                # The local controller acts immediately; the simulated late action is checked later.
                accepted_ai_action = "LOCAL_SAFE_FALLBACK"
                bypass_triggered = True
                bypass_direction = 1
            else:
                proposed = action
                if proposed == "CONTINUE":
                    rejected_ai_action = proposed
                    events.append("Safety governor rejected CONTINUE because sensor freshness is untrusted.")
                    proposed = "BYPASS_LEFT"
                if proposed == "SAFE_STOP":
                    accepted_ai_action = proposed
                    safe_stop = True
                    status = "Stopped safely after sensor fault"
                    events.append("Safety governor accepted SAFE_STOP; robot entered a safe stopped state.")
                    break
                if proposed == "BYPASS_RIGHT":
                    accepted_ai_action = proposed
                    bypass_direction = -1
                else:
                    accepted_ai_action = "BYPASS_LEFT"
                    bypass_direction = 1
                bypass_triggered = True
                events.append(f"Safety governor approved {accepted_ai_action}; detour started.")

        # For a stale stream with no guard, stale value remains too large to trigger a detour.
        # For healthy streams, both controllers use the normal obstacle response above.
        target_y = 0.0
        if bypass_triggered and 2.7 <= x <= 7.2:
            target_y = 2.0 * (bypass_direction if bypass_direction != 0 else 1)
        elif bypass_triggered and x < 2.7:
            target_y = 0.0

        if not safe_stop:
            x = min(goal_x, x + forward_step)
            delta_y = target_y - y
            if abs(delta_y) <= lane_speed:
                y = target_y
            else:
                y += math.copysign(lane_speed, delta_y)

        distance_travelled += math.hypot(x - previous_x, y - previous_y)
        path.append({"x": round(x, 3), "y": round(y, 3), "time": round(now, 3)})

        # Collision check against ground truth, never against corrupted sensor data.
        centre_distance = math.hypot(x - obstacle_x, y - obstacle_y)
        if centre_distance <= obstacle_radius + robot_radius:
            collision = True
            status = "Collision"
            events.append(f"t={now:.1f}s: collision in simulator ground truth.")
            break

        # Simulated old remote command arrives late; the governor rejects it if the
        # safety action already began. This delay is an experiment parameter, not live API latency.
        if has_delayed_response and guarded and bypass_triggered and freeze_step is not None:
            if step == freeze_step + response_delay_steps:
                rejected_ai_action = action
                events.append(
                    f"t={now:.1f}s: delayed remote action {action} arrived; discarded because the local safety action is already active."
                )

        if x >= goal_x:
            reached_goal = True
            status = "Goal reached safely"
            events.append(f"t={now:.1f}s: goal reached.")
            break

    if collision:
        status = "Collision"
    elif safe_stop:
        status = "Stopped safely after sensor fault"
    elif reached_goal:
        status = "Goal reached safely"

    return SimResult(
        label="FaultShield" if guarded else "Baseline",
        guarded=guarded,
        scenario=scenario,
        status=status,
        reached_goal=reached_goal,
        collision=collision,
        safe_stop=safe_stop,
        steps=len(path) - 1,
        distance_travelled=round(distance_travelled, 3),
        path=path,
        events=events,
        accepted_ai_action=accepted_ai_action,
        rejected_ai_action=rejected_ai_action,
    )

