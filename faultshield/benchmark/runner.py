"""
FaultShield Simulation and Benchmark Runner.
Executes episodes across Variants A, B, C, D and captures complete trajectory traces.
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from faultshield.config import (
    SIMULATION_DT,
    MAX_SIMULATION_STEPS,
    ROBOT_RADIUS,
    DEFAULT_START_POS,
    DEFAULT_GOAL_POS,
)
from faultshield.core.world import WarehouseMap
from faultshield.core.robot import DifferentialDriveRobot
from faultshield.core.sensors import SensorSuite, TelemetryPacket
from faultshield.core.controller import NavigationController
from faultshield.faults.injector import FaultInjector, FaultType
from faultshield.faults.scenarios import Scenario, BENCHMARK_SCENARIOS
from faultshield.integrity.monitor import SensorIntegrityMonitor, IncidentReport
from faultshield.agent.nemotron import NemotronRecoveryAgent, NemotronDecision
from faultshield.safety.governor import SafetyGovernor, GovernorDecision
from faultshield.benchmark.metrics import EpisodeMetrics


@dataclass
class StepTrace:
    t: float
    x_gt: float
    y_gt: float
    theta_gt: float
    x_odom: float
    y_odom: float
    theta_odom: float
    v: float
    omega: float
    min_clearance: float
    fault_active: bool
    incident_active: bool
    ai_action: Optional[str]
    governor_status: str
    is_collided: bool
    lidar_ranges: Optional[List[float]] = None
    lidar_angles: Optional[List[float]] = None


@dataclass
class EpisodeRunResult:
    scenario_id: str
    variant_id: str
    metrics: EpisodeMetrics
    trace: List[StepTrace]
    incidents: List[IncidentReport]
    ai_decisions: List[NemotronDecision]
    governor_log: List[GovernorDecision]


class BenchmarkRunner:
    def __init__(
        self,
        nemotron_agent: Optional[NemotronRecoveryAgent] = None,
    ):
        self.nemotron_agent = nemotron_agent or NemotronRecoveryAgent(use_mock=True)

    def run_episode(
        self,
        scenario: Scenario,
        variant_id: str,  # "Variant_A", "Variant_B", "Variant_C", "Variant_D"
        seed: int = 42,
        max_steps: int = MAX_SIMULATION_STEPS,
    ) -> EpisodeRunResult:
        """Run single simulation episode under specified variant."""
        world = WarehouseMap()
        start_x, start_y, start_theta = scenario.start_pos
        goal_x, goal_y = scenario.goal_pos

        robot = DifferentialDriveRobot(start_x, start_y, start_theta, radius=ROBOT_RADIUS)
        sensor_suite = SensorSuite(seed=seed)
        sensor_suite.reset(start_x, start_y, start_theta)
        controller = NavigationController(goal_x, goal_y)

        # Fault injector configuration based on variant
        if variant_id == "Variant_A":
            # Normal baseline, no injected faults
            injector = FaultInjector(fault_type=FaultType.NONE)
        else:
            injector = scenario.create_injector()

        monitor = SensorIntegrityMonitor()
        governor = SafetyGovernor()

        traces: List[StepTrace] = []
        incidents: List[IncidentReport] = []
        ai_decisions: List[NemotronDecision] = []
        governor_decisions: List[GovernorDecision] = []

        # Tracking variables
        fault_first_time: Optional[float] = None
        fault_detection_time: Optional[float] = None
        safe_state_time: Optional[float] = None
        total_path_length = 0.0
        min_overall_clearance = 999.0
        stale_rejected_count = 0
        stale_accepted_count = 0
        unnecessary_stops = 0
        ai_latencies: List[float] = []

        # Recovery state machine
        active_ai_decision: Optional[NemotronDecision] = None
        pending_ai_decision: Optional[NemotronDecision] = None
        in_fallback_hold = False
        recalibrating_timer = 0.0

        for step in range(max_steps):
            current_time = step * SIMULATION_DT

            # 1. Sense: capture raw telemetry
            raw_telemetry = sensor_suite.generate_telemetry_packet(world, robot, SIMULATION_DT)

            # Ground truth fault status
            ground_truth_fault = injector.get_ground_truth_fault_status(current_time)
            if ground_truth_fault.is_active and fault_first_time is None:
                fault_first_time = current_time

            # 2. Inject Faults into telemetry (if active)
            corrupted_telemetry = injector.inject(raw_telemetry, SIMULATION_DT)

            # 3. Check Sensor Integrity
            detected_incident: Optional[IncidentReport] = None
            if variant_id in ("Variant_C", "Variant_D"):
                detected_incident = monitor.inspect(corrupted_telemetry, current_time)
                if detected_incident is not None:
                    incidents.append(detected_incident)
                    if fault_detection_time is None:
                        fault_detection_time = current_time

            # 4. Handle Pending AI Responses arriving over network
            if pending_ai_decision is not None:
                if current_time >= pending_ai_decision.arrival_timestamp:
                    active_ai_decision = pending_ai_decision
                    pending_ai_decision = None

            # 5. Determine Command according to Variant
            cmd_v, cmd_omega = 0.0, 0.0
            current_ai_action_label = None

            # --- VARIANT A: Baseline Healthy ---
            if variant_id == "Variant_A":
                cmd_v, cmd_omega = controller.compute_control(corrupted_telemetry)
                gov_res = GovernorDecision(
                    status="NORMAL_PASS",
                    executed_v=cmd_v,
                    executed_omega=cmd_omega,
                    is_safe=True,
                    stale_age_sec=0.0,
                    reason="Variant A baseline execution.",
                )

            # --- VARIANT B: Unmitigated Corrupted Telemetry ---
            elif variant_id == "Variant_B":
                # Blindly follows corrupted telemetry without integrity monitor or governor
                cmd_v, cmd_omega = controller.compute_control(corrupted_telemetry)
                gov_res = GovernorDecision(
                    status="NORMAL_PASS",
                    executed_v=cmd_v,
                    executed_omega=cmd_omega,
                    is_safe=True,
                    stale_age_sec=0.0,
                    reason="Variant B unmitigated execution.",
                )

            # --- VARIANT C: Conventional Monitor + Fixed Fallback ---
            elif variant_id == "Variant_C":
                if detected_incident is not None or in_fallback_hold:
                    # Conventional safety rule: Freeze in place immediately upon any anomaly
                    in_fallback_hold = True
                    cmd_v, cmd_omega = 0.0, 0.0
                    gov_res = GovernorDecision(
                        status="CONVENTIONAL_SAFE_STOP",
                        executed_v=0.0,
                        executed_omega=0.0,
                        is_safe=True,
                        stale_age_sec=0.0,
                        reason="Conventional rule-based safe halt triggered by anomaly.",
                    )
                    if safe_state_time is None:
                        safe_state_time = current_time
                else:
                    cmd_v, cmd_omega = controller.compute_control(corrupted_telemetry)
                    gov_res = governor.validate_and_govern(
                        cmd_v, cmd_omega, corrupted_telemetry, current_time
                    )

            # --- VARIANT D: FaultShield (Monitor + Nemotron + Safety Governor) ---
            elif variant_id == "Variant_D":
                # If an incident was just detected and no request is pending/active, query Nemotron
                if detected_incident is not None and pending_ai_decision is None and active_ai_decision is None:
                    # Dispatch to Nemotron with scenario's simulated AI delay
                    decision = self.nemotron_agent.request_recovery(
                        detected_incident,
                        sim_time=current_time,
                        artificial_delay_sec=scenario.ai_delay_sec,
                    )
                    ai_decisions.append(decision)
                    ai_latencies.append(decision.latency_sec)
                    pending_ai_decision = decision

                # Handle active IMU recalibration timer
                if recalibrating_timer > 0.0:
                    recalibrating_timer -= SIMULATION_DT
                    cmd_v, cmd_omega = 0.0, 0.0
                    gov_res = GovernorDecision(
                        status="RECALIBRATING_IMU",
                        executed_v=0.0,
                        executed_omega=0.0,
                        is_safe=True,
                        stale_age_sec=0.0,
                        reason="Robot stationary: zeroing gyro drift bias.",
                    )
                    if recalibrating_timer <= 0.0:
                        # Reset sensor suite bias and clear active decision
                        sensor_suite.imu_gyro_bias = 0.0
                        active_ai_decision = None

                elif active_ai_decision is not None:
                    current_ai_action_label = active_ai_decision.recommended_action
                    # Pass through Safety Governor
                    gov_res = governor.validate_and_govern(
                        command_v=0.0,
                        command_omega=0.0,
                        telemetry=corrupted_telemetry,
                        current_sim_time=current_time,
                        ai_decision=active_ai_decision,
                    )

                    if gov_res.status == "REJECTED_STALE":
                        stale_rejected_count += 1
                        # Clear expired decision so robot doesn't loop on it
                        active_ai_decision = None
                        if safe_state_time is None:
                            safe_state_time = current_time
                    elif gov_res.status == "ACCEPTED":
                        if safe_state_time is None:
                            safe_state_time = current_time
                        if active_ai_decision.recommended_action == "RECALIBRATE_IMU":
                            recalibrating_timer = 1.2
                        elif active_ai_decision.recommended_action == "SAFE_STOP_AND_HOLD":
                            in_fallback_hold = True
                elif pending_ai_decision is not None:
                    # An incident was detected and remote AI query is in-flight.
                    # Safety Governor enforces provisional safe hold during latency window.
                    gov_res = GovernorDecision(
                        status="PROVISIONAL_HOLD_AWAITING_AI",
                        executed_v=0.0,
                        executed_omega=0.0,
                        is_safe=True,
                        stale_age_sec=0.0,
                        reason="Sensor anomaly detected: provisional safe hold active while awaiting Nemotron recovery response.",
                    )
                    if safe_state_time is None:
                        safe_state_time = current_time
                else:
                    # Normal navigation command governed by Safety Governor
                    nav_v, nav_omega = controller.compute_control(corrupted_telemetry)
                    gov_res = governor.validate_and_govern(
                        nav_v, nav_omega, corrupted_telemetry, current_time
                    )

            governor_decisions.append(gov_res)

            # 6. Actuate Robot with Governor's Approved Output
            prev_x, prev_y = robot.x, robot.y
            robot_state = robot.step(gov_res.executed_v, gov_res.executed_omega, SIMULATION_DT)
            step_dist = math.hypot(robot.x - prev_x, robot.y - prev_y)
            total_path_length += step_dist

            # 7. Physical Collision & Boundary Evaluation (Ground Truth)
            is_collision = world.is_collision(robot.x, robot.y, robot.radius)
            clearance = world.distance_to_nearest_obstacle(robot.x, robot.y, robot.radius)
            min_overall_clearance = min(min_overall_clearance, clearance)

            if is_collision:
                robot.mark_collision()

            # Record step trace
            traces.append(StepTrace(
                t=round(current_time, 3),
                x_gt=round(robot.x, 3),
                y_gt=round(robot.y, 3),
                theta_gt=round(robot.theta, 3),
                x_odom=round(corrupted_telemetry.odometry.x, 3),
                y_odom=round(corrupted_telemetry.odometry.y, 3),
                theta_odom=round(corrupted_telemetry.odometry.theta, 3),
                v=round(gov_res.executed_v, 3),
                omega=round(gov_res.executed_omega, 3),
                min_clearance=round(clearance, 3),
                fault_active=ground_truth_fault.is_active,
                incident_active=(detected_incident is not None),
                ai_action=current_ai_action_label,
                governor_status=gov_res.status,
                is_collided=is_collision,
                lidar_ranges=[round(float(r), 3) for r in corrupted_telemetry.lidar.ranges],
                lidar_angles=[round(float(a), 3) for a in corrupted_telemetry.lidar.angles],
            ))

            # Stop condition: collision or goal reached
            if is_collision or controller.is_goal_reached:
                break

        # Calculate metrics
        dur = round(robot.time, 3)
        collision_occ = robot.is_collided
        goal_reached = controller.is_goal_reached and not collision_occ
        fault_occ = (fault_first_time is not None)
        fault_det = (fault_detection_time is not None)

        det_lat = round(fault_detection_time - fault_first_time, 3) if (fault_det and fault_occ) else None
        tt_safe = round(safe_state_time - fault_first_time, 3) if (safe_state_time and fault_occ) else None

        # False stop calculation
        if not fault_occ and (in_fallback_hold or total_path_length < 0.5):
            unnecessary_stops = 1

        mean_ai_lat = float(np.mean(ai_latencies)) if ai_latencies else 0.0

        metrics = EpisodeMetrics(
            scenario_id=scenario.id,
            variant_id=variant_id,
            seed=seed,
            duration_sec=dur,
            goal_reached=goal_reached,
            collision_occurred=collision_occ,
            path_length_meters=round(total_path_length, 3),
            min_obstacle_clearance_meters=round(min_overall_clearance, 3),
            fault_occurred=fault_occ,
            fault_detected=fault_det,
            detection_latency_sec=det_lat,
            time_to_safe_state_sec=tt_safe,
            stale_decisions_rejected=stale_rejected_count,
            stale_decisions_accepted=stale_accepted_count,
            governor_critical_overrides=governor.emergency_stops_count,
            total_ai_decisions=len(ai_decisions),
            mean_ai_latency_sec=round(mean_ai_lat, 3),
            unnecessary_stops=unnecessary_stops,
        )

        return EpisodeRunResult(
            scenario_id=scenario.id,
            variant_id=variant_id,
            metrics=metrics,
            trace=traces,
            incidents=incidents,
            ai_decisions=ai_decisions,
            governor_log=governor_decisions,
        )
