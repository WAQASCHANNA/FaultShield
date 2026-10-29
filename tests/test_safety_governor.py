import pytest
from faultshield.core.world import WarehouseMap
from faultshield.core.robot import DifferentialDriveRobot
from faultshield.core.sensors import SensorSuite
from faultshield.safety.governor import SafetyGovernor
from faultshield.agent.nemotron import NemotronDecision


def test_safety_governor_rejects_stale_decision():
    governor = SafetyGovernor(max_ttl_sec=0.50)
    world = WarehouseMap()
    robot = DifferentialDriveRobot(1.5, 1.5, 0.0)
    sensors = SensorSuite()
    telemetry = sensors.generate_telemetry_packet(world, robot, 0.05)

    # Decision dispatched at t=1.0s
    decision = NemotronDecision(
        decision_id="DEC-TEST-01",
        incident_id="INC-01",
        recommended_action="REVERSE_SLOWLY",
        target_linear_velocity=-0.2,
        target_angular_velocity=0.0,
        confidence=0.9,
        root_cause_diagnosis="Test diagnosis",
        safety_rationale="Test rationale",
        requires_immediate_stop=False,
        dispatch_timestamp=1.0,
        arrival_timestamp=1.8,
        latency_sec=0.8,
        is_simulated=True,
        raw_response="{}",
    )

    # Evaluated at t=1.8s (age = 0.8s > 0.5s TTL)
    res = governor.validate_and_govern(
        command_v=0.0,
        command_omega=0.0,
        telemetry=telemetry,
        current_sim_time=1.8,
        ai_decision=decision,
    )

    assert res.status == "REJECTED_STALE"
    assert res.executed_v == 0.0
    assert res.executed_omega == 0.0
    assert governor.stale_rejections_count == 1


def test_safety_governor_accepts_fresh_allowlisted_action():
    governor = SafetyGovernor(max_ttl_sec=0.50)
    world = WarehouseMap()
    robot = DifferentialDriveRobot(1.5, 1.5, 0.0)
    sensors = SensorSuite()
    telemetry = sensors.generate_telemetry_packet(world, robot, 0.05)

    # Fresh decision: dispatched at 1.0s, evaluated at 1.1s (age = 0.1s < 0.5s)
    decision = NemotronDecision(
        decision_id="DEC-TEST-02",
        incident_id="INC-02",
        recommended_action="SAFE_STOP_AND_HOLD",
        target_linear_velocity=0.0,
        target_angular_velocity=0.0,
        confidence=0.95,
        root_cause_diagnosis="Valid diagnosis",
        safety_rationale="Hold in place",
        requires_immediate_stop=True,
        dispatch_timestamp=1.0,
        arrival_timestamp=1.1,
        latency_sec=0.1,
        is_simulated=True,
        raw_response="{}",
    )

    res = governor.validate_and_govern(
        command_v=0.0,
        command_omega=0.0,
        telemetry=telemetry,
        current_sim_time=1.1,
        ai_decision=decision,
    )

    assert res.status == "ACCEPTED"
    assert res.executed_v == 0.0


def test_safety_governor_rejects_unallowed_action():
    governor = SafetyGovernor()
    world = WarehouseMap()
    robot = DifferentialDriveRobot(1.5, 1.5, 0.0)
    sensors = SensorSuite()
    telemetry = sensors.generate_telemetry_packet(world, robot, 0.05)

    decision = NemotronDecision(
        decision_id="DEC-TEST-03",
        incident_id="INC-03",
        recommended_action="FULL_THROTTLE_IGNORE_SENSORS",  # Disallowed!
        target_linear_velocity=1.0,
        target_angular_velocity=0.0,
        confidence=0.5,
        root_cause_diagnosis="Bad action",
        safety_rationale="Invalid",
        requires_immediate_stop=False,
        dispatch_timestamp=1.0,
        arrival_timestamp=1.05,
        latency_sec=0.05,
        is_simulated=True,
        raw_response="{}",
    )

    res = governor.validate_and_govern(
        command_v=0.0,
        command_omega=0.0,
        telemetry=telemetry,
        current_sim_time=1.05,
        ai_decision=decision,
    )

    assert res.status == "REJECTED_DISALLOWED"
    assert res.executed_v == 0.0
    assert governor.disallowed_rejections_count == 1
