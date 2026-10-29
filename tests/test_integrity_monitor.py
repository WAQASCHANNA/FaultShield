import pytest
import numpy as np
from faultshield.core.world import WarehouseMap
from faultshield.core.robot import DifferentialDriveRobot
from faultshield.core.sensors import SensorSuite
from faultshield.integrity.monitor import SensorIntegrityMonitor
from faultshield.faults.injector import FaultInjector, FaultType


def test_monitor_detects_frozen_lidar():
    world = WarehouseMap()
    robot = DifferentialDriveRobot(1.5, 1.5, 0.0)
    sensors = SensorSuite(seed=99)
    injector = FaultInjector(fault_type=FaultType.LIDAR_FROZEN, start_time=0.0, duration=10.0)
    monitor = SensorIntegrityMonitor()

    incident_detected = None
    for step in range(12):
        t = step * 0.05
        robot.time = t
        robot.step(0.5, 0.0, 0.05)
        raw = sensors.generate_telemetry_packet(world, robot, 0.05)
        corrupted = injector.inject(raw, 0.05)
        inc = monitor.inspect(corrupted, current_sim_time=t)
        if inc is not None and "LIDAR_FROZEN" in inc.primary_fault:
            incident_detected = inc
            break

    assert incident_detected is not None
    assert incident_detected.risk_level in ("HIGH", "CRITICAL")
    assert incident_detected.active_anomalies[0].confidence > 0.90


def test_monitor_detects_imu_odometry_discrepancy():
    world = WarehouseMap()
    robot = DifferentialDriveRobot(1.5, 1.5, 0.0)
    sensors = SensorSuite(seed=77)
    injector = FaultInjector(fault_type=FaultType.IMU_BIAS, start_time=0.0, duration=5.0, magnitude=1.5)
    monitor = SensorIntegrityMonitor()

    robot.time = 0.5
    raw = sensors.generate_telemetry_packet(world, robot, 0.05)
    corrupted = injector.inject(raw, 0.05)
    inc = monitor.inspect(corrupted, current_sim_time=0.5)

    assert inc is not None
    assert any("IMU_ODOMETRY_DISCREPANCY" in a.anomaly_type for a in inc.active_anomalies)
