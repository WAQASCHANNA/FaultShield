import pytest
import numpy as np
from faultshield.core.world import WarehouseMap
from faultshield.core.robot import DifferentialDriveRobot
from faultshield.core.sensors import SensorSuite
from faultshield.faults.injector import FaultInjector, FaultType


def test_fault_injector_timing():
    injector = FaultInjector(fault_type=FaultType.LIDAR_FROZEN, start_time=2.0, duration=5.0)
    assert injector.is_fault_active(1.0) is False
    assert injector.is_fault_active(2.0) is True
    assert injector.is_fault_active(5.0) is True
    assert injector.is_fault_active(7.5) is False


def test_lidar_freeze_injection():
    world = WarehouseMap()
    robot = DifferentialDriveRobot(1.5, 1.5, 0.0)
    sensors = SensorSuite(seed=123)
    injector = FaultInjector(fault_type=FaultType.LIDAR_FROZEN, start_time=0.1, duration=10.0)

    # Step 0 (t=0.0): healthy
    p0 = sensors.generate_telemetry_packet(world, robot, 0.05)
    out0 = injector.inject(p0, 0.05)

    # Move robot forward
    robot.step(cmd_v=0.8, cmd_omega=0.0, dt=0.05)
    robot.time = 0.15

    # Step 1 (t=0.15): fault active -> LiDAR ranges should be frozen to first captured frame
    p1 = sensors.generate_telemetry_packet(world, robot, 0.05)
    out1 = injector.inject(p1, 0.05)

    # Robot moved, but corrupted LiDAR ranges must match frozen values exactly
    assert np.allclose(out1.lidar.ranges, injector.frozen_lidar_frame.ranges)


def test_imu_bias_injection():
    world = WarehouseMap()
    robot = DifferentialDriveRobot(1.5, 1.5, 0.0)
    sensors = SensorSuite(seed=42)
    injector = FaultInjector(fault_type=FaultType.IMU_BIAS, start_time=0.5, duration=5.0, magnitude=1.0)

    robot.time = 1.0  # within fault window
    raw = sensors.generate_telemetry_packet(world, robot, 0.05)
    corrupted = injector.inject(raw, 0.05)

    # Corrupted IMU yaw rate should be shifted by offset
    assert corrupted.imu.yaw_rate > raw.imu.yaw_rate + 0.4
