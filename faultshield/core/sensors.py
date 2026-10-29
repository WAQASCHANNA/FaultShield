"""
Sensor suite simulation for the mobile robot:
- 2D LiDAR Scanner
- Wheel Odometry
- Inertial Measurement Unit (IMU)
- Structured TelemetryPacket
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
from faultshield.config import (
    LIDAR_NUM_BEAMS,
    LIDAR_FOV_DEG,
    LIDAR_MIN_RANGE,
    LIDAR_MAX_RANGE,
    LIDAR_NOISE_STD,
    ODOM_NOISE_STD,
    ODOM_SLIP_FACTOR,
    IMU_GYRO_BIAS_STD,
    IMU_ACCEL_NOISE_STD,
)
from faultshield.core.world import WarehouseMap
from faultshield.core.robot import DifferentialDriveRobot, wrap_to_pi


@dataclass
class LidarReading:
    timestamp: float
    ranges: np.ndarray  # shape: (num_beams,)
    angles: np.ndarray  # shape: (num_beams,)
    min_distance: float
    forward_distance: float
    is_valid: bool = True


@dataclass
class OdometryReading:
    timestamp: float
    x: float
    y: float
    theta: float
    v: float
    omega: float
    is_valid: bool = True


@dataclass
class IMUReading:
    timestamp: float
    yaw_rate: float  # rad/s
    ax: float  # m/s^2 forward accel
    ay: float  # m/s^2 lateral accel
    is_valid: bool = True


@dataclass
class TelemetryPacket:
    timestamp: float
    step_id: int
    lidar: LidarReading
    odometry: OdometryReading
    imu: IMUReading
    is_stale: bool = False


class SensorSuite:
    def __init__(self, seed: Optional[int] = 42):
        self.rng = np.random.RandomState(seed)
        self.step_counter = 0

        # Odometry internal dead reckoning
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0

        # IMU persistent drift
        self.imu_gyro_bias = 0.0

    def reset(self, init_x: float = 1.5, init_y: float = 1.5, init_theta: float = 0.0):
        self.step_counter = 0
        self.odom_x = float(init_x)
        self.odom_y = float(init_y)
        self.odom_theta = wrap_to_pi(float(init_theta))
        self.imu_gyro_bias = self.rng.normal(0.0, IMU_GYRO_BIAS_STD)

    def read_lidar(
        self,
        world: WarehouseMap,
        robot: DifferentialDriveRobot,
        add_noise: bool = True
    ) -> LidarReading:
        """Simulate healthy 2D LiDAR raycast."""
        fov_rad = math.radians(LIDAR_FOV_DEG)
        ranges, angles = world.cast_lidar(
            ox=robot.x,
            oy=robot.y,
            robot_heading_rad=robot.theta,
            num_beams=LIDAR_NUM_BEAMS,
            fov_rad=fov_rad,
            max_range=LIDAR_MAX_RANGE,
        )

        if add_noise:
            noise = self.rng.normal(0.0, LIDAR_NOISE_STD, size=ranges.shape)
            ranges = np.clip(ranges + noise, LIDAR_MIN_RANGE, LIDAR_MAX_RANGE)

        # Center beam corresponds to direct forward line of sight
        center_idx = len(ranges) // 2
        forward_dist = float(ranges[center_idx])
        min_dist = float(np.min(ranges))

        return LidarReading(
            timestamp=robot.time,
            ranges=ranges,
            angles=angles,
            min_distance=min_dist,
            forward_distance=forward_dist,
            is_valid=True,
        )

    def update_and_read_odometry(
        self,
        robot: DifferentialDriveRobot,
        dt: float,
        add_noise: bool = True
    ) -> OdometryReading:
        """Dead reckoning wheel odometry integration with small slip and encoder noise."""
        v_noisy = robot.v
        omega_noisy = robot.omega

        if add_noise:
            # Wheel slip and quantization
            slip = 1.0 + self.rng.normal(0.0, ODOM_SLIP_FACTOR)
            v_noisy = robot.v * slip + self.rng.normal(0.0, ODOM_NOISE_STD)
            omega_noisy = robot.omega + self.rng.normal(0.0, ODOM_NOISE_STD * 0.5)

        # Integrate pose
        mid_theta = wrap_to_pi(self.odom_theta + 0.5 * omega_noisy * dt)
        self.odom_x += v_noisy * math.cos(mid_theta) * dt
        self.odom_y += v_noisy * math.sin(mid_theta) * dt
        self.odom_theta = wrap_to_pi(self.odom_theta + omega_noisy * dt)

        return OdometryReading(
            timestamp=robot.time,
            x=self.odom_x,
            y=self.odom_y,
            theta=self.odom_theta,
            v=v_noisy,
            omega=omega_noisy,
            is_valid=True,
        )

    def read_imu(
        self,
        robot: DifferentialDriveRobot,
        dt: float,
        add_noise: bool = True
    ) -> IMUReading:
        """Inertial measurement unit: measures angular yaw rate and linear accelerations."""
        yaw_rate = robot.omega + self.imu_gyro_bias
        ax = 0.0
        ay = 0.0

        if add_noise:
            yaw_rate += self.rng.normal(0.0, IMU_GYRO_BIAS_STD)
            ax = self.rng.normal(0.0, IMU_ACCEL_NOISE_STD)
            ay = self.rng.normal(0.0, IMU_ACCEL_NOISE_STD)

        return IMUReading(
            timestamp=robot.time,
            yaw_rate=yaw_rate,
            ax=ax,
            ay=ay,
            is_valid=True,
        )

    def generate_telemetry_packet(
        self,
        world: WarehouseMap,
        robot: DifferentialDriveRobot,
        dt: float
    ) -> TelemetryPacket:
        """Capture standard healthy telemetry packet from all sensors."""
        self.step_counter += 1
        lidar = self.read_lidar(world, robot)
        odom = self.update_and_read_odometry(robot, dt)
        imu = self.read_imu(robot, dt)

        return TelemetryPacket(
            timestamp=robot.time,
            step_id=self.step_counter,
            lidar=lidar,
            odometry=odom,
            imu=imu,
            is_stale=False,
        )
