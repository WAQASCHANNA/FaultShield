"""
Configurable Cross-Layer Fault Injection Engine for Autonomous Mobile Robots.
Simulates sensor corruptions, asynchronous delays, and telemetry dropouts.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
from faultshield.core.sensors import TelemetryPacket, LidarReading, OdometryReading, IMUReading
from faultshield.core.robot import wrap_to_pi


class FaultType(str, Enum):
    NONE = "NONE"
    LIDAR_FROZEN = "LIDAR_FROZEN"
    LIDAR_DROPOUT = "LIDAR_DROPOUT"
    ODOMETRY_DRIFT = "ODOMETRY_DRIFT"
    IMU_BIAS = "IMU_BIAS"
    AI_DECISION_DELAY = "AI_DECISION_DELAY"
    COMPOUND = "COMPOUND"


@dataclass
class ActiveFaultState:
    fault_type: FaultType
    is_active: bool
    time_since_trigger: float
    description: str


class FaultInjector:
    def __init__(
        self,
        fault_type: FaultType = FaultType.NONE,
        start_time: float = 3.0,
        duration: float = 10.0,
        magnitude: float = 1.0,
        ai_delay_sec: float = 1.5,
    ):
        self.fault_type = fault_type
        self.start_time = start_time
        self.duration = duration
        self.magnitude = magnitude
        self.ai_delay_sec = ai_delay_sec

        # State memory for frozen or accumulated faults
        self.frozen_lidar_frame: Optional[LidarReading] = None
        self.accumulated_odom_drift_x = 0.0
        self.accumulated_odom_drift_y = 0.0
        self.accumulated_odom_drift_theta = 0.0

    def reset(self):
        self.frozen_lidar_frame = None
        self.accumulated_odom_drift_x = 0.0
        self.accumulated_odom_drift_y = 0.0
        self.accumulated_odom_drift_theta = 0.0

    def is_fault_active(self, current_time: float) -> bool:
        if self.fault_type == FaultType.NONE:
            return False
        return (self.start_time <= current_time <= (self.start_time + self.duration))

    def get_ground_truth_fault_status(self, current_time: float) -> ActiveFaultState:
        active = self.is_fault_active(current_time)
        t_since = max(0.0, current_time - self.start_time) if active else 0.0
        desc = f"Injected: {self.fault_type.value}" if active else "Healthy Telemetry"
        return ActiveFaultState(
            fault_type=self.fault_type if active else FaultType.NONE,
            is_active=active,
            time_since_trigger=t_since,
            description=desc,
        )

    def inject(self, telemetry: TelemetryPacket, dt: float) -> TelemetryPacket:
        """
        Inject faults into raw telemetry packet if active at telemetry.timestamp.
        Returns a modified TelemetryPacket (or frozen/corrupted copy).
        """
        current_time = telemetry.timestamp
        if not self.is_fault_active(current_time):
            return telemetry

        # Create copies of sensor readings to prevent mutating original ground truth
        corrupted_lidar = telemetry.lidar
        corrupted_odom = telemetry.odometry
        corrupted_imu = telemetry.imu

        # Apply specific fault types
        is_lidar_frozen = (self.fault_type == FaultType.LIDAR_FROZEN or self.fault_type == FaultType.COMPOUND)
        is_lidar_dropout = (self.fault_type == FaultType.LIDAR_DROPOUT or self.fault_type == FaultType.AI_DECISION_DELAY)
        is_odom_drift = (self.fault_type == FaultType.ODOMETRY_DRIFT or self.fault_type == FaultType.COMPOUND)
        is_imu_bias = (self.fault_type == FaultType.IMU_BIAS or self.fault_type == FaultType.COMPOUND)

        # 1. LiDAR Freeze: locks the sensor to an earlier safe frame
        if is_lidar_frozen:
            if self.frozen_lidar_frame is None:
                # Store the last healthy frame
                self.frozen_lidar_frame = telemetry.lidar
            corrupted_lidar = LidarReading(
                timestamp=self.frozen_lidar_frame.timestamp,  # stale timestamp
                ranges=self.frozen_lidar_frame.ranges.copy(),
                angles=self.frozen_lidar_frame.angles.copy(),
                min_distance=self.frozen_lidar_frame.min_distance,
                forward_distance=self.frozen_lidar_frame.forward_distance,
                is_valid=True,
            )

        # 2. LiDAR Dropout: sensor reports max clear distance (blind to obstacle)
        elif is_lidar_dropout:
            cleared_ranges = np.full_like(telemetry.lidar.ranges, 6.0)
            corrupted_lidar = LidarReading(
                timestamp=telemetry.timestamp,
                ranges=cleared_ranges,
                angles=telemetry.lidar.angles.copy(),
                min_distance=6.0,
                forward_distance=6.0,
                is_valid=True,
            )

        # 3. Odometry Drift: Wheel slip causing accumulated error in position & heading
        if is_odom_drift:
            drift_rate_heading = 0.25 * self.magnitude  # rad/s
            self.accumulated_odom_drift_theta += drift_rate_heading * dt
            drift_v_bias = 0.40 * self.magnitude
            self.accumulated_odom_drift_x += drift_v_bias * np.cos(corrupted_odom.theta) * dt

            corrupted_odom = OdometryReading(
                timestamp=telemetry.timestamp,
                x=telemetry.odometry.x + self.accumulated_odom_drift_x,
                y=telemetry.odometry.y + self.accumulated_odom_drift_y,
                theta=wrap_to_pi(telemetry.odometry.theta + self.accumulated_odom_drift_theta),
                v=telemetry.odometry.v + drift_v_bias,
                omega=telemetry.odometry.omega + drift_rate_heading,
                is_valid=True,
            )

        # 4. IMU Bias: Gyro reports high rotation rate when robot is actually moving straight
        if is_imu_bias:
            bias_offset = 0.55 * self.magnitude  # rad/s
            corrupted_imu = IMUReading(
                timestamp=telemetry.timestamp,
                yaw_rate=telemetry.imu.yaw_rate + bias_offset,
                ax=telemetry.imu.ax + 0.3 * self.magnitude,
                ay=telemetry.imu.ay,
                is_valid=True,
            )

        return TelemetryPacket(
            timestamp=telemetry.timestamp,
            step_id=telemetry.step_id,
            lidar=corrupted_lidar,
            odometry=corrupted_odom,
            imu=corrupted_imu,
            is_stale=False,
        )
