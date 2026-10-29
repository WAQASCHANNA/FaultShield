"""
Sensor-Integrity Monitor (SIM).
Performs deterministic mathematical checks:
1. Temporal freshness & heartbeat tracking
2. Variance & frozen frame detection (LiDAR)
3. Cross-modal kinematic consistency (IMU vs Odometry)
4. Range plausibility vs workspace bounds
5. Structured Incident Report packaging for Nemotron
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
from faultshield.config import (
    MONITOR_LIDAR_FREEZE_STEPS,
    MONITOR_ODOM_IMU_YAW_DISCREPANCY_RAD,
    MONITOR_HEARTBEAT_TIMEOUT,
)
from faultshield.core.sensors import TelemetryPacket, LidarReading, OdometryReading, IMUReading


@dataclass
class AnomalyAlert:
    sensor_name: str
    anomaly_type: str
    confidence: float
    discrepancy_value: float
    threshold: float
    message: str


@dataclass
class IncidentReport:
    incident_id: str
    timestamp: float
    active_anomalies: List[AnomalyAlert]
    primary_fault: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    perceived_robot_state: Dict[str, float]
    raw_discrepancy_details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "timestamp": round(self.timestamp, 3),
            "primary_fault": self.primary_fault,
            "risk_level": self.risk_level,
            "anomalies": [
                {
                    "sensor": a.sensor_name,
                    "type": a.anomaly_type,
                    "confidence": round(a.confidence, 2),
                    "discrepancy": round(a.discrepancy_value, 4),
                    "threshold": round(a.threshold, 4),
                    "message": a.message,
                }
                for a in self.active_anomalies
            ],
            "perceived_robot_state": self.perceived_robot_state,
            "discrepancy_details": self.raw_discrepancy_details,
        }


class SensorIntegrityMonitor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_telemetry: Optional[TelemetryPacket] = None
        self.lidar_history: List[np.ndarray] = []
        self.freeze_consecutive_count = 0
        self.total_incidents_flagged = 0
        self.incident_history: List[IncidentReport] = []

    def inspect(
        self,
        telemetry: TelemetryPacket,
        current_sim_time: float
    ) -> Optional[IncidentReport]:
        """
        Evaluate incoming telemetry for corruption, freezes, drift, and discrepancies.
        Returns IncidentReport if any severe anomaly is detected, else None.
        """
        anomalies: List[AnomalyAlert] = []

        # 1. Freshness / Heartbeat Verification
        time_delta = current_sim_time - telemetry.timestamp
        if time_delta > MONITOR_HEARTBEAT_TIMEOUT:
            anomalies.append(AnomalyAlert(
                sensor_name="TELEMETRY_BUS",
                anomaly_type="STALE_STREAM_LATENCY",
                confidence=min(1.0, time_delta / 1.0),
                discrepancy_value=time_delta,
                threshold=MONITOR_HEARTBEAT_TIMEOUT,
                message=f"Telemetry packet delayed by {time_delta:.3f}s (threshold: {MONITOR_HEARTBEAT_TIMEOUT}s)",
            ))

        # 2. LiDAR Frozen Frame Detection
        curr_ranges = telemetry.lidar.ranges
        if len(self.lidar_history) > 0:
            last_ranges = self.lidar_history[-1]
            diff = np.max(np.abs(curr_ranges - last_ranges))
            # If robot is moving or noise expected, diff should be > 1e-4
            if diff < 1e-4:
                self.freeze_consecutive_count += 1
            else:
                self.freeze_consecutive_count = 0

            if self.freeze_consecutive_count >= MONITOR_LIDAR_FREEZE_STEPS:
                anomalies.append(AnomalyAlert(
                    sensor_name="LIDAR",
                    anomaly_type="LIDAR_FROZEN_FRAME",
                    confidence=0.98,
                    discrepancy_value=float(diff),
                    threshold=1e-4,
                    message=f"LiDAR ranges static for {self.freeze_consecutive_count} frames. Range variance=0.0",
                ))
        self.lidar_history.append(curr_ranges.copy())
        if len(self.lidar_history) > 10:
            self.lidar_history.pop(0)

        # 3. Cross-Sensor Kinematic Discrepancy (IMU Gyro vs Wheel Odometry Angular Velocity)
        imu_yaw_rate = telemetry.imu.yaw_rate
        odom_omega = telemetry.odometry.omega
        yaw_discrepancy = abs(imu_yaw_rate - odom_omega)

        if yaw_discrepancy > MONITOR_ODOM_IMU_YAW_DISCREPANCY_RAD:
            conf = min(1.0, yaw_discrepancy / (MONITOR_ODOM_IMU_YAW_DISCREPANCY_RAD * 2.0))
            anomalies.append(AnomalyAlert(
                sensor_name="CROSS_SENSOR_KINEMATICS",
                anomaly_type="IMU_ODOMETRY_DISCREPANCY",
                confidence=conf,
                discrepancy_value=yaw_discrepancy,
                threshold=MONITOR_ODOM_IMU_YAW_DISCREPANCY_RAD,
                message=f"IMU yaw rate ({imu_yaw_rate:+.3f} rad/s) contradicts wheel odometry ({odom_omega:+.3f} rad/s)",
            ))

        # 4. Check for unphysical LiDAR dropout (all beams showing 6.0m max in enclosed room)
        if np.all(curr_ranges >= 5.95):
            anomalies.append(AnomalyAlert(
                sensor_name="LIDAR",
                anomaly_type="LIDAR_BLIND_DROPOUT",
                confidence=0.90,
                discrepancy_value=float(np.min(curr_ranges)),
                threshold=5.95,
                message="All LiDAR beams saturated at max range (6.0m), indicating sensor blackout",
            ))

        self.prev_telemetry = telemetry

        # Determine if an IncidentReport should be generated
        if not anomalies:
            return None

        self.total_incidents_flagged += 1
        primary_fault = anomalies[0].anomaly_type
        risk_level = "CRITICAL" if any(a.confidence > 0.85 for a in anomalies) else "HIGH"

        details = "; ".join(a.message for a in anomalies)
        incident = IncidentReport(
            incident_id=f"INC-{self.total_incidents_flagged:04d}",
            timestamp=current_sim_time,
            active_anomalies=anomalies,
            primary_fault=primary_fault,
            risk_level=risk_level,
            perceived_robot_state={
                "odom_x": round(telemetry.odometry.x, 3),
                "odom_y": round(telemetry.odometry.y, 3),
                "odom_theta": round(telemetry.odometry.theta, 3),
                "odom_v": round(telemetry.odometry.v, 3),
                "imu_yaw_rate": round(telemetry.imu.yaw_rate, 3),
                "lidar_min_dist": round(telemetry.lidar.min_distance, 3),
            },
            raw_discrepancy_details=details,
        )

        self.incident_history.append(incident)
        return incident
