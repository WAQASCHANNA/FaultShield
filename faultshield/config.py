"""
FaultShield Configuration & Global Constants
Cross-layer reliability benchmark and recovery agent for mobile robots.
"""

from dataclasses import dataclass
from typing import List, Tuple

# Simulation Timestep
SIMULATION_DT = 0.05  # 20 Hz simulation loop
MAX_SIMULATION_STEPS = 600  # 30 seconds max per run

# Differential-Drive Robot Parameters
ROBOT_RADIUS = 0.35  # meters
MAX_LINEAR_VELOCITY = 1.0  # m/s
MAX_ANGULAR_VELOCITY = 1.5  # rad/s
MAX_LINEAR_ACCELERATION = 2.0  # m/s^2
MAX_ANGULAR_ACCELERATION = 3.0  # rad/s^2

# LiDAR Sensor Parameters
LIDAR_NUM_BEAMS = 24
LIDAR_FOV_DEG = 270.0  # 270 degree forward/lateral field of view
LIDAR_MIN_RANGE = 0.1  # meters
LIDAR_MAX_RANGE = 6.0  # meters
LIDAR_NOISE_STD = 0.02  # meters (Gaussian noise in healthy mode)

# Odometry Parameters
ODOM_SLIP_FACTOR = 0.01  # Normal wheel slip variance
ODOM_NOISE_STD = 0.01  # meters

# IMU Parameters
IMU_GYRO_BIAS_STD = 0.005  # rad/s drift in normal mode
IMU_ACCEL_NOISE_STD = 0.03  # m/s^2

# Sensor-Integrity Monitor Thresholds
MONITOR_LIDAR_FREEZE_STEPS = 5  # consecutive identical frames = frozen
MONITOR_ODOM_IMU_YAW_DISCREPANCY_RAD = 0.20  # ~11.5 deg discrepancy
MONITOR_ODOM_VELOCITY_DISCREPANCY = 0.25  # m/s
MONITOR_HEARTBEAT_TIMEOUT = 0.3  # seconds (missing telemetry)

# Independent Safety Governor
GOVERNOR_CRITICAL_STOP_DISTANCE = 0.42  # meters to obstacle triggers hard stop
GOVERNOR_WARNING_DISTANCE = 0.75  # meters
GOVERNOR_MAX_DECISION_TTL_SEC = 0.50  # Any Nemotron decision older than 500ms is REJECTED
GOVERNOR_RECOVERY_SPEED_LIMIT = 0.35  # m/s max during recovery

# Allowlisted Recovery Actions for Nemotron
PERMITTED_RECOVERY_ACTIONS = [
    "SAFE_STOP_AND_HOLD",
    "REVERSE_SLOWLY",
    "ROTATE_IN_PLACE_SCAN",
    "SWITCH_TO_CONSERVATIVE_CONTOURING",
    "FALLBACK_WALL_FOLLOW",
    "SAFE_STOP_AND_REPLAN",
    "RECALIBRATE_IMU",
]

# Warehouse Environment
MAP_WIDTH = 12.0  # meters
MAP_HEIGHT = 10.0  # meters
DEFAULT_START_POS = (1.5, 1.5, 0.0)  # x, y, theta
DEFAULT_GOAL_POS = (10.5, 8.5)  # x, y

# Nebius & Nemotron Settings
DEFAULT_NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_NEMOTRON_MODEL = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_REQUEST_TIMEOUT = 15.0  # seconds
