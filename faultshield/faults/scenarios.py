"""
Benchmark Evaluation Scenarios:
Standardized fault profiles representing real-world hardware & network failure modes.
"""

from dataclasses import dataclass
from typing import Dict, List
from faultshield.faults.injector import FaultType, FaultInjector


@dataclass
class Scenario:
    id: str
    name: str
    description: str
    fault_type: FaultType
    start_time: float
    duration: float
    magnitude: float
    ai_delay_sec: float
    start_pos: tuple  # (x, y, theta)
    goal_pos: tuple   # (x, y)

    def create_injector(self) -> FaultInjector:
        return FaultInjector(
            fault_type=self.fault_type,
            start_time=self.start_time,
            duration=self.duration,
            magnitude=self.magnitude,
            ai_delay_sec=self.ai_delay_sec,
        )


BENCHMARK_SCENARIOS: Dict[str, Scenario] = {
    "exp1_healthy": Scenario(
        id="exp1_healthy",
        name="Healthy sensors",
        description="Normal operation without sensor faults or latency. Baseline performance anchor.",
        fault_type=FaultType.NONE,
        start_time=0.0,
        duration=0.0,
        magnitude=0.0,
        ai_delay_sec=0.05,
        start_pos=(1.5, 1.5, 0.0),
        goal_pos=(10.5, 8.5),
    ),
    "exp2_frozen_lidar": Scenario(
        id="exp2_frozen_lidar",
        name="Stale LiDAR",
        description="LiDAR driver freezes on open corridor reading while robot approaches central aisle racks.",
        fault_type=FaultType.LIDAR_FROZEN,
        start_time=2.5,
        duration=8.0,
        magnitude=1.0,
        ai_delay_sec=0.1,
        start_pos=(1.5, 1.5, 0.0),
        goal_pos=(5.0, 5.0),
    ),
    "exp3_contradictory_telemetry": Scenario(
        id="exp3_contradictory_telemetry",
        name="Experiment 3: Contradictory Measurements",
        description="IMU gyro experiences acute drift bias while wheel encoders report straight trajectory.",
        fault_type=FaultType.IMU_BIAS,
        start_time=2.0,
        duration=7.0,
        magnitude=1.2,
        ai_delay_sec=0.1,
        start_pos=(1.5, 1.5, 0.0),
        goal_pos=(10.0, 3.0),
    ),
    "exp4_delayed_ai": Scenario(
        id="exp4_delayed_ai",
        name="Experiment 4: Delayed AI Decision",
        description="Network jitter and cloud reasoning delay (2.0s) causing recommendation staleness.",
        fault_type=FaultType.AI_DECISION_DELAY,
        start_time=2.0,
        duration=10.0,
        magnitude=1.0,
        ai_delay_sec=2.2,  # exceeds GOVERNOR_MAX_DECISION_TTL_SEC (0.5s)
        start_pos=(1.5, 1.5, 0.0),
        goal_pos=(7.0, 7.0),
    ),
    "exp5_compound_faults": Scenario(
        id="exp5_compound_faults",
        name="Experiment 5: Compound Cross-Layer Faults",
        description="Simultaneous frozen LiDAR, wheel slip drift, and AI inference latency.",
        fault_type=FaultType.COMPOUND,
        start_time=2.0,
        duration=9.0,
        magnitude=1.0,
        ai_delay_sec=1.8,
        start_pos=(1.5, 1.5, 0.0),
        goal_pos=(9.0, 7.0),
    ),
}
