"""
Quantitative metrics calculation for FaultShield benchmark runs.
Tracks safety outcomes, detection precision, recovery efficacy, and operational costs.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class EpisodeMetrics:
    scenario_id: str
    variant_id: str  # "Variant_A", "Variant_B", "Variant_C", "Variant_D"
    seed: int
    duration_sec: float
    goal_reached: bool
    collision_occurred: bool
    path_length_meters: float
    min_obstacle_clearance_meters: float

    # Fault & Safety metrics
    fault_occurred: bool
    fault_detected: bool
    detection_latency_sec: Optional[float]
    time_to_safe_state_sec: Optional[float]

    # Model & Governor metrics
    stale_decisions_rejected: int
    stale_decisions_accepted: int  # Must be ZERO in FaultShield
    governor_critical_overrides: int
    total_ai_decisions: int
    mean_ai_latency_sec: float
    unnecessary_stops: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Round floats for clean output
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


@dataclass
class BenchmarkSummary:
    variant_id: str
    num_episodes: int
    collision_rate: float
    recovery_rate: float
    goal_completion_rate: float
    mean_duration_sec: float
    mean_path_length_m: float
    mean_detection_latency_sec: float
    mean_time_to_safe_state_sec: float
    stale_acceptance_rate: float
    mean_ai_latency_sec: float


def compute_benchmark_summary(metrics_list: List[EpisodeMetrics]) -> Optional[BenchmarkSummary]:
    if not metrics_list:
        return None

    n = len(metrics_list)
    variant_id = metrics_list[0].variant_id

    collisions = sum(1 for m in metrics_list if m.collision_occurred)
    goals = sum(1 for m in metrics_list if m.goal_reached)
    stale_accepted = sum(m.stale_decisions_accepted for m in metrics_list)

    # Recoveries: fault occurred, no collision, and robot reached goal or entered safe stop without crashing
    fault_episodes = [m for m in metrics_list if m.fault_occurred]
    recovered = sum(1 for m in fault_episodes if not m.collision_occurred and (m.goal_reached or m.time_to_safe_state_sec is not None))
    recovery_rate = (recovered / len(fault_episodes)) if fault_episodes else 1.0

    durations = [m.duration_sec for m in metrics_list]
    path_lengths = [m.path_length_meters for m in metrics_list]
    latencies = [m.detection_latency_sec for m in metrics_list if m.detection_latency_sec is not None]
    tt_safe = [m.time_to_safe_state_sec for m in metrics_list if m.time_to_safe_state_sec is not None]
    ai_lats = [m.mean_ai_latency_sec for m in metrics_list if m.mean_ai_latency_sec > 0]

    return BenchmarkSummary(
        variant_id=variant_id,
        num_episodes=n,
        collision_rate=round(collisions / n, 4),
        recovery_rate=round(recovery_rate, 4),
        goal_completion_rate=round(goals / n, 4),
        mean_duration_sec=round(float(np.mean(durations)), 2),
        mean_path_length_m=round(float(np.mean(path_lengths)), 2),
        mean_detection_latency_sec=round(float(np.mean(latencies)) if latencies else 0.0, 3),
        mean_time_to_safe_state_sec=round(float(np.mean(tt_safe)) if tt_safe else 0.0, 3),
        stale_acceptance_rate=round(stale_accepted / n, 4),
        mean_ai_latency_sec=round(float(np.mean(ai_lats)) if ai_lats else 0.0, 3),
    )
