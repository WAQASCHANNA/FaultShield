import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation import run_simulation


def test_healthy_run_reaches_goal_without_collision():
    baseline = run_simulation("Healthy sensors", guarded=False)
    shielded = run_simulation("Healthy sensors", guarded=True)
    assert baseline.reached_goal
    assert shielded.reached_goal
    assert not baseline.collision
    assert not shielded.collision


def test_stale_lidar_hurts_baseline_but_guard_detects_staleness():
    baseline = run_simulation("Stale LiDAR", guarded=False)
    shielded = run_simulation("Stale LiDAR", guarded=True, ai_action="BYPASS_LEFT")
    assert baseline.collision
    assert not shielded.collision
    assert any("stale LiDAR detected" in event for event in shielded.events)


def test_invalid_ai_action_falls_back_safely():
    result = run_simulation("Stale LiDAR", guarded=True, ai_action="MOVE_FAST")
    assert not result.collision
    assert result.accepted_ai_action == "BYPASS_LEFT"


def test_combined_fault_discards_delayed_recommendation():
    result = run_simulation("Stale LiDAR + delayed AI response", guarded=True, ai_action="CONTINUE")
    assert not result.collision
    assert any("will not wait for it" in event for event in result.events)
    assert result.accepted_ai_action == "LOCAL_SAFE_FALLBACK"

