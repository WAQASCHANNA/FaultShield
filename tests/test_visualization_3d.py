import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation import run_simulation
from visualization_3d import build_3d_replay


def test_3d_replay_contains_scene_and_animation_frames():
    baseline = run_simulation("Stale LiDAR", guarded=False)
    guarded = run_simulation("Stale LiDAR", guarded=True, ai_action="BYPASS_LEFT")
    fig = build_3d_replay(baseline, guarded)
    assert len(fig.data) >= 20
    assert len(fig.frames) >= 2
    assert len(fig.frames[0].data) == 6  # two chassis and four wheels

