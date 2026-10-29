"""Interactive 3D replay visualization for FaultShield trajectories.

This module visualizes the existing kinematic simulation in a 3D scene. It is
not a full 3D physics engine; robot motion and collisions are computed by
simulation.py / runner.py and replayed here as a 3D digital-twin view.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go


def _box_faces() -> tuple[list[int], list[int], list[int]]:
    # 8 cuboid vertices. Each quad face is represented by two triangles.
    triangles = [
        (0, 1, 2), (0, 2, 3),  # bottom
        (4, 6, 5), (4, 7, 6),  # top
        (0, 4, 5), (0, 5, 1),  # front
        (1, 5, 6), (1, 6, 2),  # right
        (2, 6, 7), (2, 7, 3),  # back
        (3, 7, 4), (3, 4, 0),  # left
    ]
    return ([t[0] for t in triangles], [t[1] for t in triangles], [t[2] for t in triangles])


def _robot_body_vertices(x: float, y: float, heading: float) -> tuple[list[float], list[float], list[float]]:
    length, width = 0.66, 0.42
    z0, z1 = 0.08, 0.32
    local = [
        (-length / 2, -width / 2, z0), (length / 2, -width / 2, z0),
        (length / 2, width / 2, z0), (-length / 2, width / 2, z0),
        (-length / 2, -width / 2, z1), (length / 2, -width / 2, z1),
        (length / 2, width / 2, z1), (-length / 2, width / 2, z1),
    ]
    return _transform(local, x, y, heading)


def _wheel_vertices(x: float, y: float, heading: float, side: int) -> tuple[list[float], list[float], list[float]]:
    radius, half_width, axle_y, center_z = 0.135, 0.045, 0.245, 0.155
    segments = 14
    local: list[tuple[float, float, float]] = []
    for yy in (side * axle_y - half_width, side * axle_y + half_width):
        for i in range(segments):
            angle = 2.0 * math.pi * i / segments
            local.append((radius * math.cos(angle), yy, center_z + radius * math.sin(angle)))
    return _transform(local, x, y, heading)


def _transform(local: list[tuple[float, float, float]], x: float, y: float, heading: float) -> tuple[list[float], list[float], list[float]]:
    c, s = math.cos(heading), math.sin(heading)
    xs, ys, zs = [], [], []
    for lx, ly, lz in local:
        xs.append(x + lx * c - ly * s)
        ys.append(y + lx * s + ly * c)
        zs.append(lz)
    return xs, ys, zs


def _wheel_faces(segments: int = 14) -> tuple[list[int], list[int], list[int]]:
    i_faces: list[int] = []
    j_faces: list[int] = []
    k_faces: list[int] = []
    # Side walls between the two circular rings.
    for n in range(segments):
        nxt = (n + 1) % segments
        i_faces.extend([n, nxt, n])
        j_faces.extend([nxt, segments + nxt, segments + n])
        k_faces.extend([segments + nxt, segments + n, nxt])
    # End caps as triangle fans.
    for n in range(1, segments - 1):
        # Bottom and top end-cap triangles.
        i_faces.extend([0, segments])
        j_faces.extend([n + 1, segments + n])
        k_faces.extend([n, segments + n + 1])
    return i_faces, j_faces, k_faces


def _cylinder_mesh(cx: float, cy: float, radius: float, height: float, segments: int = 32) -> go.Mesh3d:
    xs, ys, zs = [], [], []
    for z in (0.0, height):
        for n in range(segments):
            a = 2 * math.pi * n / segments
            xs.append(cx + radius * math.cos(a))
            ys.append(cy + radius * math.sin(a))
            zs.append(z)
    ii: list[int] = []
    jj: list[int] = []
    kk: list[int] = []
    for n in range(segments):
        nxt = (n + 1) % segments
        ii.extend([n, nxt, n])
        jj.extend([nxt, segments + nxt, segments + n])
        kk.extend([segments + nxt, segments + n, nxt])
    for n in range(1, segments - 1):
        # Bottom and top end-cap triangles.
        ii.extend([0, segments])
        jj.extend([n, segments + n + 1])
        kk.extend([n + 1, segments + n])
    return go.Mesh3d(
        x=xs, y=ys, z=zs, i=ii, j=jj, k=kk, color="#ef4444", opacity=0.88,
        name="Obstacle (ground truth)", hovertext="Fixed obstacle", hoverinfo="text", flatshading=True
    )


def _cuboid_mesh(x0: float, y0: float, w: float, h: float, z_height: float, name: str, color: str = "#3b82f6") -> go.Mesh3d:
    xs = [x0, x0 + w, x0 + w, x0, x0, x0 + w, x0 + w, x0]
    ys = [y0, y0, y0 + h, y0 + h, y0, y0, y0 + h, y0 + h]
    zs = [0.0, 0.0, 0.0, 0.0, z_height, z_height, z_height, z_height]
    bi, bj, bk = _box_faces()
    return go.Mesh3d(
        x=xs, y=ys, z=zs, i=bi, j=bj, k=bk, color=color, opacity=0.80,
        name=name, hovertext=name, hoverinfo="text", flatshading=True
    )


def _heading_at(path: list[dict[str, float]], index: int) -> float:
    if len(path) < 2:
        return 0.0
    if index < len(path) - 1:
        p0, p1 = path[index], path[index + 1]
    else:
        p0, p1 = path[index - 1], path[index]
    dx, dy = p1["x"] - p0["x"], p1["y"] - p0["y"]
    return math.atan2(dy, dx) if abs(dx) + abs(dy) > 1e-9 else 0.0


def _path_point(path: list[dict[str, float]], index: int) -> dict[str, float]:
    return path[min(index, len(path) - 1)]


def _robot_meshes(point: dict[str, float], heading: float, name_prefix: str, body_color: str, wheel_color: str) -> list[go.Mesh3d]:
    bx, by, bz = _robot_body_vertices(point["x"], point["y"], heading)
    bi, bj, bk = _box_faces()
    wi, wj, wk = _wheel_faces()
    out = [go.Mesh3d(
        x=bx, y=by, z=bz, i=bi, j=bj, k=bk, color=body_color,
        name=f"{name_prefix} chassis", flatshading=True, hoverinfo="skip"
    )]
    for side, label in ((-1, "right wheel"), (1, "left wheel")):
        wx, wy, wz = _wheel_vertices(point["x"], point["y"], heading, side)
        out.append(go.Mesh3d(
            x=wx, y=wy, z=wz, i=wi, j=wj, k=wk, color=wheel_color,
            name=f"{name_prefix} {label}", flatshading=True, hoverinfo="skip"
        ))
    return out


def build_3d_replay(baseline: Any, guarded: Any) -> go.Figure:
    """Build an animated 3D view of existing 2D simulation paths."""
    base_path = baseline.path
    guard_path = guarded.path
    total_frames = max(len(base_path), len(guard_path), 2)

    fig = go.Figure()

    # Ground plane.
    gx = np.linspace(-1.0, 11.0, 25)
    gy = np.linspace(-3.5, 3.5, 19)
    Gx, Gy = np.meshgrid(gx, gy)
    Gz = np.zeros_like(Gx)
    fig.add_trace(go.Surface(
        x=Gx, y=Gy, z=Gz, surfacecolor=np.zeros_like(Gz),
        colorscale=[[0, "#dce5ee"], [1, "#dce5ee"]], showscale=False,
        opacity=0.38, hoverinfo="skip", name="Ground"
    ))

    # Floor grid, at z=0.015 to avoid z fighting.
    for xg in range(-1, 12):
        fig.add_trace(go.Scatter3d(
            x=[xg, xg], y=[-3.5, 3.5], z=[0.015, 0.015], mode="lines",
            line={"color": "#c7d2de", "width": 1}, showlegend=False, hoverinfo="skip"
        ))
    for yg in range(-3, 4):
        fig.add_trace(go.Scatter3d(
            x=[-1, 11], y=[yg, yg], z=[0.015, 0.015], mode="lines",
            line={"color": "#c7d2de", "width": 1}, showlegend=False, hoverinfo="skip"
        ))

    # Static trajectories.
    fig.add_trace(go.Scatter3d(
        x=[p["x"] for p in base_path], y=[p["y"] for p in base_path],
        z=[0.04] * len(base_path), mode="lines", name="Baseline trajectory",
        line={"color": "#64748b", "width": 5, "dash": "dash"}
    ))
    fig.add_trace(go.Scatter3d(
        x=[p["x"] for p in guard_path], y=[p["y"] for p in guard_path],
        z=[0.045] * len(guard_path), mode="lines", name="FaultShield trajectory",
        line={"color": "#0d9488", "width": 7}
    ))
    fig.add_trace(_cylinder_mesh(5.0, 0.0, 0.58, 1.1))
    fig.add_trace(go.Scatter3d(
        x=[0, 10], y=[0, 0], z=[0.08, 0.08], mode="markers+text",
        text=["START", "GOAL"], textposition="top center", name="Start / Goal",
        marker={"size": 6, "color": ["#2563eb", "#16a34a"], "symbol": "diamond"}
    ))

    # Animated robot geometry (chassis + two wheels for each robot).
    b0 = _path_point(base_path, 0)
    g0 = _path_point(guard_path, 0)
    base_robot_traces = _robot_meshes(b0, _heading_at(base_path, 0), "Baseline", "#94a3b8", "#334155")
    guard_robot_traces = _robot_meshes(g0, _heading_at(guard_path, 0), "FaultShield", "#14b8a6", "#134e4a")
    robot_trace_indices: list[int] = []
    for trace in base_robot_traces + guard_robot_traces:
        robot_trace_indices.append(len(fig.data))
        fig.add_trace(trace)

    frames = []
    for step in range(total_frames):
        bp = _path_point(base_path, step)
        gp = _path_point(guard_path, step)
        b_meshes = _robot_meshes(bp, _heading_at(base_path, min(step, len(base_path) - 1)), "Baseline", "#94a3b8", "#334155")
        g_meshes = _robot_meshes(gp, _heading_at(guard_path, min(step, len(guard_path) - 1)), "FaultShield", "#14b8a6", "#134e4a")
        frame_data = b_meshes + g_meshes
        frames.append(go.Frame(
            name=str(step), data=frame_data, traces=robot_trace_indices,
            layout=go.Layout(title_text=f"3D robot replay — step {step + 1}/{total_frames}")
        ))
    fig.frames = frames

    slider_steps = []
    # Use at most ~80 slider labels so the UI stays compact.
    stride = max(1, total_frames // 80)
    labels = list(range(0, total_frames, stride))
    if labels[-1] != total_frames - 1:
        labels.append(total_frames - 1)
    for step in labels:
        slider_steps.append({
            "args": [[str(step)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
            "label": str(step + 1), "method": "animate",
        })

    fig.update_layout(
        title={"text": f"3D robot replay — step 1/{total_frames}", "x": 0.01, "y": 0.97},
        height=540,
        margin={"l": 0, "r": 0, "t": 70, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0.22},
        scene={
            "xaxis": {"title": "X (m)", "range": [-1, 11], "backgroundcolor": "#f8fafc", "gridcolor": "#cbd5e1"},
            "yaxis": {"title": "Y (m)", "range": [-3.5, 3.5], "backgroundcolor": "#f8fafc", "gridcolor": "#cbd5e1"},
            "zaxis": {"title": "Height (m)", "range": [0, 2.2], "backgroundcolor": "#f8fafc", "gridcolor": "#cbd5e1"},
            "aspectmode": "manual", "aspectratio": {"x": 2.1, "y": 1.25, "z": 0.65},
            "camera": {"eye": {"x": 1.35, "y": -1.55, "z": 1.05}},
        },
        updatemenus=[{
            "type": "buttons", "direction": "left", "showactive": False, "x": 0.0, "y": 1.08, "xanchor": "left", "yanchor": "top",
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 75, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True, "mode": "immediate"}]},
                {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
            ],
        }],
        sliders=[{"active": 0, "x": 0.08, "y": 0.02, "len": 0.88, "currentvalue": {"prefix": "Step: ", "visible": True}, "steps": slider_steps}],
    )
    return fig


def build_3d_warehouse_replay(
    world: Any,
    baseline_traces: list[Any],
    guarded_traces: list[Any],
    goal_pos: tuple[float, float],
) -> go.Figure:
    """Build an animated 3D digital-twin replay of the multi-rack warehouse simulation."""
    base_path = [{"x": st.x_gt, "y": st.y_gt, "theta": st.theta_gt} for st in baseline_traces] or [{"x": 1.5, "y": 1.5, "theta": 0.0}]
    guard_path = [{"x": st.x_gt, "y": st.y_gt, "theta": st.theta_gt} for st in guarded_traces] or [{"x": 1.5, "y": 1.5, "theta": 0.0}]

    # Downsample slightly if traces are long so browser animation stays fluid
    stride_pts = max(1, max(len(base_path), len(guard_path)) // 60)
    base_anim = base_path[::stride_pts]
    if base_anim[-1] != base_path[-1]:
        base_anim.append(base_path[-1])
    guard_anim = guard_path[::stride_pts]
    if guard_anim[-1] != guard_path[-1]:
        guard_anim.append(guard_path[-1])

    total_frames = max(len(base_anim), len(guard_anim), 2)
    fig = go.Figure()

    # Ground plane
    gx = np.linspace(0.0, world.width, 25)
    gy = np.linspace(0.0, world.height, 21)
    Gx, Gy = np.meshgrid(gx, gy)
    Gz = np.zeros_like(Gx)
    fig.add_trace(go.Surface(
        x=Gx, y=Gy, z=Gz, surfacecolor=np.zeros_like(Gz),
        colorscale=[[0, "#dce5ee"], [1, "#dce5ee"]], showscale=False,
        opacity=0.38, hoverinfo="skip", name="Warehouse Floor"
    ))

    # Floor grid
    for xg in range(0, int(world.width) + 1):
        fig.add_trace(go.Scatter3d(
            x=[xg, xg], y=[0, world.height], z=[0.015, 0.015], mode="lines",
            line={"color": "#c7d2de", "width": 1}, showlegend=False, hoverinfo="skip"
        ))
    for yg in range(0, int(world.height) + 1):
        fig.add_trace(go.Scatter3d(
            x=[0, world.width], y=[yg, yg], z=[0.015, 0.015], mode="lines",
            line={"color": "#c7d2de", "width": 1}, showlegend=False, hoverinfo="skip"
        ))

    # 3D Warehouse Obstacles
    for obs in world.obstacles:
        color = "#ef4444" if "Pillar" in obs.name else "#3b82f6"
        fig.add_trace(_cuboid_mesh(obs.x, obs.y, obs.w, obs.h, 1.2, obs.name, color=color))

    # Static trajectories
    fig.add_trace(go.Scatter3d(
        x=[p["x"] for p in base_path], y=[p["y"] for p in base_path],
        z=[0.04] * len(base_path), mode="lines", name="Baseline (Variant B) trajectory",
        line={"color": "#64748b", "width": 5, "dash": "dash"}
    ))
    fig.add_trace(go.Scatter3d(
        x=[p["x"] for p in guard_path], y=[p["y"] for p in guard_path],
        z=[0.045] * len(guard_path), mode="lines", name="FaultShield (Variant D) trajectory",
        line={"color": "#0d9488", "width": 7}
    ))

    start_pt = guard_path[0]
    fig.add_trace(go.Scatter3d(
        x=[start_pt["x"], goal_pos[0]], y=[start_pt["y"], goal_pos[1]], z=[0.08, 0.08],
        mode="markers+text", text=["START", "GOAL"], textposition="top center", name="Start / Goal",
        marker={"size": 6, "color": ["#2563eb", "#16a34a"], "symbol": "diamond"}
    ))

    b0 = _path_point(base_anim, 0)
    g0 = _path_point(guard_anim, 0)
    base_robot_traces = _robot_meshes(b0, b0.get("theta", 0.0), "Baseline", "#94a3b8", "#334155")
    guard_robot_traces = _robot_meshes(g0, g0.get("theta", 0.0), "FaultShield", "#14b8a6", "#134e4a")
    robot_trace_indices: list[int] = []
    for trace in base_robot_traces + guard_robot_traces:
        robot_trace_indices.append(len(fig.data))
        fig.add_trace(trace)

    frames = []
    for step in range(total_frames):
        bp = _path_point(base_anim, step)
        gp = _path_point(guard_anim, step)
        b_meshes = _robot_meshes(bp, bp.get("theta", 0.0), "Baseline", "#94a3b8", "#334155")
        g_meshes = _robot_meshes(gp, gp.get("theta", 0.0), "FaultShield", "#14b8a6", "#134e4a")
        frames.append(go.Frame(
            name=str(step), data=b_meshes + g_meshes, traces=robot_trace_indices,
            layout=go.Layout(title_text=f"3D warehouse replay — step {step + 1}/{total_frames}")
        ))
    fig.frames = frames

    slider_steps = [
        {
            "args": [[str(step)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
            "label": str(step + 1), "method": "animate",
        }
        for step in range(total_frames)
    ]

    fig.update_layout(
        title={"text": f"3D warehouse replay — step 1/{total_frames}", "x": 0.01, "y": 0.97},
        height=540,
        margin={"l": 0, "r": 0, "t": 70, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0.22},
        scene={
            "xaxis": {"title": "Warehouse X (m)", "range": [0, world.width], "backgroundcolor": "#f8fafc", "gridcolor": "#cbd5e1"},
            "yaxis": {"title": "Warehouse Y (m)", "range": [0, world.height], "backgroundcolor": "#f8fafc", "gridcolor": "#cbd5e1"},
            "zaxis": {"title": "Height (m)", "range": [0, 2.5], "backgroundcolor": "#f8fafc", "gridcolor": "#cbd5e1"},
            "aspectmode": "manual", "aspectratio": {"x": 1.8, "y": 1.5, "z": 0.55},
            "camera": {"eye": {"x": 1.35, "y": -1.45, "z": 1.05}},
        },
        updatemenus=[{
            "type": "buttons", "direction": "left", "showactive": False, "x": 0.0, "y": 1.08, "xanchor": "left", "yanchor": "top",
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 75, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True, "mode": "immediate"}]},
                {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
            ],
        }],
        sliders=[{"active": 0, "x": 0.08, "y": 0.02, "len": 0.88, "currentvalue": {"prefix": "Step: ", "visible": True}, "steps": slider_steps}],
    )
    return fig

