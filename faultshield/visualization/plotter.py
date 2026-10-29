"""
Interactive Plotly visualizations for FaultShield:
- 2D Warehouse floor plan with obstacles, ground-truth path, and drifted odometry
- Dynamic telemetry & fault incident timeline
- 4-Variant comparative benchmark charts
"""

import math
from typing import List, Dict, Any, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from faultshield.core.world import WarehouseMap
from faultshield.benchmark.runner import StepTrace, EpisodeRunResult
from faultshield.benchmark.metrics import BenchmarkSummary


def create_warehouse_figure(
    world: WarehouseMap,
    traces: List[StepTrace],
    goal_pos: tuple,
    title: str = "Robot Trajectory & Warehouse Map",
    show_lidar_step: Optional[int] = None,
) -> go.Figure:
    fig = go.Figure()

    # 1. Warehouse Obstacles (Racks & Pillars)
    for obs in world.obstacles:
        fig.add_shape(
            type="rect",
            x0=obs.x,
            y0=obs.y,
            x1=obs.x + obs.w,
            y1=obs.y + obs.h,
            line=dict(color="#3b82f6", width=2),
            fillcolor="rgba(59, 130, 246, 0.25)",
        )
        fig.add_annotation(
            x=obs.x + obs.w / 2.0,
            y=obs.y + obs.h / 2.0,
            text=obs.name,
            showarrow=False,
            font=dict(size=10, color="#94a3b8"),
        )

    # 2. Outer Perimeter Walls
    fig.add_shape(
        type="rect",
        x0=0.0,
        y0=0.0,
        x1=world.width,
        y1=world.height,
        line=dict(color="#64748b", width=3),
        fillcolor="rgba(15, 23, 42, 0.05)",
    )

    if traces:
        gt_x = [st.x_gt for st in traces]
        gt_y = [st.y_gt for st in traces]
        odom_x = [st.x_odom for st in traces]
        odom_y = [st.y_odom for st in traces]

        # 3. Ground Truth Path (Solid cyan/emerald)
        fig.add_trace(
            go.Scatter(
                x=gt_x,
                y=gt_y,
                mode="lines",
                name="Ground Truth Trajectory",
                line=dict(color="#06b6d4", width=3),
                hoverinfo="text",
                hovertext=[f"t: {st.t:.2f}s | Clear: {st.min_clearance:.2f}m" for st in traces],
            )
        )

        # 4. Perceived Drifted Odometry Path (Dashed orange/amber)
        fig.add_trace(
            go.Scatter(
                x=odom_x,
                y=odom_y,
                mode="lines",
                name="Perceived Odometry",
                line=dict(color="#f59e0b", width=2, dash="dot"),
            )
        )

        # 5. Start Position Marker
        fig.add_trace(
            go.Scatter(
                x=[gt_x[0]],
                y=[gt_y[0]],
                mode="markers+text",
                name="Start",
                marker=dict(size=14, color="#10b981", symbol="circle"),
                text=["START"],
                textposition="bottom center",
            )
        )

        # 6. Final Robot Position Marker & Orientation Arrow
        last_st = traces[-1]
        fig.add_trace(
            go.Scatter(
                x=[last_st.x_gt],
                y=[last_st.y_gt],
                mode="markers",
                name="Robot Final Pose",
                marker=dict(
                    size=16,
                    color="#ef4444" if last_st.is_collided else "#3b82f6",
                    symbol="diamond",
                ),
            )
        )

        # Heading direction vector
        arrow_len = 0.6
        hx = last_st.x_gt + arrow_len * math.cos(last_st.theta_gt)
        hy = last_st.y_gt + arrow_len * math.sin(last_st.theta_gt)
        fig.add_annotation(
            x=hx,
            y=hy,
            ax=last_st.x_gt,
            ay=last_st.y_gt,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor="#ef4444" if last_st.is_collided else "#3b82f6",
        )

        # Collision explosion marker if collided
        if last_st.is_collided:
            fig.add_trace(
                go.Scatter(
                    x=[last_st.x_gt],
                    y=[last_st.y_gt],
                    mode="markers+text",
                    name="COLLISION",
                    marker=dict(size=24, color="#dc2626", symbol="x"),
                    text=["COLLISION!"],
                    textposition="top center",
                    textfont=dict(color="#dc2626", size=14, family="Arial Black"),
                )
            )

        # 6b. Dynamic LiDAR Ray Fan at show_lidar_step
        if show_lidar_step is not None and 0 <= show_lidar_step < len(traces):
            curr_st = traces[show_lidar_step]
            # Draw robot current scrubbed pose
            fig.add_trace(
                go.Scatter(
                    x=[curr_st.x_gt],
                    y=[curr_st.y_gt],
                    mode="markers",
                    name="Current Scrubbed Pose",
                    marker=dict(
                        size=18,
                        color="#06b6d4" if not curr_st.is_collided else "#dc2626",
                        symbol="circle",
                        line=dict(color="#ffffff", width=2),
                    ),
                    hoverinfo="text",
                    hovertext=f"t: {curr_st.t:.2f}s | Clear: {curr_st.min_clearance:.2f}m",
                )
            )
            if curr_st.lidar_ranges and curr_st.lidar_angles:
                ray_x, ray_y = [], []
                for r, ang in zip(curr_st.lidar_ranges, curr_st.lidar_angles):
                    ray_x.extend([curr_st.x_gt, curr_st.x_gt + r * math.cos(ang), None])
                    ray_y.extend([curr_st.y_gt, curr_st.y_gt + r * math.sin(ang), None])

                beam_color = "rgba(239, 68, 68, 0.50)" if curr_st.fault_active else "rgba(16, 185, 129, 0.40)"
                beam_label = "Corrupted / Frozen LiDAR" if curr_st.fault_active else "Healthy LiDAR Beams"
                fig.add_trace(
                    go.Scatter(
                        x=ray_x,
                        y=ray_y,
                        mode="lines",
                        name=beam_label,
                        line=dict(color=beam_color, width=1.5),
                        hoverinfo="none",
                    )
                )

    # 7. Goal Position Target
    gx, gy = goal_pos
    fig.add_trace(
        go.Scatter(
            x=[gx],
            y=[gy],
            mode="markers+text",
            name="Target Goal",
            marker=dict(size=16, color="#8b5cf6", symbol="star"),
            text=["GOAL"],
            textposition="top center",
            textfont=dict(color="#8b5cf6", size=12),
        )
    )

    # Layout styling with clean dark-mode friendly palette
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#f8fafc")),
        xaxis=dict(
            title="Warehouse X (meters)",
            range=[-0.5, world.width + 0.5],
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.15)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Warehouse Y (meters)",
            range=[-0.5, world.height + 0.5],
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.15)",
            scaleanchor="x",
            scaleratio=1,
            zeroline=False,
        ),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#e2e8f0"),
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=540,
    )

    return fig


def create_telemetry_timeline_figure(traces: List[StepTrace]) -> go.Figure:
    if not traces:
        return go.Figure()

    times = [st.t for st in traces]
    v_vals = [st.v for st in traces]
    omega_vals = [st.omega for st in traces]
    clearance_vals = [st.min_clearance for st in traces]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Obstacle Clearance & Safety Margin", "Linear & Angular Velocity"),
    )

    # 1. Clearance Plot
    fig.add_trace(
        go.Scatter(
            x=times,
            y=clearance_vals,
            mode="lines",
            name="Min Obstacle Clearance (m)",
            line=dict(color="#06b6d4", width=2.5),
        ),
        row=1,
        col=1,
    )

    # Safety critical threshold reference line
    fig.add_hline(
        y=0.42,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="Critical Collision Boundary (0.42m)",
        annotation_position="bottom right",
        row=1,
        col=1,
    )

    # Highlight fault active intervals
    fault_times = [st.t for st in traces if st.fault_active]
    if fault_times:
        fig.add_vrect(
            x0=min(fault_times),
            x1=max(fault_times),
            fillcolor="rgba(239, 68, 68, 0.12)",
            line_width=0,
            annotation_text="Fault Active",
            annotation_position="top left",
            row=1,
            col=1,
        )

    # 2. Velocities Plot
    fig.add_trace(
        go.Scatter(
            x=times,
            y=v_vals,
            mode="lines",
            name="Linear Velocity v (m/s)",
            line=dict(color="#10b981", width=2),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=omega_vals,
            mode="lines",
            name="Angular Velocity ω (rad/s)",
            line=dict(color="#a855f7", width=1.5, dash="dot"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Simulation Time (seconds)", row=2, col=1, gridcolor="rgba(148, 163, 184, 0.15)")
    fig.update_yaxes(title_text="Meters", row=1, col=1, gridcolor="rgba(148, 163, 184, 0.15)")
    fig.update_yaxes(title_text="Velocity", row=2, col=1, gridcolor="rgba(148, 163, 184, 0.15)")

    return fig


def create_variant_comparison_chart(summaries: List[BenchmarkSummary]) -> go.Figure:
    """Generate comparative grouped bar chart for Variants A, B, C, D."""
    variants = [s.variant_id for s in summaries]
    collision_rates = [s.collision_rate * 100 for s in summaries]
    recovery_rates = [s.recovery_rate * 100 for s in summaries]
    completion_rates = [s.goal_completion_rate * 100 for s in summaries]

    fig = go.Figure(
        data=[
            go.Bar(name="Collision Rate (%)", x=variants, y=collision_rates, marker_color="#ef4444"),
            go.Bar(name="Recovery Rate (%)", x=variants, y=recovery_rates, marker_color="#10b981"),
            go.Bar(name="Goal Completion (%)", x=variants, y=completion_rates, marker_color="#3b82f6"),
        ]
    )

    fig.update_layout(
        barmode="group",
        title="Cross-Variant Reliability Comparison (A vs B vs C vs D)",
        xaxis_title="System Variant",
        yaxis_title="Percentage (%)",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
