"""
Base Autonomous Mobile Robot Navigation Controller.
Uses reactive potential fields / vector field histogram (VFH) to navigate towards goals
while repelling from perceived LiDAR obstacles and tracking odometry poses.
"""

import math
from typing import Tuple, Optional
import numpy as np
from faultshield.config import (
    MAX_LINEAR_VELOCITY,
    MAX_ANGULAR_VELOCITY,
    ROBOT_RADIUS,
)
from faultshield.core.sensors import TelemetryPacket
from faultshield.core.robot import wrap_to_pi


class NavigationController:
    def __init__(
        self,
        goal_x: float = 10.5,
        goal_y: float = 8.5,
        goal_tolerance: float = 0.5,
        k_att: float = 1.0,
        k_rep: float = 0.8,
        repulsion_threshold: float = 1.5,
    ):
        self.goal_x = float(goal_x)
        self.goal_y = float(goal_y)
        self.goal_tolerance = float(goal_tolerance)
        self.k_att = k_att
        self.k_rep = k_rep
        self.repulsion_threshold = repulsion_threshold
        self.is_goal_reached = False

    def reset_goal(self, gx: float, gy: float):
        self.goal_x = float(gx)
        self.goal_y = float(gy)
        self.is_goal_reached = False

    def compute_control(
        self,
        telemetry: TelemetryPacket
    ) -> Tuple[float, float]:
        """
        Compute (cmd_v, cmd_omega) using perceived sensor telemetry.
        Notice: Relies purely on telemetry.odometry and telemetry.lidar.
        """
        odom = telemetry.odometry
        lidar = telemetry.lidar

        dx = self.goal_x - odom.x
        dy = self.goal_y - odom.y
        dist_to_goal = math.hypot(dx, dy)

        if dist_to_goal <= self.goal_tolerance:
            self.is_goal_reached = True
            return 0.0, 0.0

        self.is_goal_reached = False

        # 1. Attractive vector towards goal
        goal_heading = math.atan2(dy, dx)
        att_fx = math.cos(goal_heading)
        att_fy = math.sin(goal_heading)

        # 2. Repulsive vector from perceived LiDAR obstacles
        rep_fx = 0.0
        rep_fy = 0.0

        if lidar.is_valid and len(lidar.ranges) > 0:
            for r, angle in zip(lidar.ranges, lidar.angles):
                if r < self.repulsion_threshold and r > 0.05:
                    # Strength increases inversely with distance
                    strength = self.k_rep * (1.0 / r - 1.0 / self.repulsion_threshold) / (r * r)
                    # Vector pointing away from obstacle
                    rep_fx -= strength * math.cos(angle)
                    rep_fy -= strength * math.sin(angle)

        # Combined resultant desired direction
        total_fx = self.k_att * att_fx + rep_fx
        total_fy = self.k_att * att_fy + rep_fy

        desired_heading = math.atan2(total_fy, total_fx)
        heading_error = wrap_to_pi(desired_heading - odom.theta)

        # Proportional heading controller
        k_omega = 2.0
        cmd_omega = k_omega * heading_error

        # Forward speed slows down during sharp turns or close obstacles
        speed_factor = max(0.0, math.cos(heading_error))
        cmd_v = MAX_LINEAR_VELOCITY * speed_factor * min(1.0, dist_to_goal / 2.0)

        # If perceived obstacle right in front is very close, stop forward motion
        if lidar.forward_distance < (ROBOT_RADIUS + 0.30):
            cmd_v = 0.0
            cmd_omega = math.copysign(MAX_ANGULAR_VELOCITY * 0.8, -rep_fy if rep_fy != 0 else 1.0)

        # Clamp output commands
        cmd_v = max(0.0, min(cmd_v, MAX_LINEAR_VELOCITY))
        cmd_omega = max(-MAX_ANGULAR_VELOCITY, min(cmd_omega, MAX_ANGULAR_VELOCITY))

        return cmd_v, cmd_omega
