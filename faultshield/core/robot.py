"""
Robot Kinematic Model (Ground Truth).
Implements differential-drive motion dynamics, acceleration limits, and trajectory tracking.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple
from faultshield.config import (
    ROBOT_RADIUS,
    MAX_LINEAR_VELOCITY,
    MAX_ANGULAR_VELOCITY,
    MAX_LINEAR_ACCELERATION,
    MAX_ANGULAR_ACCELERATION,
    SIMULATION_DT,
)


def wrap_to_pi(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


@dataclass
class RobotState:
    t: float
    x: float
    y: float
    theta: float
    v: float
    omega: float
    is_collided: bool = False


class DifferentialDriveRobot:
    def __init__(
        self,
        init_x: float = 1.5,
        init_y: float = 1.5,
        init_theta: float = 0.0,
        radius: float = ROBOT_RADIUS,
    ):
        self.radius = radius
        self.reset(init_x, init_y, init_theta)

    def reset(self, x: float = 1.5, y: float = 1.5, theta: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.theta = wrap_to_pi(float(theta))
        self.v = 0.0
        self.omega = 0.0
        self.time = 0.0
        self.is_collided = False
        self.trajectory: List[RobotState] = [self.get_state()]

    def get_state(self) -> RobotState:
        return RobotState(
            t=self.time,
            x=self.x,
            y=self.y,
            theta=self.theta,
            v=self.v,
            omega=self.omega,
            is_collided=self.is_collided,
        )

    def step(
        self,
        cmd_v: float,
        cmd_omega: float,
        dt: float = SIMULATION_DT,
    ) -> RobotState:
        """Advance robot ground-truth state by dt with acceleration and velocity clamping."""
        if self.is_collided:
            # Robot cannot move once immobilized by collision
            self.v = 0.0
            self.omega = 0.0
            self.time += dt
            state = self.get_state()
            self.trajectory.append(state)
            return state

        # Clamp commanded targets to absolute maximum limits
        target_v = max(-MAX_LINEAR_VELOCITY, min(cmd_v, MAX_LINEAR_VELOCITY))
        target_omega = max(-MAX_ANGULAR_VELOCITY, min(cmd_omega, MAX_ANGULAR_VELOCITY))

        # Acceleration limiting
        dv_max = MAX_LINEAR_ACCELERATION * dt
        dv = target_v - self.v
        dv_clamped = max(-dv_max, min(dv, dv_max))
        self.v += dv_clamped

        domega_max = MAX_ANGULAR_ACCELERATION * dt
        domega = target_omega - self.omega
        domega_clamped = max(-domega_max, min(domega, domega_max))
        self.omega += domega_clamped

        # Kinematics integration (midpoint heading)
        theta_mid = wrap_to_pi(self.theta + 0.5 * self.omega * dt)
        self.x += self.v * math.cos(theta_mid) * dt
        self.y += self.v * math.sin(theta_mid) * dt
        self.theta = wrap_to_pi(self.theta + self.omega * dt)
        self.time += dt

        state = self.get_state()
        self.trajectory.append(state)
        return state

    def mark_collision(self):
        self.is_collided = True
        self.v = 0.0
        self.omega = 0.0
        if self.trajectory:
            self.trajectory[-1].is_collided = True
