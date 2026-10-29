"""
Warehouse simulation world and obstacle geometry.
Provides fast raycasting for LiDAR simulation and exact collision detection.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class RectObstacle:
    name: str
    x: float  # bottom-left x
    y: float  # bottom-left y
    w: float  # width
    h: float  # height

    def contains_point(self, px: float, py: float) -> bool:
        return (self.x <= px <= self.x + self.w) and (self.y <= py <= self.y + self.h)

    def distance_to_point(self, px: float, py: float) -> float:
        # Clamped distance from point to axis-aligned box
        cx = max(self.x, min(px, self.x + self.w))
        cy = max(self.y, min(py, self.y + self.h))
        return math.hypot(px - cx, py - cy)

    def get_segments(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        p1 = (self.x, self.y)
        p2 = (self.x + self.w, self.y)
        p3 = (self.x + self.w, self.y + self.h)
        p4 = (self.x, self.y + self.h)
        return [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]


class WarehouseMap:
    def __init__(self, width: float = 12.0, height: float = 10.0):
        self.width = width
        self.height = height
        self.obstacles: List[RectObstacle] = []
        self._segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        self._build_default_warehouse()

    def _build_default_warehouse(self):
        """Construct a realistic warehouse layout with aisles, storage racks, and central pillar."""
        # Perimeter boundary walls as segments
        w, h = self.width, self.height
        self._segments.extend([
            ((0.0, 0.0), (w, 0.0)),
            ((w, 0.0), (w, h)),
            ((w, h), (0.0, h)),
            ((0.0, h), (0.0, 0.0)),
        ])

        # Warehouse interior obstacles:
        # Racks in Aisle 1 (left)
        self.add_obstacle(RectObstacle("Rack_A1", 2.5, 1.5, 1.2, 3.0))
        self.add_obstacle(RectObstacle("Rack_A2", 2.5, 5.5, 1.2, 3.0))

        # Central Pillars / Loading Area
        self.add_obstacle(RectObstacle("Pillar_Center", 5.5, 4.0, 1.0, 2.0))

        # Racks in Aisle 2 (right)
        self.add_obstacle(RectObstacle("Rack_B1", 8.0, 1.5, 1.2, 3.0))
        self.add_obstacle(RectObstacle("Rack_B2", 8.0, 5.5, 1.2, 3.0))

    def add_obstacle(self, obstacle: RectObstacle):
        self.obstacles.append(obstacle)
        self._segments.extend(obstacle.get_segments())

    def is_collision(self, x: float, y: float, radius: float) -> bool:
        """Check if circular robot at (x, y) with given radius is in collision with walls or obstacles."""
        # Check boundary walls
        if (x - radius <= 0.0 or x + radius >= self.width or
                y - radius <= 0.0 or y + radius >= self.height):
            return True

        # Check rectangular obstacles
        for obs in self.obstacles:
            if obs.distance_to_point(x, y) <= radius:
                return True

        return False

    def distance_to_nearest_obstacle(self, x: float, y: float, radius: float = 0.0) -> float:
        """Calculate clearance distance to the closest obstacle or wall."""
        dist = min(x, self.width - x, y, self.height - y)
        for obs in self.obstacles:
            d = obs.distance_to_point(x, y)
            if d < dist:
                dist = d
        return max(0.0, dist - radius)

    @staticmethod
    def _ray_segment_intersect(
        ox: float, oy: float, dx: float, dy: float,
        p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> Optional[float]:
        """Compute distance along ray (ox, oy) + t*(dx, dy) to line segment p1->p2."""
        x1, y1 = p1
        x2, y2 = p2
        sx = x2 - x1
        sy = y2 - y1

        denom = dx * sy - dy * sx
        if abs(denom) < 1e-9:
            return None  # Parallel or collinear

        t = ((x1 - ox) * sy - (y1 - oy) * sx) / denom
        u = ((x1 - ox) * dy - (y1 - oy) * dx) / denom

        if t >= 0.0 and 0.0 <= u <= 1.0:
            return t
        return None

    def cast_ray(self, ox: float, oy: float, angle_rad: float, max_range: float = 6.0) -> float:
        """Cast a single ray from (ox, oy) at angle_rad, returning distance to closest intersection."""
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

        min_t = max_range
        for p1, p2 in self._segments:
            t = self._ray_segment_intersect(ox, oy, dx, dy, p1, p2)
            if t is not None and 0.0 < t < min_t:
                min_t = t

        return min_t

    def cast_lidar(
        self,
        ox: float,
        oy: float,
        robot_heading_rad: float,
        num_beams: int = 24,
        fov_rad: float = math.radians(270.0),
        max_range: float = 6.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cast simulated LiDAR fan centered around robot_heading_rad.
        Returns:
            ranges: np.ndarray of shape (num_beams,)
            angles: np.ndarray of shape (num_beams,) (absolute angles in world frame)
        """
        half_fov = fov_rad / 2.0
        rel_angles = np.linspace(-half_fov, half_fov, num_beams)
        abs_angles = robot_heading_rad + rel_angles
        ranges = np.zeros(num_beams, dtype=np.float64)

        for i, a in enumerate(abs_angles):
            ranges[i] = self.cast_ray(ox, oy, a, max_range)

        return ranges, abs_angles
