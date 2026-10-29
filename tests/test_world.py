import pytest
from faultshield.core.world import WarehouseMap, RectObstacle


def test_warehouse_initialization():
    world = WarehouseMap(width=12.0, height=10.0)
    assert world.width == 12.0
    assert world.height == 10.0
    assert len(world.obstacles) >= 5


def test_collision_detection():
    world = WarehouseMap()
    # Center of Rack_A1 is around x=3.1, y=3.0 (Rack_A1: x=2.5, y=1.5, w=1.2, h=3.0)
    assert world.is_collision(3.0, 3.0, radius=0.35) is True

    # Free corridor point
    assert world.is_collision(1.5, 1.5, radius=0.35) is False

    # Out of warehouse bounds
    assert world.is_collision(-0.1, 1.5, radius=0.35) is True
    assert world.is_collision(1.5, 10.2, radius=0.35) is True


def test_lidar_raycast():
    world = WarehouseMap()
    ranges, angles = world.cast_lidar(ox=1.5, oy=1.5, robot_heading_rad=0.0, num_beams=12)
    assert len(ranges) == 12
    assert len(angles) == 12
    # Ranges must be strictly positive and <= max range
    assert (ranges > 0.0).all()
    assert (ranges <= 6.0).all()
