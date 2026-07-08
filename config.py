from dataclasses import asdict, dataclass
from typing import Tuple
import math


@dataclass
class ForagingEnvConfig:
    world_width: int = 800
    world_height: int = 600

    robot_radius: float = 12.0
    food_radius: float = 40.0 
    success_threshold: float = 40.0 

    odor_radius: float = 140.0
    sensor_range: float = 60.0 
    num_sensors: int = 8

    max_steps: int = 300

    forward_step: float = 8.0
    reverse_step: float = 6.0
    turn_angle_deg: float = 20.0

    border_margin: float = 20.0
    min_spawn_distance: float = 120.0

    render_fps: int = 30
    background_color: Tuple[int, int, int] = (245, 245, 245)
    obstacle_color: Tuple[int, int, int] = (80, 80, 80)
    robot_color: Tuple[int, int, int] = (40, 120, 220)
    food_color: Tuple[int, int, int] = (50, 180, 80)
    sensor_color: Tuple[int, int, int] = (220, 120, 120)
    odor_color: Tuple[int, int, int] = (120, 200, 120)

    @property
    def turn_angle_rad(self) -> float:
        return math.radians(self.turn_angle_deg)

    @classmethod
    def from_dict(cls, data: dict) -> "ForagingEnvConfig":
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)
