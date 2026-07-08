import math
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from config import ForagingEnvConfig


class ForagingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: Optional[ForagingEnvConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.config = config or ForagingEnvConfig()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.config.num_sensors + 1,),
            dtype=np.float32,
        )

        self.robot_pos = np.zeros(2, dtype=np.float32)
        self.robot_theta = 0.0
        self.food_pos = np.zeros(2, dtype=np.float32)
        self.step_count = 0

        self.obstacles = self._build_default_obstacles()

        self.window = None
        self.clock = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.step_count = 0

        self.robot_pos, self.robot_theta = self._sample_robot_pose()
        self.food_positions = [self._sample_food_position() for _ in range(2)] 

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        self.step_count += 1
        self._apply_action(action)

        obs = self._get_obs()
        reward = 1.0 if self._is_success() else 0.0
        terminated = self._is_success()
        truncated = self.step_count >= self.config.max_steps
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                self.window = pygame.display.set_mode((self.config.world_width, self.config.world_height))
                pygame.display.set_caption("ForagingEnv")
            else:
                self.window = pygame.Surface((self.config.world_width, self.config.world_height))
            self.clock = pygame.time.Clock()

        canvas = self.window
        canvas.fill(self.config.background_color)

        self._draw_odor(canvas)
        self._draw_obstacles(canvas)
        self._draw_food(canvas)
        self._draw_robot(canvas)
        self._draw_sensors(canvas)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.config.render_fps)
            return None

        rgb = pygame.surfarray.array3d(canvas)
        return np.transpose(rgb, (1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

    def _get_obs(self) -> np.ndarray:
        prox = self._compute_proximity_readings()
        odor = self._compute_odor()
        obs = np.array(prox + [odor], dtype=np.float32)
        return obs

    def _get_info(self) -> dict:
        return {
            "step_count": self.step_count,
            "distance_to_food": float(min(np.linalg.norm(self.robot_pos - f) for f in self.food_positions)), 
            "success": self._is_success(),
            "robot_position": self.robot_pos.copy(),
        }

    def _apply_action(self, action: int):
        candidate_pos = self.robot_pos.copy()
        candidate_theta = self.robot_theta

        if action == 0:
            candidate_pos = self.robot_pos + self._heading_vector(self.robot_theta) * self.config.forward_step
        elif action == 1:
            candidate_theta = self.robot_theta - self.config.turn_angle_rad
        elif action == 2:
            candidate_theta = self.robot_theta + self.config.turn_angle_rad
        elif action == 3:
            candidate_pos = self.robot_pos - self._heading_vector(self.robot_theta) * self.config.reverse_step

        if action in (0, 3):
            if not self._collides(candidate_pos):
                self.robot_pos = candidate_pos.astype(np.float32)
        else:
            self.robot_theta = self._wrap_angle(candidate_theta)

    def _compute_odor(self) -> float:
        d = np.linalg.norm(self.robot_pos - self.food_pos)
        if d >= self.config.odor_radius:
            return 0.0
        return float(max(0.0, 1.0 - d / self.config.odor_radius))

    def _compute_proximity_readings(self) -> List[float]:
        readings = []
        angle_step = 2.0 * math.pi / self.config.num_sensors

        for i in range(self.config.num_sensors):
            angle = self.robot_theta + i * angle_step
            dist = self._raycast_to_obstacle(angle)
            reading = 1.0 - min(dist, self.config.sensor_range) / self.config.sensor_range
            readings.append(float(np.clip(reading, 0.0, 1.0)))

        return readings

    def _raycast_to_obstacle(self, angle: float) -> float:
        direction = self._heading_vector(angle)
        steps = int(self.config.sensor_range)

        for t in range(1, steps + 1):
            point = self.robot_pos + direction * t
            if self._point_hits_obstacle(point) or not self._point_in_bounds(point):
                return float(t)

        return float(self.config.sensor_range)

    def _is_success(self) -> bool:
        return any(np.linalg.norm(self.robot_pos - f) < self.config.success_threshold
                   for f in self.food_positions)

    def _sample_robot_pose(self) -> Tuple[np.ndarray, float]:
        for _ in range(1000):
            pos = self._sample_free_point()
            theta = self.np_random.uniform(0.0, 2.0 * math.pi)
            return pos, float(theta)
        raise RuntimeError("Failed to sample robot pose.")

    def _sample_food_position(self) -> np.ndarray:
        for _ in range(1000):
            pos = self._sample_free_point()
            if np.linalg.norm(pos - self.robot_pos) >= self.config.min_spawn_distance:
                return pos
        raise RuntimeError("Failed to sample food position.")

    def _sample_free_point(self) -> np.ndarray:
        for _ in range(1000):
            x = self.np_random.uniform(
                self.config.border_margin + self.config.robot_radius,
                self.config.world_width - self.config.border_margin - self.config.robot_radius,
            )
            y = self.np_random.uniform(
                self.config.border_margin + self.config.robot_radius,
                self.config.world_height - self.config.border_margin - self.config.robot_radius,
            )
            p = np.array([x, y], dtype=np.float32)
            if not self._collides(p):
                return p
        raise RuntimeError("Failed to sample free point.")

    def _collides(self, pos: np.ndarray) -> bool:
        if not self._point_in_bounds(pos, radius=self.config.robot_radius):
            return True

        x, y = pos
        for ox, oy, ow, oh in self.obstacles:
            nearest_x = np.clip(x, ox, ox + ow)
            nearest_y = np.clip(y, oy, oy + oh)
            dx = x - nearest_x
            dy = y - nearest_y
            if dx * dx + dy * dy <= self.config.robot_radius ** 2:
                return True
        return False

    def _point_hits_obstacle(self, point: np.ndarray) -> bool:
        x, y = point
        for ox, oy, ow, oh in self.obstacles:
            if ox <= x <= ox + ow and oy <= y <= oy + oh:
                return True
        return False

    def _point_in_bounds(self, point: np.ndarray, radius: float = 0.0) -> bool:
        x, y = point
        return (
            radius <= x <= self.config.world_width - radius
            and radius <= y <= self.config.world_height - radius
        )

    def _build_default_obstacles(self) -> List[Tuple[float, float, float, float]]:
        return [
            (180, 120, 80, 220),
            (340, 60, 60, 160),
        ]

    def _heading_vector(self, angle: float) -> np.ndarray:
        return np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)

    def _wrap_angle(self, angle: float) -> float:
        return float((angle + math.pi) % (2.0 * math.pi) - math.pi)

    def _draw_obstacles(self, canvas):
        for ox, oy, ow, oh in self.obstacles:
            pygame.draw.rect(canvas, self.config.obstacle_color, pygame.Rect(ox, oy, ow, oh))

    def _draw_food(self, canvas):
        for f in self.food_positions:
            pygame.draw.circle(
                canvas,
                self.config.food_color,
                f.astype(int),
                int(self.config.food_radius),
        )
    
    def _draw_robot(self, canvas):
        pygame.draw.circle(
            canvas,
            self.config.robot_color,
            self.robot_pos.astype(int),
            int(self.config.robot_radius),
        )

        # Shorter heading marker for a cleaner look.
        tip = self.robot_pos + self._heading_vector(self.robot_theta) * (self.config.robot_radius + 4)
        pygame.draw.line(canvas, (20, 20, 20), self.robot_pos.astype(int), tip.astype(int), 2)

    def _draw_sensors(self, canvas):
        angle_step = 2.0 * math.pi / self.config.num_sensors
        for i in range(self.config.num_sensors):
            angle = self.robot_theta + i * angle_step
            end = self.robot_pos + self._heading_vector(angle) * self.config.sensor_range
            pygame.draw.line(canvas, self.config.sensor_color, self.robot_pos.astype(int), end.astype(int), 1)

    def _draw_odor(self, canvas):
        surface = pygame.Surface((self.config.world_width, self.config.world_height), pygame.SRCALPHA)
        for f in self.food_positions:
            pygame.draw.circle(
                surface,
                (*self.config.odor_color, 40),
                f.astype(int),
                int(self.config.odor_radius),
            )
        canvas.blit(surface, (0, 0))

