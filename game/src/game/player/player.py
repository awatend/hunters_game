from __future__ import annotations

import configparser

from dataclasses import dataclass, field
from typing import List

try:
    import redis
except ImportError as error:
    redis = None
    _REDIS_IMPORT_ERROR = error
from game.internal_map.environment import Environment

def _csv(value: str, cast=float) -> List:
    return [cast(item.strip()) for item in value.split(",")]


@dataclass
class Player:
    """Runtime state and motion parameters for one player."""

    id: int
    name: str
    type: str
    width: float
    sensor_swath: float
    cruise_speed: float
    turning_speed: float
    max_curvature: float
    initial_pose: List[float]
    surface: int
    neighborhood: int
    w_safe: float
    w_explore: float
    w_exploit: float
    min_safe_dist: float
    current_pose: List[float] | None = None
    environment: Environment | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if len(self.initial_pose) != 3:
            raise ValueError("initial_pose must contain N, E, and depth")
        self.current_pose = list(self.initial_pose)

    @classmethod
    def from_section(
        cls,
        section: configparser.SectionProxy,
        fallback_name: str,
    ) -> "Player":
        return cls(
            id=section.getint("id"),
            name=section.get("name", fallback_name),
            type=section.get("type", "auv").lower(),
            width=section.getfloat("width"),
            sensor_swath=section.getfloat("sensor_swath"),
            cruise_speed=section.getfloat("cruise_speed"),
            turning_speed=section.getfloat("turning_speed"),
            max_curvature=section.getfloat("max_curvature"),
            initial_pose=_csv(section["initial_pose"]),
            surface=section.getint("surface", fallback=50),
            neighborhood=section.getint("neighborhood", fallback=8),
            w_safe=section.getfloat("w_safe", fallback=0.5),
            w_explore=section.getfloat("w_explore", fallback=1.0),
            w_exploit=section.getfloat("w_exploit", fallback=0.5),
            min_safe_dist=section.getfloat("min_safe_dist", fallback=6.0),
        )
