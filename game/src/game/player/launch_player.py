"""Load and simulate players described by an INI file.

Use ``[player]`` for one player or ``[player.<name>]`` for multiple players.
For example::

    python -m game.player.launch_player players.ini

    python launcher.py --steps 1000 config.ini
"""
from __future__ import annotations

import argparse
import configparser
import math
import os
from pathlib import Path
from typing import Dict, List
import time
import numpy as np
import shapely.geometry as sg
from shapely import wkt
try:
    import redis
except ImportError as error:
    redis = None
    _REDIS_IMPORT_ERROR = error
from game.internal_map.cell_geometry import RectangularGeometry
from game.internal_map.environment import Environment
from game.player.player import Player

def _csv(value: str, cast=float) -> List:
    return [cast(item.strip()) for item in value.split(",")]



class LaunchPlayer:
    """Load an arbitrary number of players from one INI file."""

    def __init__(self, config_file: str | Path, redis_client=None):
        self.config_file = Path(config_file).resolve()
        self.redis_client = redis_client
        parser = configparser.ConfigParser()
        if not parser.read(self.config_file):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        self.config = parser
        geometry = parser["geometry"]
        rows, cols = _csv(geometry["shape"], int)
        self.geometry = RectangularGeometry(
            shape=(rows, cols),
            resolution=geometry.getfloat("resolution"),
            origin=tuple(_csv(geometry.get("origin", "0, 0"))),
            connectivity=geometry.getint("connectivity", fallback=8),
        )

        sections = [
            (name, parser[name])
            for name in parser.sections()
            if name.lower().startswith("player.")
        ]
        if not sections and parser.has_section("player"):
            sections = [("player", parser["player"])]
        if not sections:
            raise ValueError("INI file must contain [player] or [player.<name>] sections")

        self.players: Dict[str, Player] = {}
        for section_name, section in sections:
            fallback_name = section_name.split(".", 1)[-1]
            current_player = Player.from_section(section, fallback_name)
            if current_player.name in self.players:
                raise ValueError(f"Duplicate player name: {current_player.name}")
            current_player.environment = self.setup_environment()
            self.players[current_player.name] = current_player

        simulation = parser["simulation"] if parser.has_section("simulation") else {}
        self.steps = int(simulation.get("steps", simulation.get("iterations", 0)))
        self.treasure_location = self.set_treasure_location()

    def setup_environment(self) -> Environment:
        """Create and initialize one player's environment."""
        environment = Environment(self.geometry)
        environment.Initial_area[:] = True
        environment.update(np.full(self.geometry.shape, 0.01, dtype=np.float32))
        return environment

    def set_treasure_location(self):
        """Select a random feasible grid cell for the treasure."""
        environment = next(iter(self.players.values())).environment
        if environment is None:
            raise RuntimeError("Player environment has not been initialized")
        environment.set_collision_area()
        coords = np.argwhere(environment.Initial_area & ~environment.Collision_area)
        if len(coords) == 0:
            raise ValueError("Cannot place treasure: no feasible environment cells")
        return tuple(coords[np.random.randint(len(coords))])

    @staticmethod
    def is_treasure_found(mask, treasure_location) -> bool:
        """Return whether a coverage mask contains the treasure."""
        if treasure_location is None:
            return False
        i, j = treasure_location
        return bool(mask[i, j])

    def request_allocation(self, player: Player, polygon):
        """Request an allocation for a player."""
        if self.redis_client is None:
            raise RuntimeError("Redis client is not configured")
        current_time = int (time.time())
        status= self.redis_client.xadd(
            "allocation_requests",
            {
                "id": str(player.id),
                "time": str(current_time),
                "roi": polygon.wkt,
            },
        )
        return current_time

    def get_allocation_result(self, player: Player, request_time):
        """Wait for an allocation result for up to five minutes."""
        if self.redis_client is None:
            raise RuntimeError("Redis client is not configured")

        deadline = time.monotonic() + 5 * 60
        while True:
            messages = self.redis_client.xrevrange("allocation_results", count=100)
            #print(f"messages:{messages}")

            latest = None
            for  id, data in messages:
                if float(data["time"]) >= float(request_time):
                    latest = data
                    break

            if latest is not None:
                status = int(latest["status"])
                if latest["roi"]:
                    area = wkt.loads(latest["roi"])
                else:
                    area = None
                print(f"status:{status},area:{area}")
                return status, area

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"No allocation result received for player {player.id} "
                    "within 5 minutes"
                )
            time.sleep(min(10, remaining))

    def take_action(self, player: Player):
        """Generate a coverage action for one player."""
        if player.environment is None:
            raise RuntimeError(f"Player {player.name} has no environment")
        environment = player.environment
        environment.set_collision_area()
        start = environment.world_to_grid(*player.current_pose[:2])
        seed = environment.select_seed_ij(start_location=start)
        if seed is None:
            return None
        return environment.grow_surface_from_seed(
            seed_ij=seed,
            S=player.surface,
            neighborhood=player.neighborhood,
            w_safe=player.w_safe,
            w_explore=player.w_explore,
            w_exploit=player.w_exploit,
            min_safe_dist_m=player.min_safe_dist,
        )[0]

    def route_planner(self, polygon, player: Player, output_dir=None):
        """Plan coverage of a polygon using the player's motion parameters."""
        if polygon is None or polygon.is_empty:
            return None
        try:
            from marcov import (
                OptimizeSwathLengths, PathPlanner, RouteOrder, RoutePlanner,
                SwathGenerator, TurnType, Vehicle,
            )
        except ImportError as error:
            raise ImportError("route_planner requires the marcov package") from error
        if isinstance(polygon, sg.MultiPolygon):
            polygon = max(polygon.geoms, key=lambda item: item.area)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        vehicle = Vehicle(
            width=player.width,
            cov_width=player.sensor_swath,
            cruise_speed=player.cruise_speed,
            turn_speed=player.turning_speed,
            max_curvature=player.max_curvature,
            name=player.name,
        )
        swaths = SwathGenerator(vehicle).generate(
            polygon, angle=OptimizeSwathLengths(step=math.pi / 36), headland_distance=7.0
        )
        route = RoutePlanner().plan(swaths, order=RouteOrder.SPIRAL)
        path_planner = PathPlanner(vehicle, turn_type=TurnType.DUBINS, step=1.0)
        path=path_planner.plan(
            route, headland=polygon
        )
        if not path.states:
            return None
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        waypoints=path_planner.simplify_path(path,  tolerance=1.0)
        return waypoints

    def run(self, steps: int | None = None):
        """Return player telemetry for each requested simulation step."""
        count = self.steps if steps is None else steps
        if count < 0:
            raise ValueError("steps must be non-negative")

        for step in range(count):
            for player in self.players.values():
                action = self.take_action(player)
                if action is not None:
                    found = self.is_treasure_found(action, self.treasure_location)
                    if found:
                        print(f"Step {step}: Player {player.name} found the treasure at {self.treasure_location}")
                        return
                    polygon = player.environment.geom.mask_to_polygon(action)
                    print (f'polygon: {polygon}')
                    waypoints = self.route_planner(polygon, player)
                    request_time = self.request_allocation(player, polygon)
                    status, area = self.get_allocation_result(player, request_time)
                    if status == 0:
                        player.environment.mark_covered(action)
                        player.current_pose = list(waypoints[-1]) if waypoints else player.current_pose
                    elif status== 1:
                        print(f"Step {step}: Player {player.name} area already covered")
                        player.environment.mark_covered(
                            player.environment.geom.polygon_to_mask(area)
                        )
                    elif status == 2:
                        print(f"Step {step}: Player {player.name} area unsafe")
                        player.environment.add_collision_area(
                            player.environment.geom.polygon_to_mask(area)
                        )









def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="INI file containing one or more players")
    parser.add_argument("--steps", type=int, help="override the configured simulation step count")
    args = parser.parse_args()
    if args.steps is None:
        args.steps = 100  # Default step count
    if redis is None:
        raise RuntimeError(
            "The redis package is required. Install it with: pip install redis"
        ) from _REDIS_IMPORT_ERROR
    client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    client.ping()
    launcher = LaunchPlayer(args.config, redis_client=client)
    launcher.run( steps=args.steps)
    print(f"Loaded {len(launcher.players)} player(s): {', '.join(launcher.players)}")


if __name__ == "__main__":
    main()
