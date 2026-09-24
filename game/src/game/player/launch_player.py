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
from typing import Dict
import time
import json
import numpy as np
import shapely.geometry as sg
from shapely import wkt
try:
    import redis
except ImportError as error:
    redis = None
    _REDIS_IMPORT_ERROR = error
from game.internal_map.environment import Environment
from game.player.player import Player
from game.prior.load_prior import LoadPrior



class LaunchPlayer:
    """Load an arbitrary number of players from one INI file."""

    def __init__(self, config_file: str | Path, redis_client=None):
        self.config_file = Path(config_file).resolve()
        self.redis_client = redis_client
        parser = configparser.ConfigParser()
        if not parser.read(self.config_file):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        self.config = parser
        redis_config = parser["redis"] if parser.has_section("redis") else None
        self.request_stream = (
            redis_config.get("request_stream", "allocation_requests")
            if redis_config else "allocation_requests"
        )
        self.result_hash = (
            redis_config.get("result_hash", "allocation_results")
            if redis_config else "allocation_results"
        )
        self.coverage_stream = (
            redis_config.get("coverage_stream", "coverage_progress")
            if redis_config else "coverage_progress"
        )
        self.obstacle_stream = (
            redis_config.get("obstacle_stream", "obstacles")
            if redis_config else "obstacles"
        )
        if not parser.has_section("prior"):
            raise ValueError("INI file must contain a [prior] section")

        prior_config = dict(parser["prior"])
        prior_path = prior_config.get("path")
        if prior_path is not None:
            configured_path = Path(prior_path.strip().strip("\"'"))
            if not configured_path.is_absolute():
                from_config = self.config_file.parent / configured_path
                prior_config["path"] = str(
                    from_config if from_config.exists() else configured_path
                )

        self.prior = LoadPrior(**prior_config)
        self.geometry = self.prior.geometry

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
        environment.update(self.prior.field)
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

    def plot_areas(self, output: str | Path | None = None, show: bool = True):
        """Plot initial and covered areas using the active hexagonal grid.

        Covered cells are combined across all loaded players.  The initial
        area is shown in blue and covered cells are shown in orange.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MatplotlibPolygon

        environments = [
            player.environment
            for player in self.players.values()
            if player.environment is not None
        ]
        if not environments:
            raise RuntimeError("No player environment has been initialized")

        initial = np.logical_or.reduce(
            [environment.Initial_area for environment in environments]
        )
        covered = np.logical_or.reduce(
            [environment.total_Covered_area for environment in environments]
        )
        if initial.shape != self.geometry.shape or covered.shape != self.geometry.shape:
            raise ValueError("Environment masks do not match the active geometry")

        figure, axis = plt.subplots(figsize=(10, 8))
        for row in range(self.geometry.shape[0]):
            for col in range(self.geometry.shape[1]):
                is_initial = bool(initial[row, col])
                is_covered = bool(covered[row, col])
                if not is_initial and not is_covered:
                    continue
                vertices = [
                    (east, north)
                    for north, east in self.geometry.cell_vertices(row, col)
                ]
                facecolor = "tab:orange" if is_covered else "tab:blue"
                axis.add_patch(MatplotlibPolygon(
                    vertices,
                    closed=True,
                    facecolor=facecolor,
                    edgecolor="white",
                    linewidth=0.2,
                    alpha=0.8,
                ))

        axis.set_aspect("equal")
        extent = self.geometry.world_extent()
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_xlabel("easting (m)")
        axis.set_ylabel("northing (m)")
        axis.set_title("Initial and covered hexagonal areas")
        from matplotlib.patches import Patch
        axis.legend(handles=[
            Patch(facecolor="tab:blue", label="Initial area"),
            Patch(facecolor="tab:orange", label="Covered area"),
        ])
        figure.tight_layout()
        if output is not None:
            figure.savefig(output, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        return figure, axis

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
        message_id = self.redis_client.xadd(
            self.request_stream,
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
            raw_result = self.redis_client.hget(self.result_hash, str(player.id))
            latest = json.loads(raw_result) if raw_result is not None else None
            if latest is not None and float(latest["time"]) < float(request_time):
                latest = None

            if latest is not None:
                status = int(latest["status"])
                if latest["roi"]:
                    area = wkt.loads(latest["roi"])
                else:
                    area = None
                self.redis_client.hdel(self.result_hash, str(player.id))
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
        start = environment.world_to_grid(player.current_pose[1], player.current_pose[0])
        print(f"Start point for action {start}")
        #seed = environment.select_seed_ij(start_location=start)
        #if seed is None:
            #return None
        #surface grow from seed=start
        return environment.grow_surface_from_seed(
            seed_ij=start,
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
                SwathGenerator, TurnType, Vehicle,plot_path,plot_route,save_figure,
                plot_area,plot_headland, plot_coverage,save_figure

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
            polygon, angle=OptimizeSwathLengths(step=math.pi / 50), headland_distance=player.sensor_swath/2
        )
        original_swaths = swaths
        print(f"Swaths generated: {len(swaths.swaths)}")
        route = RoutePlanner().plan(swaths, order=RouteOrder.BOUSTROPHEDON)
        print(f"Route planned: {len(route.swaths)} swaths, {len(route.connections)} connections")

        path_planner = PathPlanner(vehicle, turn_type=TurnType.DUBINS, step=1.0)
        path=path_planner.plan(route, headland=polygon)
        if not path.states:
            return None
        if output_dir is not None:
            figure = plot_coverage(polygon, swath_result=original_swaths, route_result=route, path=path, show_headland=True)
            current_time = int(time.time())
            save_figure(figure, os.path.join(output_dir, f"thor_{current_time}.png"))


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
                    #print (f'polygon: {polygon}')
                    #print(f"boundaries{player.environment.geom.mask_to_boundary_polygon(action)}")
                    waypoints = self.route_planner(polygon, player, output_dir=f"output/{player.name}")
                    print (f'sending an allocation request')
                    request_time = self.request_allocation(player, polygon)
                    print (f'waiting for allocation')
                    status, area = self.get_allocation_result(player, request_time)
                    if status == 0:
                        player.environment.mark_covered(action)
                        player.current_pose = list(waypoints[-1]) if waypoints else player.current_pose
                        print(f"current_pose: {player.current_pose}")
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
    config = configparser.ConfigParser()
    config.read(args.config)
    redis_config = config["redis"] if config.has_section("redis") else None
    client = redis.Redis(
        host=redis_config.get("host", "localhost") if redis_config else "localhost",
        port=redis_config.getint("port", fallback=6379) if redis_config else 6379,
        db=redis_config.getint("db", fallback=0) if redis_config else 0,
        decode_responses=True,
    )
    client.ping()
    launcher = LaunchPlayer(args.config, redis_client=client)
    launcher.run( steps=args.steps)
    print(f"Loaded {len(launcher.players)} player(s): {', '.join(launcher.players)}")


if __name__ == "__main__":
    main()
