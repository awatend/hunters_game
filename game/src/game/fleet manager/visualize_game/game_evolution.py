"""Animate cumulative player coverage from the Redis coverage stream."""
from __future__ import annotations

import argparse
import configparser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon as MatplotlibPolygon
from matplotlib.patches import RegularPolygon
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon
import redis

from game.internal_map.environment import Environment
from game.prior.load_prior import LoadPrior

COLORS = ("tab:blue", "tab:orange", "tab:pink", "tab:green", "tab:purple", "tab:brown", "tab:red", "tab:pink", "tab:gray", "tab:olive", "tab:cyan")


class GameEvolution:
    """Read coverage events and animate each agent in a separate color."""

    def __init__(self, config_file: str | Path):
        parser = configparser.ConfigParser()
        self.config_file = Path(config_file).resolve()
        if not parser.read(self.config_file):
            raise FileNotFoundError(self.config_file)
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
        self.environment = Environment(self.geometry)
        self.environment.Initial_area[:] = True
        self.environment.update(self.prior.field)

        redis_config = parser["redis"]
        self.stream = redis_config.get("stream", "coverage_progress")
        self.redis_client = redis.Redis(
            host=redis_config.get("host", "localhost"),
            port=redis_config.getint("port", fallback=6379),
            db=redis_config.getint("db", fallback=0),
            decode_responses=True,
        )
        self.redis_client.ping()
        self.last_id = redis_config.get("start_id", "0-0")
        self.patches = {}
        self.latest_boundaries = {}
        self.agent_colors = {}
        self.legend_handles = {}

    def _add_polygon(self, agent_id: str, polygon_text: str) -> None:
        if not polygon_text:
            return
        geometry = wkt.loads(polygon_text)
        if geometry.is_empty:
            return
        is_new_agent = agent_id not in self.agent_colors
        color = self.agent_colors.setdefault(agent_id, COLORS[len(self.agent_colors) % len(COLORS)])
        polygons = geometry.geoms if isinstance(geometry, MultiPolygon) else (geometry,)
        for boundary in self.latest_boundaries.pop(agent_id, []):
            boundary.remove()
        latest_boundaries = []
        for polygon in polygons:
            if not isinstance(polygon, Polygon):
                continue
            patch = MatplotlibPolygon(
                list(polygon.exterior.coords),
                closed=True,
                facecolor=color,
                edgecolor=color,
                alpha=0.35,
                label=f"Agent {agent_id}",
            )
            self.ax.add_patch(patch)
            self.patches.setdefault(agent_id, []).append(patch)
            boundary = MatplotlibPolygon(
                list(polygon.exterior.coords),
                closed=True,
                fill=False,
                edgecolor="red",
                linewidth=2.0,
                zorder=3,
            )
            self.ax.add_patch(boundary)
            latest_boundaries.append(boundary)
            if is_new_agent:
                self.legend_handles[agent_id] = patch
                self.ax.legend(
                    self.legend_handles.values(),
                    [f"Agent {name}" for name in self.legend_handles],
                    loc="upper right",
                    title="Players",
                )
                is_new_agent = False
        self.latest_boundaries[agent_id] = latest_boundaries

    def update(self, _frame):
        messages = self.redis_client.xread({self.stream: self.last_id}, count=100, block=0)
        for _, entries in messages:
            for message_id, data in entries:
                self.last_id = message_id
                self._add_polygon(str(data["agent_id"]), data.get("roi", ""))
        self.ax.set_title(f"Coverage evolution | stream: {self.stream}")
        return (
            [patch for patches in self.patches.values() for patch in patches]
            + [boundary for boundaries in self.latest_boundaries.values() for boundary in boundaries]
        )

    def show(self, interval_ms: int = 1000) -> None:
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self._plot_prior()
        xmin, xmax, ymin, ymax = self.geometry.world_extent()
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("E [m]")
        self.ax.set_ylabel("N [m]")
        self.ax.grid(True, alpha=0.25)
        self.ax.legend([], [], loc="upper right", title="Players")
        self.animation = FuncAnimation(
            self.fig, self.update, interval=interval_ms, blit=False, cache_frame_data=False
        )
        plt.show()

    def _plot_prior(self) -> None:
        """Draw the loaded prior beneath the animated coverage polygons."""
        field = self.prior.field
        finite = field[np.isfinite(field)]
        if finite.size == 0:
            raise ValueError("The configured prior contains no finite cells")

        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(float(finite.min()), float(finite.max()))
        for row, col in zip(*np.nonzero(np.isfinite(field))):
            north, east = self.geometry.cell_to_world(row, col)
            self.ax.add_patch(RegularPolygon(
                (east, north),
                numVertices=6,
                radius=self.geometry.cell_radius * 1.001,
                orientation=np.pi / 6,
                facecolor=cmap(norm(field[row, col])),
                edgecolor="none",
                alpha=0.6,
                zorder=0,
            ))
        self.fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=self.ax,
            label=self.prior._data_name,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_name("visualization.ini"))
    parser.add_argument("--interval-ms", type=int, default=2000)
    args = parser.parse_args()
    GameEvolution(args.config).show(args.interval_ms)


if __name__ == "__main__":
    main()