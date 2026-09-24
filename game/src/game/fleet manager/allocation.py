"""Redis-backed fleet allocation service."""
from __future__ import annotations

import time
import json
import argparse
import configparser
from pathlib import Path
from typing import Any

import redis
from shapely import wkt
from shapely.geometry.base import BaseGeometry

from game.internal_map.environment import Environment
from game.prior.load_prior import LoadPrior


class Allocator:
    """Process obstacle and allocation streams.

    Parameters
    ----------
    environment:
        Environment owned by the allocator. Obstacles received on the
        ``obstacles`` stream are rasterized onto this environment.
    redis_client:
        Configured ``redis.Redis`` client. Responses are expected to use
        ``decode_responses=True``.
    """

    def __init__(
        self,
        environment: Environment,
        redis_client: Any,
        *,
        request_stream: str = "allocation_requests",
        result_hash: str = "allocation_results",
        progress_stream: str = "coverage_progress",
        obstacle_stream: str = "obstacles",
        coverage_stream: str = "coverage_progress",
    ) -> None:
        self.environment = environment
        self.redis_client = redis_client
        self.request_stream = request_stream
        self.result_hash = result_hash
        self.progress_stream = progress_stream
        self.obstacle_stream = obstacle_stream
        self.coverage_stream = coverage_stream
        self._last_request_id = "0-0"
        self._last_obstacle_id = "0-0"
        self._last_coverage_id = "0-0"
        self._load_existing_state()

    def add_static_obstacles(self) -> None:
        """Apply new obstacle records from the obstacle stream.

        Each stream record may contain
        ``roi``, ``polygon``, or ``obstacle`` with a WKT value.
        """
        while True:
            messages = self.redis_client.xread(
                {self.obstacle_stream: self._last_obstacle_id},
                count=100,
            )
            if not messages:
                break
            for _, entries in messages:
                for message_id, data in entries:
                    self._last_obstacle_id = message_id
                    self._apply_obstacle_record(data)

    def _load_existing_state(self) -> None:
        """Replay coverage and obstacles already present in Redis."""
        self._load_existing_coverage()
        self.add_static_obstacles()

    def _load_existing_coverage(self) -> None:
        """Initialize coverage masks from the complete coverage stream."""
        while True:
            messages = self.redis_client.xread(
                {self.coverage_stream: self._last_coverage_id},
                count=100,
            )
            if not messages:
                break
            for _, entries in messages:
                for message_id, data in entries:
                    self._last_coverage_id = message_id
                    self._apply_coverage_record(data)

    def _apply_coverage_record(self, data: dict[str, Any]) -> None:
        polygon_text = self._extract_polygon_text(data)
        if polygon_text:
            mask = self.environment.geom.polygon_to_mask(wkt.loads(polygon_text))
            self.environment.Covered_area |= mask
            self.environment.total_Covered_area |= mask

    def _apply_obstacle_record(self, data: dict[str, Any]) -> None:
        polygon_text = self._extract_polygon_text(data)
        if polygon_text:
            self._add_obstacle(wkt.loads(polygon_text))

    def _add_obstacle(self, polygon: BaseGeometry) -> None:
        if polygon.is_empty:
            return
        obstacle_mask = self.environment.geom.polygon_to_mask(polygon)
        self.environment.static_Collision_area |= obstacle_mask
        self.environment.set_collision_area()

    @staticmethod
    def _extract_polygon_text(payload: Any) -> str | None:
        if isinstance(payload, bytes):
            payload = payload.decode()
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            for key in ("roi", "polygon", "obstacle"):
                value = payload.get(key)
                if value:
                    return value.decode() if isinstance(value, bytes) else str(value)
        return None

    def process_allocation_request(self, message_id: str, data: dict[str, Any]) -> str:
        """Process one request and publish its result.

        Agents 1 and 2 are acknowledged with an empty ROI and their
        requested polygon is published to ``coverage_progress``. Agent 3 is
        acknowledged with the requested polygon as its allocated ROI.
        """

        agent_id = str(data.get("id", data.get("agent_id", "")))
        current_time = str(data.get("time", int(time.time())))
        polygon_text = str(data.get("roi", ""))
        if not agent_id or not polygon_text:
            raise ValueError("Allocation request requires id and roi")

        polygon = wkt.loads(polygon_text)
        mask=self.environment.geom.polygon_to_mask(polygon)
        unsafe_region = self.environment.static_Collision_area & mask
        overlapped_area = self.environment.Covered_area & mask
        if unsafe_region.any():
            unsafe_polygon=self.environment.geom.mask_to_polygon(unsafe_region)
            result = {
                "time": current_time,
                "status": "2",
                "roi": unsafe_polygon.wkt,
            }
            self.redis_client.hset(
                self.result_hash,
                agent_id,
                json.dumps(result),
            )
            print("Allocation denied:unsafe area", self._last_request_id)
        if overlapped_area.any() :
            overlapped_polygon=self.environment.geom.mask_to_polygon(overlapped_area)
            result = {
                "time": current_time,
                "status": "1",
                "roi": overlapped_polygon.wkt,
            }
            self.redis_client.hset(
                self.result_hash,
                agent_id,
                json.dumps(result),
            )
            print("Allocation denied:overalpped area", self._last_request_id)

        if not unsafe_region.any() and not overlapped_area.any():
            result = {
                "id": agent_id,
                "time": current_time,
                "status": "0",
                "roi": polygon.wkt,
            }
            print("Allocation granted:", self._last_request_id)

            self.redis_client.hset(
                self.result_hash,
                agent_id,
                json.dumps(result),
            )
            self.environment.Covered_area |= mask
            #Add the allocated area in a redis stream for coverage progress
            # stream should contain the agent_id, time and the allocated roi
            self.redis_client.xadd(
                self.coverage_stream,
                {
                    "agent_id": agent_id,
                    "time": current_time,
                    "roi": polygon.wkt,
                },
            )

        return message_id

    def process_once(self, block_ms: int = 1000) -> bool:
        """Process pending state and allocation messages once."""
        self.add_static_obstacles()
        self._load_existing_coverage()
        messages = self.redis_client.xread(
            {self.request_stream: self._last_request_id},
            count=1,
            block=block_ms,
        )
        if not messages:
            print("no new allocation request")
            return False
        for _, entries in messages:
            for message_id, data in entries:
                self.process_allocation_request(message_id, data)
                self._last_request_id = message_id
        return True

    def run_forever(self, block_ms: int = 1000) -> None:
        """Continuously process obstacles and allocation requests."""
        while True:
            self.process_once(block_ms=block_ms)


def main() -> None:
    """Start the Redis allocation service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("allocation.ini"),
        help="INI file containing the prior and Redis settings",
    )
    parser.add_argument("--block-ms", type=int, default=1000)
    args = parser.parse_args()

    config_file = args.config.resolve()
    config = configparser.ConfigParser()
    if not config.read(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    if not config.has_section("prior"):
        raise ValueError("INI file must contain a [prior] section")

    prior_config = dict(config["prior"])
    prior_path = prior_config.get("path")
    if prior_path is not None:
        configured_path = Path(prior_path.strip().strip("\"'"))
        if not configured_path.is_absolute():
            from_config = config_file.parent / configured_path
            prior_config["path"] = str(
                from_config if from_config.exists() else configured_path
            )

    prior = LoadPrior(**prior_config)
    environment = Environment(prior.geometry)
    environment.Initial_area[:] = True
    environment.update(prior.field)

    redis_config = config["redis"] if config.has_section("redis") else None
    redis_host = redis_config.get("host", "localhost") if redis_config else "localhost"
    redis_port = redis_config.getint("port", fallback=6379) if redis_config else 6379
    redis_db = redis_config.getint("db", fallback=0) if redis_config else 0

    redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        decode_responses=True,
    )
    redis_client.ping()

    redis_config = config["redis"] if config.has_section("redis") else None
    request_stream = redis_config.get("request_stream", "allocation_requests") if redis_config else "allocation_requests"
    result_hash = redis_config.get("result_hash", "allocation_results") if redis_config else "allocation_results"
    coverage_stream = redis_config.get("coverage_stream", "coverage_progress") if redis_config else "coverage_progress"
    obstacle_stream = redis_config.get("obstacle_stream", "obstacles") if redis_config else "obstacles"

    service = Allocator(
        environment,
        redis_client,
        request_stream=request_stream,
        result_hash=result_hash,
        progress_stream=coverage_stream,
        coverage_stream=coverage_stream,
        obstacle_stream=obstacle_stream,
    )
    print(
        f"Allocator listening on {redis_host}:{redis_port} "
        f"(requests={service.request_stream}, results={service.result_hash})"
    )
    try:
        service.run_forever(block_ms=args.block_ms)
    except KeyboardInterrupt:
        print("Allocator stopped")


if __name__ == "__main__":
    main()
