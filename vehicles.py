from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from  environment import Environment
from typing import List, Set, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from coverage_planner import BoustrophedonCoveragePlanner, GridPt


NED = Tuple[float, float, float]
Trajectory = List[NED]

# -----------------------------
# Minimal helpers
# -----------------------------
def dist2d(a: NED, b: NED) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class AUV:
    name: str
    pos: NED
    speed_mps: float = 1.2
    comm_range_m: float = 200.0
    state: str = "surface"
    internal_map: Optional["Environment"] = None  # forward ref
    safety_bubble: Optional[np.ndarray] = field(default=None)
    safety_bubble_radius:float=10.0
    reward: float = 0.0
    cov_quality: float = 0.8 #coverage quality factor for reward calculation
    beta: float = 0.9  # discount factor for reward calculation
    delta: float = 1.5 # punishment factor for overlap
    step: int = 0

    trajectory: Trajectory = field(default_factory=list)
    traj_index: int = 0
    depth_target: float = 0.0

    def set_internal_map(self, env: "Environment") -> None:
        self.internal_map = env

    def dive(self, depth: Optional[float] = None) -> None:
        if depth is not None:
            self.depth_target = float(depth)
        self.state = "diving"
        self.pos = (self.pos[0], self.pos[1], self.depth_target)

    def surface(self) -> None:
        self.state = "surface"
        self.pos = (self.pos[0], self.pos[1], 0.0)

    def set_safety_bubble(self, pos, bubble: int) -> None:
        if self.internal_map is None:
            raise RuntimeError("internal_map is None.")

        H, W = self.internal_map.shape

        # Create empty boolean mask
        bubble_mask = np.zeros((H, W), dtype=bool)

        # Convert world position to grid indices
        ci, cj = self.internal_map.world_to_grid(pos[0],pos[1])

        # Clamp to bounds
        ci = max(0, min(H - 1, int(ci)))
        cj = max(0, min(W - 1, int(cj)))

        # Fill Chebyshev-distance square (excluding center)
        for di in range(-bubble, bubble + 1):
            for dj in range(-bubble, bubble + 1):
                if di == 0 and dj == 0:
                    continue  # exclude center cell

                ni, nj = ci + di, cj + dj
                if 0 <= ni < H and 0 <= nj < W:
                    bubble_mask[ni, nj] = True

        # Store both radius and mask
        self.safety_bubble_radius = bubble
        self.safety_bubble = bubble_mask
        return bubble_mask



    def generate_trajectory_boustrophedon(
        self,
        cover_mask: np.ndarray,
        start_ij: Optional[GridPt] = None,
        sweep_axis: str = "row",
        planner_neighborhood: int = 4,
        depth: Optional[float] = None,
        connect: bool = True,
    ) -> Trajectory:
        """
        Always generate trajectory using boustrophedon on the provided cover_mask.
        cover_mask: bool grid (env.shape) True = free/coverable.
        start_ij: grid start (i,j). If None, uses nearest to current position.
        """
        if self.internal_map is None:
            raise RuntimeError("AUV.internal_map is None. Set it before generating trajectories.")

        env = self.internal_map
        occ_map = cover_mask.astype(bool)

        # Determine start in grid
        if start_ij is None:
            # map current world pos to grid indices (assumes Environment has world_to_grid)
            # fallback: clamp
            si, sj = env.world_to_grid(self.pos[0], self.pos[1])
            si = max(0, min(env.shape[0] - 1, si))
            sj = max(0, min(env.shape[1] - 1, sj))
            start_ij = (si, sj)

        planner = BoustrophedonCoveragePlanner(occ_map=occ_map, neighborhood=planner_neighborhood)
        _, grid_path, _ = planner.plan(start=start_ij, sweep_axis=sweep_axis, connect=connect)

        D = self.pos[2] if depth is None else float(depth)
        traj: Trajectory = []
        for (i, j) in grid_path:
            N, E = env.grid_to_world(i, j)
            traj.append((float(N), float(E), float(D)))

        self.trajectory = traj
        self.traj_index = 0
        return traj

    def follow_trajectory(self, dt: float = 1.0) -> NED:
        """Simple point-to-point follower."""
        if not self.trajectory:
            return self.pos

        tN, tE, tD = self.trajectory[self.traj_index]
        N, E, D = self.pos

        # keep depth equal to target waypoint depth (simple)
        D = tD

        dx = tN - N
        dy = tE - E
        dist = math.hypot(dx, dy)

        if dist < 1e-6:
            self.traj_index = min(self.traj_index + 1, len(self.trajectory) - 1)
            self.pos = (tN, tE, D)
            return self.pos

        step = self.speed_mps * float(dt)
        if step >= dist:
            N, E = tN, tE
            self.traj_index = min(self.traj_index + 1, len(self.trajectory) - 1)
        else:
            N += step * (dx / dist)
            E += step * (dy / dist)

        self.pos = (N, E, D)
        return self.pos

    def telemetry(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pos_ned": self.pos,
            "state": self.state,
            "speed_mps": self.speed_mps,
            "traj_index": self.traj_index,
            "traj_len": len(self.trajectory),
        }

    def calculate_cost(self, cover_mask: np.ndarray) -> float:
        cost= 1
        self.cost= cost

    def calculate_reward(self, cover_mask: np.ndarray) -> float:
        beta = self.beta  # discount factor
        alpha = self.cov_quality  # coverage quality factor
        step=self.step
        delta=self.delta
        t_k=int(np.sum(cover_mask))
        O_k=int(np.sum(self.internal_map.total_Covered_area & cover_mask)) # overlap with already covered
        J = (beta ** step) * (math.log(1.0 + ( t_k-delta *O_k) * alpha) - math.log(self.calculate_cost(cover_mask) ))
        self.reward += J
        self.step+=1
        return self.reward


@dataclass
class ASV:
    name: str
    pos: NED
    speed_mps: float = 2.0
    comm_range_m: float = 150.0
    internal_map: Optional["Environment"] = None
    safety_bubble: Optional[np.ndarray] = field(default=None)
    safety_bubble_radius: float = 1.0
    reward: float = 0.0
    cov_quality: float = 0.8  # coverage quality factor for reward calculation
    beta: float = 0.9  # discount factor for reward calculation
    step: int = 0

    trajectory: Trajectory = field(default_factory=list)
    traj_index: int = 0

    def set_internal_map(self, env: "Environment") -> None:
        self.internal_map = env

    def communicate(self, auv: AUV, message: Any) -> Dict[str, Any]:
        r = dist2d(self.pos, auv.pos)
        link_ok = (r <= self.comm_range_m)
        return {
            "asv": self.name,
            "auv": auv.name,
            "range_m": float(r),
            "link_ok": bool(link_ok),
            "payload": message if link_ok else None,
        }

    def set_safety_bubble(self, pos, bubble: int) -> None:
        if self.internal_map is None:
            raise RuntimeError("internal_map is None.")

        H, W = self.internal_map.shape

        # Create empty boolean mask
        bubble_mask = np.zeros((H, W), dtype=bool)

        # Convert world position to grid indices
        ci, cj = self.internal_map.world_to_grid(pos[0], pos[1])

        # Clamp to bounds
        ci = max(0, min(H - 1, int(ci)))
        cj = max(0, min(W - 1, int(cj)))

        # Fill Chebyshev-distance square (excluding center)
        for di in range(-bubble, bubble + 1):
            for dj in range(-bubble, bubble + 1):
                if di == 0 and dj == 0:
                    continue  # exclude center cell

                ni, nj = ci + di, cj + dj
                if 0 <= ni < H and 0 <= nj < W:
                    bubble_mask[ni, nj] = True

        # Store both radius and mask
        self.safety_bubble_radius = bubble
        self.safety_bubble = bubble_mask
        return bubble_mask


    def generate_trajectory_boustrophedon(
        self,
        cover_mask: np.ndarray,
        start_ij: Optional[GridPt] = None,
        sweep_axis: str = "row",
        planner_neighborhood: int = 4,
        connect: bool = True,

    ) -> Trajectory:
        """
        Always generate trajectory using boustrophedon on provided cover_mask.
        ASV stays at D=0.
        """
        if self.internal_map is None:
            raise RuntimeError("ASV.internal_map is None. Set it before generating trajectories.")

        env = self.internal_map
        occ_map = cover_mask.astype(bool)

        if start_ij is None:
            si, sj = env.world_to_grid(self.pos[0], self.pos[1])
            si = max(0, min(env.shape[0] - 1, si))
            sj = max(0, min(env.shape[1] - 1, sj))
            start_ij = (si, sj)

        planner = BoustrophedonCoveragePlanner(occ_map=occ_map, neighborhood=planner_neighborhood)
        _, grid_path, _ = planner.plan(start=start_ij, sweep_axis=sweep_axis)

        traj: Trajectory = []
        for (i, j) in grid_path:
            N, E = env.grid_to_world(i, j)
            traj.append((float(N), float(E), 0.0))

        self.trajectory = traj
        self.traj_index = 0
        return traj,

    def follow_trajectory(self, dt: float = 1.0) -> NED:
        if not self.trajectory:
            return self.pos

        tN, tE, _ = self.trajectory[self.traj_index]
        N, E, _ = self.pos

        dx = tN - N
        dy = tE - E
        dist = math.hypot(dx, dy)

        step = self.speed_mps * float(dt)
        if dist < 1e-6 or step >= dist:
            N, E = tN, tE
            self.traj_index = min(self.traj_index + 1, len(self.trajectory) - 1)
        else:
            N += step * (dx / dist)
            E += step * (dy / dist)

        self.pos = (N, E, 0.0)
        return self.pos


