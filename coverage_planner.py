# ----------------------------
# Spiral-STC planner inspired by Gabriely et.al. (2002) -
#
# ----------------------------
#from typing import Any
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
#from environment import Environment
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math

class SpiralSpanningTreeCoveragePlanner:
    def __init__(self, occ_map: np.ndarray):
        self.origin_map_height = occ_map.shape[0]
        self.origin_map_width = occ_map.shape[1]

        if self.origin_map_height % 2 == 1 or self.origin_map_width % 2 == 1:
            raise ValueError("occ_map width/height must be even")

        self.occ_map = occ_map.astype(bool)
        self.merged_map_height = self.origin_map_height // 2
        self.merged_map_width = self.origin_map_width // 2
        self.edge = []

    def plan(self, start):
        visit_times = np.zeros((self.merged_map_height, self.merged_map_width), dtype=int)
        visit_times[start[0], start[1]] = 1

        route = []
        self.edge = []
        self._perform_spanning_tree_coverage(start, visit_times, route)

        # Convert route to fine-grid path (list of [p,q] steps in original occ_map coordinates)
        path = []
        for idx in range(len(route) - 1):
            p = route[idx]
            q = route[idx + 1]
            dp = abs(p[0] - q[0]) + abs(p[1] - q[1])
            if dp == 0:
                path.append(self._round_trip_path(route[idx - 1], p))
            elif dp == 1:
                path.append(self._move(p, q))
            elif dp == 2:
                mid = self._intermediate_node(p, q)
                path.append(self._move(p, mid))
                path.append(self._move(mid, q))
            else:
                raise RuntimeError("adjacent route nodes distance > 2 (unexpected)")

        return self.edge, route, path

    def _valid_merged_node(self, i, j) -> bool:
        if not (0 <= i < self.merged_map_height and 0 <= j < self.merged_map_width):
            return False
        # merged cell is free only if its 2x2 block is all free
        return bool(
            self.occ_map[2 * i, 2 * j]
            and self.occ_map[2 * i + 1, 2 * j]
            and self.occ_map[2 * i, 2 * j + 1]
            and self.occ_map[2 * i + 1, 2 * j + 1]
        )

    def _perform_spanning_tree_coverage(self, current_node, visit_times, route):
        order = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # CCW

        found = False
        route.append(current_node)

        for di, dj in order:
            ni, nj = current_node[0] + di, current_node[1] + dj
            if self._valid_merged_node(ni, nj) and visit_times[ni, nj] == 0:
                neighbor = (ni, nj)
                self.edge.append((current_node, neighbor))
                found = True
                visit_times[ni, nj] = 1
                self._perform_spanning_tree_coverage(neighbor, visit_times, route)

        if not found:
            # backtrace
            for node in reversed(route):
                if visit_times[node[0], node[1]] == 2:
                    continue
                visit_times[node[0], node[1]] += 1
                route.append(node)

                # stop when we find a node with an unvisited valid neighbor
                for di, dj in order:
                    ni, nj = node[0] + di, node[1] + dj
                    if self._valid_merged_node(ni, nj) and visit_times[ni, nj] == 0:
                        return

    def _direction(self, p, q):
        if p[0] == q[0] and p[1] < q[1]:
            return "E"
        if p[0] == q[0] and p[1] > q[1]:
            return "W"
        if p[0] < q[0] and p[1] == q[1]:
            return "S"
        if p[0] > q[0] and p[1] == q[1]:
            return "N"
        raise RuntimeError("Only E/W/S/N supported")

    def _sub_node(self, node, direction):
        i, j = node
        if direction == "SE":
            return [2 * i + 1, 2 * j + 1]
        if direction == "SW":
            return [2 * i + 1, 2 * j]
        if direction == "NE":
            return [2 * i, 2 * j + 1]
        if direction == "NW":
            return [2 * i, 2 * j]
        raise RuntimeError("Bad sub-node direction")

    def _move(self, p, q):
        d = self._direction(p, q)
        if d == "E":
            return [self._sub_node(p, "SE"), self._sub_node(q, "SW")]
        if d == "W":
            return [self._sub_node(p, "NW"), self._sub_node(q, "NE")]
        if d == "S":
            return [self._sub_node(p, "SW"), self._sub_node(q, "NW")]
        if d == "N":
            return [self._sub_node(p, "NE"), self._sub_node(q, "SE")]
        raise RuntimeError("move direction error")

    def _round_trip_path(self, last, pivot):
        d = self._direction(last, pivot)
        if d == "E":
            return [self._sub_node(pivot, "SE"), self._sub_node(pivot, "NE")]
        if d == "S":
            return [self._sub_node(pivot, "SW"), self._sub_node(pivot, "SE")]
        if d == "W":
            return [self._sub_node(pivot, "NW"), self._sub_node(pivot, "SW")]
        if d == "N":
            return [self._sub_node(pivot, "NE"), self._sub_node(pivot, "NW")]
        raise RuntimeError("round-trip direction error")

    def _intermediate_node(self, p, q):
        p_ngb, q_ngb = set(), set()
        for m, n in self.edge:
            if m == p:
                p_ngb.add(n)
            if n == p:
                p_ngb.add(m)
            if m == q:
                q_ngb.add(n)
            if n == q:
                q_ngb.add(m)

        itsc = p_ngb.intersection(q_ngb)
        if len(itsc) != 1:
            raise RuntimeError("No unique intermediate node found")
        return list(itsc)[0]

@dataclass(frozen=True)
class Lane:
    row: int
    j0: int
    j1: int  # inclusive
    lane_id: int

GridPt = Tuple[int, int]
class BoustrophedonCoveragePlanner:
    def __init__(self, occ_map: np.ndarray, neighborhood: int = 4):
        self.occ_map = occ_map.astype(bool)
        self.H, self.W = self.occ_map.shape
        if neighborhood not in (4, 8):
            raise ValueError("neighborhood must be 4 or 8")
        self.neighborhood = neighborhood

        self.lanes: List[Lane] = []
        self.lanes_by_row: Dict[int, List[Lane]] = {}

    # -------------------------
    # Public API
    # -------------------------
    def plan(
        self,
        start: Optional[GridPt] = None,
        sweep_axis: str = "row",
        connect: bool = True,
    ) -> Tuple[List[Lane], List[GridPt], List[int]]:
        """
        Parameters
        ----------
        start : (i,j) or None
            Start location on the grid. If None, choose the first free cell.
        sweep_axis : "row" or "col"
            Sweep direction. "row" means lanes are horizontal (vary j),
            "col" means lanes are vertical (vary i).
        connect : bool
            If True, connect discontinuous lanes using BFS shortest paths.

        Returns
        -------
        lanes : List[Lane]
        path : List[(i,j)] full coverage path
        order : List[int] lane_ids in visitation order
        """
        if not self.occ_map.any():
            return [], [], []

        if start is None:
            start = tuple(map(int, np.argwhere(self.occ_map)[0]))

        if not self._is_free(start):
            start = self._nearest_free(start)
            if start is None:
                return [], [], []

        # Build lanes
        if sweep_axis.lower().startswith("row"):
            self._build_horizontal_lanes()
            axis = "row"
        elif sweep_axis.lower().startswith("col"):
            self._build_vertical_lanes()
            axis = "col"
        else:
            raise ValueError("sweep_axis must be 'row' or 'col'")

        if not self.lanes:
            return [], [], []

        # Choose first lane: closest lane point to start
        first_lane_idx, first_pt = self._choose_first_lane(start, axis=axis)

        # Visit lanes in boustrophedon order: alternating direction per lane
        order = self._lane_visit_order(first_lane_idx)

        # Build full path by concatenating lane traces, with connectors if needed
        path: List[GridPt] = []
        last_pt: Optional[GridPt] = None

        for k, lane_id in enumerate(order):
            lane = self._lane_by_id(lane_id)

            # decide direction: alternate
            forward = (k % 2 == 0)
            lane_pts = self._lane_points(lane, axis=axis, forward=forward)

            # If we want to start at closest point on first lane
            if k == 0 and first_pt is not None:
                lane_pts = self._rotate_lane_to_start(lane_pts, first_pt)

            # connector from last_pt -> lane_pts[0]
            if last_pt is not None and lane_pts:
                if connect and last_pt != lane_pts[0]:
                    conn = self._shortest_path(last_pt, lane_pts[0])
                    if conn is None:
                        # If no connection exists, we skip connector and "teleport" (or you can stop)
                        # Here: just jump
                        pass
                    else:
                        # append connector excluding duplicate start
                        if conn and path and conn[0] == path[-1]:
                            path.extend(conn[1:])
                        else:
                            path.extend(conn)

            # append lane points
            if lane_pts:
                if path and lane_pts[0] == path[-1]:
                    path.extend(lane_pts[1:])
                else:
                    path.extend(lane_pts)
                last_pt = lane_pts[-1]

        return self.lanes, path, order

    # -------------------------
    # Lane construction
    # -------------------------
    def _build_horizontal_lanes(self) -> None:
        self.lanes = []
        self.lanes_by_row = {}
        lane_id = 0
        for i in range(self.H):
            intervals = self._free_intervals_in_row(i)
            row_lanes = []
            for (j0, j1) in intervals:
                row_lanes.append(Lane(row=i, j0=j0, j1=j1, lane_id=lane_id))
                lane_id += 1
            if row_lanes:
                self.lanes_by_row[i] = row_lanes
                self.lanes.extend(row_lanes)

    def _build_vertical_lanes(self) -> None:
        # Symmetric: treat columns like rows by scanning i intervals
        self.lanes = []
        self.lanes_by_row = {}  # will store by "col index" in this mode
        lane_id = 0
        for j in range(self.W):
            intervals = self._free_intervals_in_col(j)
            col_lanes = []
            for (i0, i1) in intervals:
                # store as Lane but interpret (row=i0..i1) later
                # We'll store row=j (meaning col index), and j0=i0, j1=i1
                col_lanes.append(Lane(row=j, j0=i0, j1=i1, lane_id=lane_id))
                lane_id += 1
            if col_lanes:
                self.lanes_by_row[j] = col_lanes
                self.lanes.extend(col_lanes)

    def _free_intervals_in_row(self, i: int) -> List[Tuple[int, int]]:
        intervals = []
        j = 0
        while j < self.W:
            # skip obstacles
            while j < self.W and not self.occ_map[i, j]:
                j += 1
            if j >= self.W:
                break
            j0 = j
            while j < self.W and self.occ_map[i, j]:
                j += 1
            j1 = j - 1
            intervals.append((j0, j1))
        return intervals

    def _free_intervals_in_col(self, j: int) -> List[Tuple[int, int]]:
        intervals = []
        i = 0
        while i < self.H:
            while i < self.H and not self.occ_map[i, j]:
                i += 1
            if i >= self.H:
                break
            i0 = i
            while i < self.H and self.occ_map[i, j]:
                i += 1
            i1 = i - 1
            intervals.append((i0, i1))
        return intervals

    # -------------------------
    # Lane visitation order
    # -------------------------
    def _choose_first_lane(self, start: GridPt, axis: str) -> Tuple[int, Optional[GridPt]]:
        # pick lane whose points are closest to start
        best_lane_id = self.lanes[0].lane_id
        best_pt = None
        best_d2 = float("inf")

        for lane in self.lanes:
            pts = self._lane_points(lane, axis=axis, forward=True)
            # test a couple of points (endpoints) is enough
            candidates = []
            if pts:
                candidates = [pts[0], pts[-1], pts[len(pts) // 2]]
            for p in candidates:
                d2 = (p[0] - start[0]) ** 2 + (p[1] - start[1]) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_lane_id = lane.lane_id
                    best_pt = p

        return best_lane_id, best_pt

    def _lane_visit_order(self, first_lane_id: int) -> List[int]:
        """
        Simple boustrophedon ordering: visit lanes by increasing sweep index
        starting near first lane's sweep line.
        """
        # lanes are grouped by "row index" in lanes_by_row (row for horizontal, col for vertical)
        # We'll order sweep lines by distance to first lane's sweep index
        first_lane = self._lane_by_id(first_lane_id)
        sweep0 = first_lane.row

        sweep_lines = sorted(self.lanes_by_row.keys(), key=lambda r: abs(r - sweep0))

        # Inside each sweep line, visit its intervals left->right in that line’s scan direction
        order = []
        for idx, sweep in enumerate(sweep_lines):
            row_lanes = self.lanes_by_row[sweep]
            # alternate the *interval ordering* too, for nicer continuity
            if idx % 2 == 0:
                row_lanes_sorted = sorted(row_lanes, key=lambda ln: ln.j0)
            else:
                row_lanes_sorted = sorted(row_lanes, key=lambda ln: ln.j0, reverse=True)
            order.extend([ln.lane_id for ln in row_lanes_sorted])
        # rotate so we start with first_lane_id
        if first_lane_id in order:
            k = order.index(first_lane_id)
            order = order[k:] + order[:k]
        return order

    # -------------------------
    # Path generation helpers
    # -------------------------
    def _lane_points(self, lane: Lane, axis: str, forward: bool) -> List[GridPt]:
        if axis == "row":
            i = lane.row
            js = range(lane.j0, lane.j1 + 1)
            pts = [(i, j) for j in js]
        else:  # axis == "col"
            j = lane.row  # stored "row" = col index
            is_ = range(lane.j0, lane.j1 + 1)  # stored j0/j1 are i0/i1
            pts = [(i, j) for i in is_]

        return pts if forward else list(reversed(pts))

    def _rotate_lane_to_start(self, pts: List[GridPt], start_pt: GridPt) -> List[GridPt]:
        if not pts:
            return pts
        # rotate to nearest occurrence of start_pt (or nearest point if not exact)
        if start_pt in pts:
            k = pts.index(start_pt)
            return pts[k:] + pts[:k]
        # else rotate to closest point
        best_k = 0
        best_d2 = float("inf")
        for k, p in enumerate(pts):
            d2 = (p[0] - start_pt[0]) ** 2 + (p[1] - start_pt[1]) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_k = k
        return pts[best_k:] + pts[:best_k]

    # -------------------------
    # Shortest path connector (BFS)
    # -------------------------
    def _shortest_path(self, start: GridPt, goal: GridPt) -> Optional[List[GridPt]]:
        if start == goal:
            return [start]
        if not self._is_free(start) or not self._is_free(goal):
            return None

        q = deque([start])
        prev = {start: None}

        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for nb in self._neighbors(cur):
                if nb not in prev and self._is_free(nb):
                    prev[nb] = cur
                    q.append(nb)

        if goal not in prev:
            return None

        # reconstruct
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def _neighbors(self, p: GridPt):
        i, j = p
        if self.neighborhood == 4:
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for di, dj in steps:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.H and 0 <= nj < self.W:
                yield (ni, nj)

    # -------------------------
    # Utility
    # -------------------------
    def _is_free(self, p: GridPt) -> bool:
        return bool(self.occ_map[p[0], p[1]])

    def _nearest_free(self, p: GridPt) -> Optional[GridPt]:
        # BFS expanding from p until we find a free cell
        if self._is_free(p):
            return p
        q = deque([p])
        seen = {p}
        while q:
            cur = q.popleft()
            for nb in self._neighbors(cur):
                if nb in seen:
                    continue
                if self._is_free(nb):
                    return nb
                seen.add(nb)
                q.append(nb)
        return None

    def _lane_by_id(self, lane_id: int) -> Lane:
        for ln in self.lanes:
            if ln.lane_id == lane_id:
                return ln
        raise KeyError(f"lane_id {lane_id} not found")


class DirectionalBoustrophedonPlanner:
    """
    Wraps your existing Boustrophedon logic but:
      - allows sweeping at arbitrary angles (via rotating the occ_map)
      - can auto-pick the angle that produces longer lanes (fewer turns)
    """

    def __init__(self, occ_map: np.ndarray, neighborhood: int = 8):
        self.occ_map = occ_map.astype(bool)  # True = free
        self.H, self.W = self.occ_map.shape
        if neighborhood not in (4, 8):
            raise ValueError("neighborhood must be 4 or 8")
        self.neighborhood = neighborhood

    # -------------------------
    # Public API
    # -------------------------
    def plan(
        self,
        start: Optional[GridPt] = None,
        sweep_angle_deg: Optional[float] = None,
        auto_angles_deg: Optional[List[float]] = None,
        connect: bool = True,
    ):
        """
        Parameters
        ----------
        start : (i,j) or None
        sweep_angle_deg : float or None
            If provided, sweep at this angle (degrees).
            0 deg means "horizontal sweep" (lanes left<->right).
        auto_angles_deg : list[float] or None
            If sweep_angle_deg is None, try these angles and pick the best.
            Example: [0, 15, 30, ..., 165]
        connect : bool
            BFS-connect between lanes (in ORIGINAL grid at the end)

        Returns
        -------
        best_angle_deg : float
        lanes_rot : List[Lane] lanes in rotated grid
        path_orig : List[GridPt] final path in original grid
        order_rot : List[int] lane id order (rotated grid)
        """
        if not self.occ_map.any():
            return None, [], [], []

        if start is None:
            start = tuple(map(int, np.argwhere(self.occ_map)[0]))

        if not self._is_free(self.occ_map, start):
            start = self._nearest_free(self.occ_map, start)
            if start is None:
                return None, [], [], []

        # Angle selection
        if sweep_angle_deg is None:
            if auto_angles_deg is None:
                # Good default: sample many angles (you can tighten this for speed)
                auto_angles_deg = list(range(0, 180, 10))
            best = None
            for ang in auto_angles_deg:
                cand = self._plan_at_angle(start, ang, connect=False)  # score only
                if cand is None:
                    continue
                score = cand["score"]
                if best is None or score > best["score"]:
                    best = {**cand, "angle": ang, "score": score}
            if best is None:
                return None, [], [], []
            sweep_angle_deg = best["angle"]
            chosen = best
        else:
            chosen = self._plan_at_angle(start, sweep_angle_deg, connect=False)
            if chosen is None:
                return None, [], [], []

        # Build final path: unrotate rotated path back into original grid and BFS-connect in original
        path_rot = chosen["path_rot"]
        lanes_rot = chosen["lanes_rot"]
        order_rot = chosen["order_rot"]
        rot = chosen["rot_params"]
        start_rot = chosen["start_rot"]

        # Convert rotated grid points -> original grid points (rounded)
        path_orig_pts = [self._rot_pt_to_orig_pt(p, rot) for p in path_rot]
        path_orig_pts = [self._clamp_pt(p) for p in path_orig_pts]

        # Densify & ensure feasibility in original grid with BFS between consecutive points
        if connect:
            path_orig = []
            last = None
            for p in path_orig_pts:
                if last is None:
                    if self._is_free(self.occ_map, p):
                        path_orig.append(p)
                        last = p
                    else:
                        nf = self._nearest_free(self.occ_map, p)
                        if nf is None:
                            continue
                        path_orig.append(nf)
                        last = nf
                    continue

                if p == last:
                    continue

                if not self._is_free(self.occ_map, p):
                    nf = self._nearest_free(self.occ_map, p)
                    if nf is None:
                        continue
                    p = nf

                sp = self._shortest_path(self.occ_map, last, p)
                if sp is None:
                    # no connector: "teleport"
                    path_orig.append(p)
                    last = p
                else:
                    # append excluding duplicate
                    if path_orig and sp[0] == path_orig[-1]:
                        path_orig.extend(sp[1:])
                    else:
                        path_orig.extend(sp)
                    last = p
        else:
            path_orig = path_orig_pts

        return sweep_angle_deg, lanes_rot, path_orig, order_rot

    # -------------------------
    # Core: plan at a given angle (in rotated grid)
    # -------------------------
    def _plan_at_angle(self, start_orig: GridPt, angle_deg: float, connect: bool):
        occ_rot, rot_params = self._rotate_bool_grid_nn(self.occ_map, angle_deg)
        if not occ_rot.any():
            return None

        start_rot = self._orig_pt_to_rot_pt(start_orig, rot_params)
        start_rot = self._clamp_pt_to_shape(start_rot, occ_rot.shape)

        if not self._is_free(occ_rot, start_rot):
            start_rot = self._nearest_free(occ_rot, start_rot)
            if start_rot is None:
                return None

        # Use your existing row-based boustrophedon on rotated grid
        base = BoustrophedonCoveragePlanner(occ_map=occ_rot, neighborhood=self.neighborhood)
        lanes_rot, path_rot, order_rot = base.plan(start=start_rot, sweep_axis="row", connect=connect)

        if not path_rot:
            return None

        # Score: prefer long lanes / fewer turns
        # Simple proxy: total covered points / number_of_lanes  (avg lane length)
        nlanes = max(1, len(lanes_rot))
        score = len(path_rot) / nlanes

        return {
            "lanes_rot": lanes_rot,
            "path_rot": path_rot,
            "order_rot": order_rot,
            "rot_params": rot_params,
            "start_rot": start_rot,
            "score": score,
        }

    # -------------------------
    # Rotation helpers (nearest-neighbor rasterization)
    # -------------------------
    def _rotate_bool_grid_nn(self, grid: np.ndarray, angle_deg: float):
        """
        Rotate boolean grid by angle (deg) around its center.
        Returns rotated grid (NN rasterization) + params to map points back/forth.
        """
        H, W = grid.shape
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        th = math.radians(angle_deg)
        c, s = math.cos(th), math.sin(th)

        ys, xs = np.nonzero(grid)
        if len(xs) == 0:
            return np.zeros((1, 1), dtype=bool), None

        # rotate all free points
        x0 = xs - cx
        y0 = ys - cy
        xr = c * x0 - s * y0
        yr = s * x0 + c * y0

        # new bounds
        minx, maxx = float(xr.min()), float(xr.max())
        miny, maxy = float(yr.min()), float(yr.max())

        newW = int(math.ceil(maxx - minx + 1))
        newH = int(math.ceil(maxy - miny + 1))

        # shift so min maps to 0
        x_shift = -minx
        y_shift = -miny

        xri = np.rint(xr + x_shift).astype(int)
        yri = np.rint(yr + y_shift).astype(int)

        occ_rot = np.zeros((newH, newW), dtype=bool)
        occ_rot[yri, xri] = True

        rot_params = {
            "angle_deg": angle_deg,
            "c": c,
            "s": s,
            "cx": cx,
            "cy": cy,
            "x_shift": x_shift,
            "y_shift": y_shift,
            "orig_shape": (H, W),
            "rot_shape": (newH, newW),
        }
        return occ_rot, rot_params

    def _orig_pt_to_rot_pt(self, p: GridPt, rot):
        i, j = p
        y0 = i - rot["cy"]
        x0 = j - rot["cx"]
        # forward rotate
        xr = rot["c"] * x0 - rot["s"] * y0
        yr = rot["s"] * x0 + rot["c"] * y0
        x = int(round(xr + rot["x_shift"]))
        y = int(round(yr + rot["y_shift"]))
        return (y, x)

    def _rot_pt_to_orig_pt(self, p: GridPt, rot):
        i, j = p
        # remove shift
        xr = j - rot["x_shift"]
        yr = i - rot["y_shift"]
        # inverse rotate by -theta (transpose of rotation matrix)
        x0 = rot["c"] * xr + rot["s"] * yr
        y0 = -rot["s"] * xr + rot["c"] * yr
        x = int(round(x0 + rot["cx"]))
        y = int(round(y0 + rot["cy"]))
        return (y, x)

    # -------------------------
    # Grid utilities (original BFS, reused)
    # -------------------------
    def _neighbors(self, shape, p: GridPt):
        H, W = shape
        i, j = p
        if self.neighborhood == 4:
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for di, dj in steps:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                yield (ni, nj)

    def _is_free(self, occ, p: GridPt) -> bool:
        return bool(occ[p[0], p[1]])

    def _nearest_free(self, occ, p: GridPt) -> Optional[GridPt]:
        if self._is_free(occ, p):
            return p
        q = deque([p])
        seen = {p}
        while q:
            cur = q.popleft()
            for nb in self._neighbors(occ.shape, cur):
                if nb in seen:
                    continue
                if self._is_free(occ, nb):
                    return nb
                seen.add(nb)
                q.append(nb)
        return None

    def _shortest_path(self, occ, start: GridPt, goal: GridPt) -> Optional[List[GridPt]]:
        if start == goal:
            return [start]
        if not self._is_free(occ, start) or not self._is_free(occ, goal):
            return None
        q = deque([start])
        prev = {start: None}
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for nb in self._neighbors(occ.shape, cur):
                if nb not in prev and self._is_free(occ, nb):
                    prev[nb] = cur
                    q.append(nb)
        if goal not in prev:
            return None
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def _clamp_pt(self, p: GridPt) -> GridPt:
        i, j = p
        i = max(0, min(self.H - 1, i))
        j = max(0, min(self.W - 1, j))
        return (i, j)

    def _clamp_pt_to_shape(self, p: GridPt, shape) -> GridPt:
        H, W = shape
        i, j = p
        i = max(0, min(H - 1, i))
        j = max(0, min(W - 1, j))
        return (i, j)
# ----------------------------
# Example usage with your surface mask -> occ_map
# ----------------------------
def surface_mask_to_occ_map(surface_mask: np.ndarray) -> np.ndarray:
    """
    For boustrophedon on the *same* grid resolution as the Environment surface:
    - True = free
    - False = obstacle
    """
    return surface_mask.astype(bool)


# ----------------------------
# Helper: surface mask -> occ_map for STC (2x2 expansion)
# ----------------------------


def surface_to_occ_map(surface_mask: np.ndarray) -> np.ndarray:
    """
    Convert HxW surface mask into (2H)x(2W) occ_map required by Spiral-STC:
    - True (free) where inside surface, False elsewhere.
    """
    H, W = surface_mask.shape
    occ = np.zeros((2 * H, 2 * W), dtype=bool)
    for i in range(H):
        for j in range(W):
            if surface_mask[i, j]:
                occ[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = True
    return occ

# ----------------------------
# Demo
# ----------------------------


import math
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

NED = Tuple[float, float, float]

def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))



class SimpleAUV:
    def __init__(
        self,
        pos: NED = (0.0, 0.0, 0.0),
        trajectory: List[NED] | None = None,
        speed_mps: float = 1.0,
        yaw_rate_max_rps: float = math.radians(15.0),
        wp_radius_m: float = 1.0,
        depth_rate_max_mps: float | None = None,   # set to e.g. 0.2 to limit; None snaps to target depth
        accel_max_mps2: float | None = None,        # set to e.g. 0.3 to limit; None means instant speed
    ):
        self.pos = pos
        self.yaw = 0.0
        self.v = 0.0
        self.trajectory = trajectory or []
        self.traj_index = 0

        self.speed_mps = speed_mps
        self.yaw_rate_max_rps = yaw_rate_max_rps
        self.wp_radius_m = wp_radius_m
        if depth_rate_max_mps is not None:
            self.depth_rate_max_mps = depth_rate_max_mps
        if accel_max_mps2 is not None:
            self.accel_max_mps2 = accel_max_mps2


    def follow_trajectory(self, dt: float = 1.0) -> NED:
        """Waypoint follower with simple AUV kinematics (speed + turn-rate + depth-rate limits)."""
        if not self.trajectory:
            return self.pos

        tN, tE, tD = self.trajectory[self.traj_index]
        N, E, D = self.pos

        # --- guidance: desired yaw to waypoint ---
        dx = tN - N
        dy = tE - E
        dist_xy = math.hypot(dx, dy)

        # waypoint acceptance radius (meters)
        wp_radius = getattr(self, "wp_radius_m", 1.0)

        # If we're close enough in XY, advance waypoint
        if dist_xy <= wp_radius:
            self.traj_index = min(self.traj_index + 1, len(self.trajectory) - 1)
            # keep moving smoothly; don't "snap" position
            return self.pos

        desired_yaw = math.atan2(dy, dx)  # NED: atan2(East, North) if using (dy, dx) with N,E
        # NOTE: since we used dy=tE-E and dx=tN-N, atan2(dy, dx) gives yaw where 0=North, +pi/2=East.

        # --- yaw kinematics: limit turn rate ---
        yaw_rate_max = getattr(self, "yaw_rate_max_rps", math.radians(15.0))  # rad/s
        yaw_err = wrap_pi(desired_yaw - getattr(self, "yaw", 0.0))
        yaw_step = clamp(yaw_err, -yaw_rate_max * dt, yaw_rate_max * dt)
        self.yaw = wrap_pi(getattr(self, "yaw", 0.0) + yaw_step)

        # --- speed (optional accel limit) ---
        v_cmd = getattr(self, "speed_mps", 1.0)
        if hasattr(self, "accel_max_mps2"):
            v = getattr(self, "v", v_cmd)
            amax = self.accel_max_mps2
            v = v + clamp(v_cmd - v, -amax * dt, amax * dt)
            self.v = max(0.0, v)
        else:
            self.v = v_cmd

        # --- integrate horizontal motion using yaw ---
        # N_dot = v*cos(yaw), E_dot = v*sin(yaw)
        N += self.v * math.cos(self.yaw) * dt
        E += self.v * math.sin(self.yaw) * dt

        # --- depth tracking: limit vertical rate (optional) ---
        depth_rate_max = getattr(self, "depth_rate_max_mps", None)
        if depth_rate_max is None:
            D = tD  # your original "snap depth" behavior
        else:
            # move depth toward target at limited rate
            d_err = tD - D
            D += clamp(d_err, -depth_rate_max * dt, depth_rate_max * dt)

        self.pos = (N, E, D)
        return self.pos


def animate_auv_3d(path_orig, occ_map=None, resolution_m=0.01, depth_m=10.0, step_per_frame=1):
    """
    Animate vehicle motion in 3D from a boustrophedon grid path.

    Parameters
    ----------
    path_orig : list[(i,j)]
        Output path from planner (original grid coords).
    occ_map : np.ndarray[H,W] bool or None
        If provided, can be used to set plot bounds (True=free).
    resolution_m : float
        Meters per grid cell.
    depth_m : float
        Constant depth used for animation (D positive down).
    step_per_frame : int
        How many path points to advance each animation frame (speed-up).
    """
    if not path_orig:
        raise ValueError("path_orig is empty")

    pts = np.asarray(path_orig, dtype=float)  # (i,j)
    # grid -> meters: N ~ i*res, E ~ j*res
    Ns = pts[:, 0] * resolution_m
    Es = pts[:, 1] * resolution_m
    Zs = np.full_like(Ns, -float(depth_m))  # plot Up = -Depth

    # bounds
    pad = 5.0 * resolution_m
    if occ_map is not None and np.any(occ_map):
        free = np.argwhere(occ_map.astype(bool))
        Nf = free[:, 0] * resolution_m
        Ef = free[:, 1] * resolution_m
        x_min, x_max = float(Nf.min()) - pad, float(Nf.max()) + pad
        y_min, y_max = float(Ef.min()) - pad, float(Ef.max()) + pad
    else:
        x_min, x_max = float(Ns.min()) - pad, float(Ns.max()) + pad
        y_min, y_max = float(Es.min()) - pad, float(Es.max()) + pad

    z_min, z_max = float(Zs.min()) - pad, float(max(0.0, Zs.max())) + pad

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Boustrophedon Path Animation (N, E, -D)")
    ax.set_xlabel("North (m)")
    ax.set_ylabel("East (m)")
    ax.set_zlabel("Up (m) = -Depth")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Optional: show free-space footprint as a faint plane of points (can be heavy if huge map)
    if occ_map is not None:
        free = np.argwhere(occ_map.astype(bool))
        if len(free) <= 20000:  # safety cap
            Nf = free[:, 0] * resolution_m
            Ef = free[:, 1] * resolution_m
            Zf = np.full_like(Nf, 0.0)  # draw at "surface" plane
            ax.scatter(Nf, Ef, Zf, s=1)

    (path_line,) = ax.plot([], [], [])
    (auv_pt,) = ax.plot([], [], [], marker="o")
    (head_line,) = ax.plot([], [], [])

    trailN, trailE, trailZ = [], [], []
    idx = {"k": 0}

    def init():
        path_line.set_data([], [])
        path_line.set_3d_properties([])
        auv_pt.set_data([], [])
        auv_pt.set_3d_properties([])
        head_line.set_data([], [])
        head_line.set_3d_properties([])
        return path_line, auv_pt, head_line

    def update(_frame):
        k = idx["k"]
        if k >= len(Ns):
            anim.event_source.stop()
            return path_line, auv_pt, head_line

        N, E, Z = float(Ns[k]), float(Es[k]), float(Zs[k])

        trailN.append(N)
        trailE.append(E)
        trailZ.append(Z)

        path_line.set_data(trailN, trailE)
        path_line.set_3d_properties(trailZ)

        auv_pt.set_data([N], [E])
        auv_pt.set_3d_properties([Z])

        # heading from current -> next (or previous if at end)
        k2 = min(k + 1, len(Ns) - 1)
        dN = float(Ns[k2] - Ns[k])
        dE = float(Es[k2] - Es[k])
        yaw = math.atan2(dE, dN) if (dN != 0.0 or dE != 0.0) else 0.0

        L = 2.0 * resolution_m
        hx = N + L * math.cos(yaw)
        hy = E + L * math.sin(yaw)
        head_line.set_data([N, hx], [E, hy])
        head_line.set_3d_properties([Z, Z])

        idx["k"] += int(step_per_frame)
        return path_line, auv_pt, head_line

    anim = FuncAnimation(fig, update, init_func=init, interval=1000, blit=False)
    plt.show()

if __name__ == "__main__":
    from environment import Environment
    env = Environment(shape=(600, 800), resolution=0.1, origin=(0.0, 0.0))
    env.Initial_area[0:500, 0:700] = True
    env.Covered_area[12:19, 25:15] = True
    env.Collision_area[10:15, 10:15] = True
    env.Within_range[40:50, 10:35] = True

    seed_ij = env.select_seed_ij(start_location=(200, 300))
    mask, idx = env.grow_surface_from_seed(
        seed_ij=seed_ij, S=100, neighborhood=8,
        w_range=1.0, w_safe=1.0, w_explore=0.0, w_exploit=0.8,
        min_safe_dist_m=5, w_compactness=0.5
    )

    occ = surface_mask_to_occ_map(mask)

    planner = BoustrophedonCoveragePlanner(occ_map=occ, neighborhood=8)
    lanes, path, order = planner.plan(start=seed_ij, sweep_axis="row")
    print(f"Planned path {path} ")

    animate_auv_3d(path, occ_map=occ, resolution_m=0.1, depth_m=10.0, step_per_frame=2)











'''
def main():
    np.random.seed(3)

    # 1) Build an Environment with some simple masks
    env = Environment(shape=(60, 80), resolution=1.0)

    # Initial area: everything allowed
    env.Initial_area[:, :] = True

    # Collision: random rectangles
    env.Collision_area[10:18, 10:28] = True
    env.Collision_area[30:45, 40:55] = True
    env.Collision_area[20:28, 60:75] = True

    # Covered: imagine we already covered a left-side block
    env.Covered_area[20:40, 5:20] = True

    # Within_range: a disk around some point (centered near the covered block)
    ci, cj = 30, 20
    ii, jj = np.indices(env.shape)
    env.Within_range = ((ii - ci) ** 2 + (jj - cj) ** 2) <= (22 ** 2)

    # 2) Grow a connected surface near a start location (grid indices)
    start_ij = (28, 22)
    surface = env.grow_surface_from_start(
        start_location=start_ij,
        S=350,
        neighborhood=8,
        w_range=1.0,
        w_safe=2.0,
        w_frontier=1.5,
        w_near=1.0,
        w_compact=2.0,
        prefer_touch_covered=True,
        min_safe_dist_m=2.0,
    )

    # 3) Convert surface mask to STC occupancy map and plan coverage
    occ_map = surface_to_occ_map(surface)

    stc = SpiralSpanningTreeCoveragePlanner(occ_map)

    # STC start is in merged coordinates (H x W), which matches env grid indices
    stc_start = start_ij
    edge, route, path = stc.plan(stc_start)

    # 4) Visualize: surface + coverage path (simple plot)
    plt.figure(figsize=(10, 5))
    plt.title("Grown surface (blue) + obstacles (black) + covered (green)")
    img = np.zeros(env.shape, dtype=float)
    img[env.Collision_area] = 0.0
    img[env.Initial_area & ~env.Collision_area] = 0.2
    img[env.Within_range] = np.maximum(img[env.Within_range], 0.35)
    img[env.Covered_area] = 0.6
    img[surface] = 0.9
    plt.imshow(img, origin="upper")
    plt.scatter([start_ij[1]], [start_ij[0]], marker="x", s=80)
    plt.colorbar(label="(visual code)")

    # Overlay the fine-grid STC path on the 2H x 2W map and scale to env grid for viewing
    # Each fine node is in [0..2H-1, 0..2W-1]; divide by 2 to plot in env grid coordinates.
    traj= []
    if path:
        last = path[0][0]
        traj.append((last[0] / 2.0, last[1] / 2.0))
        for p, q in path:
            traj.append((p[0] / 2.0, p[1] / 2.0))
            traj.append((q[0] / 2.0, q[1] / 2.0))
        xs = [t[1] for t in traj]
        ys = [t[0] for t in traj]
        plt.plot(xs, ys, linewidth=1.5)

    plt.tight_layout()
    plt.show()

def main_minimal():
    env = Environment(shape=(600, 800), resolution=0.1, origin=(0.0, 0.0))
    env.Initial_area[0:500, 0:700] = True
    env.Covered_area[12:19, 25:15] = True
    env.Collision_area[10:15, 10:15] = True
    env.Within_range[40:50, 10:35] = True

    seed_ij = env.select_seed_ij(start_location=(200, 300))
    mask, idx = env.grow_surface_from_seed(
        seed_ij=seed_ij, S=200, neighborhood=8,
        w_range=1.0, w_safe=1.0, w_explore=0.0, w_exploit=0.8,
        min_safe_dist_m=5, w_compactness=0.5
    )

    occ = surface_mask_to_occ_map(mask)

    planner = BoustrophedonCoveragePlanner(occ_map=occ, neighborhood=8)
    lanes, path, order = planner.plan(start=seed_ij, sweep_axis="row")

    ini = surface_to_occ_map(env.Initial_area).astype(bool)
    cov = surface_to_occ_map(env.Covered_area).astype(bool)

    plt.figure(figsize=(10, 4))

    # Background
    plt.imshow(np.zeros_like(occ, dtype=float), origin="upper")

    # Layered overlays (so nothing gets overwritten)
    plt.imshow(occ.astype(float), origin="upper", alpha=0.25)  # free space (from grown mask)
    #plt.imshow(ini.astype(float), origin="upper", alpha=0.45)  # initial area
    #plt.imshow(cov.astype(float), origin="upper", alpha=0.75)  # covered area

    # Path
    if path:
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        plt.plot(xs, ys, linewidth=1.5)
        plt.scatter([xs[0]], [ys[0]], marker="x", s=80)

    plt.title("Boustrophedon sweep with Initial & Covered areas")
    plt.axis("equal")
    plt.tight_layout()

    print(f"lanes: {lanes}")
    print(f"order: {order}")
    plt.show()
if __name__ == "__main__":
    main_minimal()

'''