import numpy as np
from scipy.ndimage import distance_transform_edt
import heapq
import math


class Environment:
    def __init__(self, shape, resolution, origin=(0.0, 0.0)):
        """
        shape: (H, W) grid size
        resolution: meters per cell
        origin: (N0, E0) world coords of grid[0,0] (cell corner convention)
        """
        self.shape = tuple(shape)
        self.resolution = float(resolution)
        self.origin = (float(origin[0]), float(origin[1]))

        self.Initial_area = np.zeros(self.shape, dtype=bool)
        self.Within_range = np.zeros(self.shape, dtype=bool)
        self.Collision_area = np.zeros(self.shape, dtype=bool)
        self.Covered_area = np.zeros(self.shape, dtype=bool)

    # -------------------------
    # Region-growing heuristic
    # -------------------------
    def grow_surface_from_seed(
        self,
        A_start=None,
        S=200,
        seed_ij=(0,0),
        neighborhood=8,
        # weights for the score terms:
        w_range=1.0,
        w_safe=2.0,
        w_frontier=1.0,
        w_near=1.0,
        w_compact=1.0,
        # constraints / options:
        min_safe_dist_m=0.0,
        require_seed_in_surface=True,
        prefer_touch_covered=True,
        force_touch_covered=False,
        min_range_fraction=None,
    ):
        """
        Build a connected surface (mask) of exactly S cells by best-first region growing.

        A_start: (N, E) point that should be inside / near the surface (used to pick seed if seed_ij not given)
        S: number of cells in the surface
        seed_ij: (i, j) grid index to start from; overrides A_start if provided
        neighborhood: 4 or 8 neighbors
        Weights:
          - w_range: reward for being inside Within_range
          - w_safe: reward for being far from Collision_area
          - w_frontier: reward for sharing border with Covered_area
          - w_near: penalty for being far from A_start
          - w_compact: reward for adding cells adjacent to already-chosen cells

        min_safe_dist_m: reject cells closer than this to collision cells
        prefer_touch_covered: include frontier reward in score
        force_touch_covered: if True, try hard to ensure the final region touches Covered_area
                             (uses a simple “two-phase” fallback if needed)
        min_range_fraction: if not None, enforce at least this fraction of cells inside Within_range

        Returns:
          surface_mask: np.ndarray[bool] with shape self.shape (True = selected)
        """
        H, W = self.shape
        if S <= 0:
            return np.zeros(self.shape, dtype=bool)

        if neighborhood not in (4, 8):
            raise ValueError("neighborhood must be 4 or 8")

        # --- feasibility mask ---
        feasible = self.Initial_area & ~self.Collision_area
        # Range overlap is an objective, but usually you still want to stay in Within_range for operations.
        # If you want to allow growth outside Within_range, change the next line to: feasible = self.Initial_area & ~self.Collision_area
        feasible &= self.Within_range

        if not feasible.any():
            return np.zeros(self.shape, dtype=bool)

        # --- choose seed ---
        if seed_ij is None:
            if A_start is None:
                # fall back: pick any feasible cell closest to covered frontier if possible
                # else just pick the first feasible cell
                seed_ij = tuple(map(int, np.argwhere(feasible)[0]))
            else:
                si, sj = self.world_to_grid(A_start[0], A_start[1])
                # clamp
                si = max(0, min(H - 1, si))
                sj = max(0, min(W - 1, sj))
                if feasible[si, sj]:
                    seed_ij = (si, sj)
                else:
                    # choose nearest feasible cell to A (Euclidean in grid)
                    coords = np.argwhere(feasible)
                    # minimize squared distance to (si,sj)
                    d2 = (coords[:, 0] - si) ** 2 + (coords[:, 1] - sj) ** 2
                    k = int(np.argmin(d2))
                    seed_ij = (int(coords[k, 0]), int(coords[k, 1]))
        else:
            si, sj = int(seed_ij[0]), int(seed_ij[1])
            if not (0 <= si < H and 0 <= sj < W):
                raise ValueError("seed_ij out of bounds")
            if not feasible[si, sj]:
                # move to nearest feasible cell
                coords = np.argwhere(feasible)
                d2 = (coords[:, 0] - si) ** 2 + (coords[:, 1] - sj) ** 2
                k = int(np.argmin(d2))
                seed_ij = (int(coords[k, 0]), int(coords[k, 1]))

        # --- distance to collision (safety field) ---
        # distance_transform_edt returns distance in cells to nearest True in the input.
        # We want distance to collision cells, so compute dt on ~Collision_area.
        dist_to_collision_m = distance_transform_edt(~self.Collision_area) * self.resolution

        # hard safety buffer
        if min_safe_dist_m > 0.0:
            feasible &= dist_to_collision_m >= float(min_safe_dist_m)
            if not feasible.any():
                return np.zeros(self.shape, dtype=bool)
            if not feasible[seed_ij]:
                coords = np.argwhere(feasible)
                d2 = (coords[:, 0] - seed_ij[0]) ** 2 + (coords[:, 1] - seed_ij[1]) ** 2
                k = int(np.argmin(d2))
                seed_ij = (int(coords[k, 0]), int(coords[k, 1]))

        # --- range reward (binary) ---
        range_reward = self.Within_range.astype(np.float32)  # 1 in range else 0

        # --- frontier reward: touches covered area ---
        # f[i]=1 if any neighbor is covered
        frontier_touch = np.zeros(self.shape, dtype=bool)
        if prefer_touch_covered or force_touch_covered:
            cov = self.Covered_area
            # compute adjacency to covered via shifts
            shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if neighborhood == 8:
                shifts += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for di, dj in shifts:
                shifted = np.zeros_like(cov)
                if di >= 0:
                    i_src = slice(0, H - di)
                    i_dst = slice(di, H)
                else:
                    i_src = slice(-di, H)
                    i_dst = slice(0, H + di)
                if dj >= 0:
                    j_src = slice(0, W - dj)
                    j_dst = slice(dj, W)
                else:
                    j_src = slice(-dj, W)
                    j_dst = slice(0, W + dj)
                shifted[i_dst, j_dst] = cov[i_src, j_src]
                frontier_touch |= shifted
        frontier_reward = frontier_touch.astype(np.float32)

        # --- distance-to-A penalty map ---
        if A_start is None:
            # if user says no A, set nearA penalty to 0 everywhere
            dist_to_A_m = np.zeros(self.shape, dtype=np.float32)
        else:
            Ai, Aj = self.world_to_grid(A_start[0], A_start[1])
            Ai = max(0, min(H - 1, Ai))
            Aj = max(0, min(W - 1, Aj))
            ii, jj = np.indices(self.shape)
            dist_to_A_m = np.sqrt((ii - Ai) ** 2 + (jj - Aj) ** 2).astype(np.float32) * self.resolution

        # Normalize helper (avoid division by zero)
        def _norm01(arr, mask=None):
            if mask is not None and mask.any():
                vals = arr[mask]
            else:
                vals = arr
            mn = float(np.min(vals))
            mx = float(np.max(vals))
            if mx - mn < 1e-9:
                return np.zeros_like(arr, dtype=np.float32)
            return ((arr - mn) / (mx - mn)).astype(np.float32)

        feasible_mask = feasible
        safe_norm = _norm01(dist_to_collision_m.astype(np.float32), feasible_mask)
        nearA_norm = _norm01(dist_to_A_m.astype(np.float32), feasible_mask)

        # neighbor offsets
        neigh4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neigh8 = neigh4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        offsets = neigh8 if neighborhood == 8 else neigh4
        deg = len(offsets)

        def neighbors(i, j):
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    yield ni, nj

        # surface selection mask
        selected = np.zeros(self.shape, dtype=bool)

        # compactness: number of selected neighbors for a candidate
        def compactness(i, j):
            cnt = 0
            for ni, nj in neighbors(i, j):
                if selected[ni, nj]:
                    cnt += 1
            return cnt / deg

        # base cell score (without compactness)
        # score = +range +safe +frontier -nearA +compact
        def score(i, j):
            s = 0.0
            s += w_range * float(range_reward[i, j])
            s += w_safe * float(safe_norm[i, j])
            if prefer_touch_covered:
                s += w_frontier * float(frontier_reward[i, j])
            if A_start is not None:
                s -= w_near * float(nearA_norm[i, j])
            s += w_compact * float(compactness(i, j))
            return s

        # Priority queue of boundary candidates: (-score, (i,j))
        heap = []
        in_heap = np.zeros(self.shape, dtype=bool)

        def push(i, j):
            if selected[i, j] or in_heap[i, j]:
                return
            if not feasible_mask[i, j]:
                return
            heapq.heappush(heap, (-score(i, j), (i, j)))
            in_heap[i, j] = True

        # init with seed
        if require_seed_in_surface:
            selected[seed_ij] = True

        for ni, nj in neighbors(*seed_ij):
            push(ni, nj)

        # Track range fraction if requested
        range_count = int(selected[seed_ij] and self.Within_range[seed_ij])
        target_range_min = None
        if min_range_fraction is not None:
            min_range_fraction = float(min_range_fraction)
            min_range_fraction = max(0.0, min(1.0, min_range_fraction))
            target_range_min = int(math.ceil(min_range_fraction * S))

        # Grow
        while selected.sum() < S and heap:
            _, (i, j) = heapq.heappop(heap)
            in_heap[i, j] = False

            if selected[i, j] or not feasible_mask[i, j]:
                continue

            # Optional: enforce minimum range fraction during growth (softly as a filter)
            if target_range_min is not None:
                remaining = S - int(selected.sum())
                # if we must still collect range cells, avoid picking non-range when it would make it impossible
                need = target_range_min - range_count
                if need > 0:
                    # If we pick a non-range cell now, can we still reach the needed range cells?
                    # Conservative check: remaining-1 picks left must accommodate need
                    if (not self.Within_range[i, j]) and (remaining - 1) < need:
                        continue

            # add cell
            selected[i, j] = True
            if self.Within_range[i, j]:
                range_count += 1

            # expand boundary
            for ni, nj in neighbors(i, j):
                push(ni, nj)

        # If we didn’t reach S due to constraints, return what we have
        # (you can also choose to raise an error, but returning partial is often more useful)
        # Now enforce “touch covered” if requested
        if force_touch_covered and self.Covered_area.any():
            touches = False
            sel_coords = np.argwhere(selected)
            for (i, j) in sel_coords:
                for ni, nj in neighbors(i, j):
                    if self.Covered_area[ni, nj]:
                        touches = True
                        break
                if touches:
                    break

            if not touches:
                # Simple fallback: add a short “connector” path from selected to covered-adjacent,
                # then re-grow to size S around it.
                selected = self._connect_and_regrow_to_touch_covered(
                    selected=selected,
                    feasible=feasible_mask,
                    dist_to_collision_m=dist_to_collision_m,
                    A_start=A_start,
                    S=S,
                    offsets=offsets,
                    weights=(w_range, w_safe, w_frontier, w_near, w_compact),
                    min_safe_dist_m=min_safe_dist_m,
                    neighborhood=neighborhood,
                )

        return selected

    def _connect_and_regrow_to_touch_covered(
        self,
        selected,
        feasible,
        dist_to_collision_m,
        A_start,
        S,
        offsets,
        weights,
        min_safe_dist_m,
        neighborhood,
    ):
        """
        Internal helper: if region doesn't touch covered, connect it with a low-cost path then regrow.
        This is a lightweight heuristic, not an exact optimizer.
        """
        H, W = self.shape
        w_range, w_safe, w_frontier, w_near, w_compact = weights

        # Build target set: cells adjacent to Covered_area
        cov = self.Covered_area
        shifts = offsets
        adj_cov = np.zeros_like(cov, dtype=bool)
        for di, dj in shifts:
            shifted = np.zeros_like(cov)
            if di >= 0:
                i_src = slice(0, H - di)
                i_dst = slice(di, H)
            else:
                i_src = slice(-di, H)
                i_dst = slice(0, H + di)
            if dj >= 0:
                j_src = slice(0, W - dj)
                j_dst = slice(dj, W)
            else:
                j_src = slice(-dj, W)
                j_dst = slice(0, W + dj)
            shifted[i_dst, j_dst] = cov[i_src, j_src]
            adj_cov |= shifted

        adj_cov &= feasible
        if not adj_cov.any():
            return selected

        # Multi-source Dijkstra from current selected region to adj_cov, with a cost favoring safety + range
        # Cost per cell: low cost if safe & in range, high cost if near collision
        safe = dist_to_collision_m.astype(np.float32)
        safe_norm = (safe - safe.min()) / (safe.max() - safe.min() + 1e-9)
        # cost: prefer high safe_norm => low cost
        cost = 1.0 + (1.0 - safe_norm) * 5.0
        # prefer within range (already feasible includes Within_range in this class setup, but keep generality)
        cost *= np.where(self.Within_range, 1.0, 2.0)

        # Dijkstra
        INF = 1e18
        dist = np.full(self.shape, INF, dtype=np.float64)
        prev = np.full((H, W, 2), -1, dtype=np.int32)
        heap = []

        sources = np.argwhere(selected)
        for i, j in sources:
            dist[i, j] = 0.0
            heapq.heappush(heap, (0.0, int(i), int(j)))

        target_mask = adj_cov
        target = None

        while heap:
            dcur, i, j = heapq.heappop(heap)
            if dcur != dist[i, j]:
                continue
            if target_mask[i, j]:
                target = (i, j)
                break
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and feasible[ni, nj]:
                    nd = dcur + float(cost[ni, nj])
                    if nd < dist[ni, nj]:
                        dist[ni, nj] = nd
                        prev[ni, nj, 0] = i
                        prev[ni, nj, 1] = j
                        heapq.heappush(heap, (nd, ni, nj))

        if target is None:
            return selected

        # Reconstruct path
        pi, pj = target
        path = []
        while prev[pi, pj, 0] != -1:
            path.append((pi, pj))
            ni, nj = int(prev[pi, pj, 0]), int(prev[pi, pj, 1])
            pi, pj = ni, nj
        path.reverse()

        # Add path cells into selected
        for i, j in path:
            selected[i, j] = True

        # If we overshot size, keep only S by trimming worst cells (rare), else regrow from updated selected
        if selected.sum() > S:
            # crude trim: remove farthest-from-A first if A_start exists, else remove lowest-safety first
            coords = np.argwhere(selected)
            if A_start is not None:
                Ai, Aj = self.world_to_grid(A_start[0], A_start[1])
                Ai = max(0, min(H - 1, Ai))
                Aj = max(0, min(W - 1, Aj))
                keys = (coords[:, 0] - Ai) ** 2 + (coords[:, 1] - Aj) ** 2
            else:
                keys = -dist_to_collision_m[coords[:, 0], coords[:, 1]]
            order = np.argsort(keys)[::-1]  # worst first
            remove_n = int(selected.sum() - S)
            for idx in order[:remove_n]:
                i, j = int(coords[idx, 0]), int(coords[idx, 1])
                selected[i, j] = False
            return selected

        # Regrow to fill to S using the public method but seeding from an existing selected set:
        # easiest: pick a seed inside selected and grow again, treating current selected as pre-selected.
        # We'll do a lightweight fill: greedy add boundary neighbors with safety preference.
        offsets = offsets
        feasible = feasible

        # precompute a simple priority: safe + frontier
        safe = dist_to_collision_m.astype(np.float32)
        safe_norm = (safe - safe.min()) / (safe.max() - safe.min() + 1e-9)

        def neighbors(i, j):
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    yield ni, nj

        heap = []
        in_heap = np.zeros(self.shape, dtype=bool)

        def frontier_touch(i, j):
            for ni, nj in neighbors(i, j):
                if self.Covered_area[ni, nj]:
                    return 1.0
            return 0.0

        def push(i, j):
            if selected[i, j] or in_heap[i, j] or (not feasible[i, j]):
                return
            sc = 2.0 * float(safe_norm[i, j]) + float(frontier_touch(i, j))
            heapq.heappush(heap, (-sc, (i, j)))
            in_heap[i, j] = True

        # initialize boundary candidates
        coords = np.argwhere(selected)
        for i, j in coords:
            for ni, nj in neighbors(int(i), int(j)):
                push(ni, nj)

        while selected.sum() < S and heap:
            _, (i, j) = heapq.heappop(heap)
            in_heap[i, j] = False
            if selected[i, j] or not feasible[i, j]:
                continue
            selected[i, j] = True
            for ni, nj in neighbors(i, j):
                push(ni, nj)

        return selected
