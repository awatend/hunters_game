from __future__ import annotations

import numpy as np
import heapq
from scipy.ndimage import distance_transform_edt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
#from coverage_planner import BoustrophedonCoveragePlanner


class Environment:
    def __init__(self, shape, resolution, origin=(0.0, 0.0)):
        """
        Parameters
        ----------
        shape : (H, W)
            Grid size (rows, cols)
        resolution : float
            Cell size in meters
        origin : (N0, E0)
            World coordinates of grid[0,0]
        """
        self.shape = shape
        self.resolution = float(resolution)
        self.origin = origin
        self.target=None #treasure location
        self.data = np.zeros(shape, dtype=np.float32) #Heatmap of prior knowldge f where the treasure is

        # --- Core regions ---
        self.Initial_area   = np.zeros(shape, dtype=bool)
        self.Within_range   = np.zeros(shape, dtype=bool)
        self.Collision_area = np.zeros(shape, dtype=bool)
        self.static_Collision_area = np.zeros(shape, dtype=bool)
        self.dynamic_Collision_area = np.zeros(shape, dtype=bool)
        self.Covered_area   = np.zeros(shape, dtype=bool)
        self.total_Covered_area = np.zeros(shape, dtype=bool)


    # ------------------------------------------------
    # Utility functions
    # ------------------------------------------------
    def reset_coverage(self):
        self.Covered_area[:] = False
        self.total_Covered_area[:] = False

    def available_area(self):
        """
        Area where motion is allowed:
        Initial ∩ Within_range ∩ not Collision
        """
        return (
            self.Initial_area
            & ~self.Collision_area
        )

    def uncovered_area(self):
        """
        Area that is allowed but not yet covered
        """
        return self.available_area() & ~self.total_Covered_area

    def set_collision_area(self):
        self.collision_area = self.static_Collision_area | self.dynamic_Collision_area



    def update(self):
        rows, cols = self.shape
        self.data[:] = 0.0

        # Grid of coordinates
        y, x = np.mgrid[0:rows, 0:cols]

        # Random number of blobs
        n_blobs = np.random.randint(3, 10)

        for _ in range(n_blobs):
            # Random blob center
            cx = np.random.uniform(0, cols)
            cy = np.random.uniform(0, rows)

            # Random blob size
            sigma_x = np.random.uniform(cols * 0.05, cols * 0.15)
            sigma_y = np.random.uniform(rows * 0.05, rows * 0.15)

            # Random blob intensity
            amplitude = np.random.uniform(0.4, 1.0)

            # Gaussian blob
            blob = amplitude * np.exp(
                -(((x - cx) ** 2) / (2 * sigma_x ** 2) +
                  ((y - cy) ** 2) / (2 * sigma_y ** 2))
            )

            self.data += blob

        # Normalize to [0, 1]
        self.data -= self.data.min()
        if self.data.max() > 0:
            self.data /= self.data.max()

        self.data = self.data.astype(np.float32)


    def compute_score_fields(self):
        covered = self.total_Covered_area.astype(bool)
        within = self.Within_range.astype(bool)
        collision = self.Collision_area.astype(bool)

        sigma_explore_m = 50.0
        sigma_exploit_m = 5.0
        sigma_range_m = 5.0
        sigma_safe_m = 5.0

        if covered.any():
            dist_to_covered_m = distance_transform_edt(~covered) * float(self.resolution)
            exploit_field = np.exp(-(dist_to_covered_m / sigma_exploit_m) ** 2).astype(np.float32) * (1 - self.data)
            exploit_field[covered] = 0.0
            explore_field = self.data
        else:
            exploit_field = np.full(self.shape, 0.5, dtype=np.float32)
            explore_field = self.data

        if within.any():
            dist_to_range_m = distance_transform_edt(~within) * float(self.resolution)
            range_field = np.exp(-(dist_to_range_m / sigma_range_m) ** 2).astype(np.float32)
        else:
            range_field = np.zeros(self.shape, dtype=np.float32)

        if collision.any():
            dist_to_collision_m = distance_transform_edt(~collision) * float(self.resolution)
            safe_field = np.exp(-(dist_to_collision_m / sigma_safe_m) ** 2).astype(np.float32)
        else:
            safe_field = np.ones(self.shape, dtype=np.float32)  # All cells are safe if no collision

        # Normalize all fields to [0, 1]
        def normalize_field(field):
            min_val = field.min()
            max_val = field.max()
            if max_val - min_val < 1e-9:
                return np.ones_like(field, dtype=np.float32) * 0.5
            return ((field - min_val) / (max_val - min_val)).astype(np.float32)

        range_field = normalize_field(range_field)
        safe_field = normalize_field(safe_field)
        explore_field = normalize_field(explore_field)
        exploit_field = normalize_field(exploit_field)

        return range_field, safe_field, explore_field, exploit_field

    def calculate_score(self, i, j, w_range, w_safety, w_explore, w_exploit,
                        range_field, safe_field, explore_field, exploit_field):
        """
        Calculate score for a cell (i, j).
        All field components should be normalized to [0, 1].
        """
        return (
            w_range * range_field[i, j]
            + w_safety * safe_field[i, j]
            + w_explore * explore_field[i, j]
            + w_exploit * exploit_field[i, j]
        )

    def select_seed_ijs(
            self,
            start_location=None,w_range=1.0,w_safe=1.0,w_explore=1.0,w_exploit=1.0
    ):
        """
        Pick a seed cell (i,j) in grid coordinates.
        Prefers cells with good neighborhood (most neighbors are candidates).
        """
        range_field, safe_field, explore_field, exploit_field = self.compute_score_fields()
        N_samples = 200
        H, W = self.shape

        def get_neighborhood_mask_fast(i, j, size):
            """Returns boolean mask of neighborhood cells around (i,j) using numpy slicing"""
            nb_mask = np.zeros((H, W), dtype=bool)
            i_min = max(0, i - size)
            i_max = min(H, i + size + 1)
            j_min = max(0, j - size + 1)
            j_max = min(W, j + size + 1)
            nb_mask[i_min:i_max, j_min:j_max] = True
            nb_mask[i, j] = False  # exclude center
            return nb_mask

        # Make masks boolean to avoid problems with ~ on int arrays
        candidates = (
                self.Initial_area.astype(bool)
                & ~self.Collision_area.astype(bool)
                & ~self.total_Covered_area.astype(bool)
        )

        if start_location is not None:
            si = max(0, min(H - 1, int(start_location[0])))
            sj = max(0, min(W - 1, int(start_location[1])))

            r = self.coverage_ratio()
            size = int(max(H, W) * 3 * (np.exp(2 * r) - 1) / (np.exp(2) - 1) + 1)
            #size = int(max(H, W)/4)
            location_neighborhood = get_neighborhood_mask_fast(si, sj, size)
            candidates = candidates & location_neighborhood
            #print(f"size: {size},  ratio: {r}")

        if not candidates.any():
            return (0, 0)

        coords = np.argwhere(candidates)
        rng = np.random.default_rng()

        # Sample without replacement if possible
        n = min(N_samples, len(coords))
        sampled_idx = rng.choice(len(coords), size=n, replace=False)

        best_seed = None
        best_score = -np.inf

        for k in sampled_idx:
            i, j = map(int, coords[k])

            # Vectorized neighborhood scoring
            i_min, i_max = max(0, i - 5), min(H, i + 6)
            j_min, j_max = max(0, j - 5), min(W, j + 6)

            scores = (
                w_range * range_field[i_min:i_max, j_min:j_max]
                + w_safe * safe_field[i_min:i_max, j_min:j_max]
                + w_explore * explore_field[i_min:i_max, j_min:j_max]
                + w_exploit * exploit_field[i_min:i_max, j_min:j_max]
            ).ravel()

            if start_location is not None:
                ii = np.arange(i_min, i_max)[:, np.newaxis]
                jj = np.arange(j_min, j_max)[np.newaxis, :]
                dist = np.sqrt((ii - start_location[0]) ** 2 + (jj - start_location[1]) ** 2)
                w_dist = 0.001
                #scores = scores - w_dist * dist.ravel()

            s = np.mean(scores)

            if s > best_score and s>0:
                best_score = s
                best_seed = (i, j)
                #print(f"safety: {safe_field[i, j]:.4f}, range: {range_field[i, j]:.4f}, explore: {explore_field[i, j]:.4f}, exploit: {exploit_field[i, j]:.4f}")
        print(f"Selected seed: {best_seed} with score {best_score:.4f}")
        return best_seed #if best_seed is not None else tuple(map(int, coords[0]))

    def select_seed_ij(
            self,
            start_location=None,
            w_range=1.0,
            w_safe=1.0,
            w_explore=1.0,
            w_exploit=1.0
    ):
        """
        Pick a seed cell (i,j) in grid coordinates.
        Optimized with vectorized operations.
        """
        range_field, safe_field, explore_field, exploit_field = self.compute_score_fields()

        N_samples = 50
        max_resamples = 10
        H, W = self.shape
        rng = np.random.default_rng()

        def get_neighborhood_mask_fast(i, j, size):
            """Returns boolean mask using numpy slicing (vectorized)."""
            nb_mask = np.zeros((H, W), dtype=bool)
            i_min = max(0, i - size)
            i_max = min(H, i + size + 1)
            j_min = max(0, j - size + 1)
            j_max = min(W, j + size + 1)
            nb_mask[i_min:i_max, j_min:j_max] = True
            nb_mask[i, j] = False
            return nb_mask

        candidates = (
                self.Initial_area.astype(bool)
                & ~self.Collision_area.astype(bool)
                & ~self.total_Covered_area.astype(bool)
        )

        if start_location is not None:
            si = max(0, min(H - 1, int(start_location[0])))
            sj = max(0, min(W - 1, int(start_location[1])))

            #r = self.coverage_ratio()
            #size = int(max(H, W) * 3 * (np.exp(10 * r) - 1) / (np.exp(2) - 1) + 1)
            size = int(max(H, W)/2)
            location_neighborhood = get_neighborhood_mask_fast(si, sj, size)
            candidates = candidates & location_neighborhood
            #print(f"size: {size}, ratio: {r}")

        if not candidates.any():
            return (0, 0)

        coords = np.argwhere(candidates)
        n = min(N_samples, len(coords))

        overall_best_seed = None
        overall_best_score = -np.inf

        for attempt in range(max_resamples):
            sampled_idx = rng.choice(len(coords), size=n, replace=False)

            best_seed = None
            best_score = -np.inf

            for k in sampled_idx:
                i, j = map(int, coords[k])

                # Vectorized neighborhood scoring
                i_min, i_max = max(0, i - 5), min(H, i + 6)
                j_min, j_max = max(0, j - 5), min(W, j + 6)

                scores = (
                    w_range * range_field[i_min:i_max, j_min:j_max]
                    + w_safe * safe_field[i_min:i_max, j_min:j_max]
                    + w_explore * explore_field[i_min:i_max, j_min:j_max]
                    + w_exploit * exploit_field[i_min:i_max, j_min:j_max]
                ).ravel()

                if start_location is not None:
                    ii = np.arange(i_min, i_max)[:, np.newaxis]
                    jj = np.arange(j_min, j_max)[np.newaxis, :]
                    dist = np.sqrt((ii - start_location[0]) ** 2 + (jj - start_location[1]) ** 2)
                    w_dist = 0.001
                    scores = scores - w_dist * dist.ravel()

                if len(scores) > 0:
                    s = np.mean(scores)

                    if s > best_score:
                        best_score = s
                        best_seed = (i, j)

            if best_score > overall_best_score:
                overall_best_score = best_score
                overall_best_seed = best_seed

            #print(f"Attempt {attempt + 1}: best_seed={best_seed}, best_score={best_score:.4f}")

            if best_score >= 0:
                i, j = best_seed
                print(
                    f"Selected seed: {best_seed} with score {best_score:.4f} | "
                    f"safety: {safe_field[i, j]:.4f}, "
                    f"range: {range_field[i, j]:.4f}, "
                    f"explore: {explore_field[i, j]:.4f}, "
                    f"exploit: {exploit_field[i, j]:.4f}"
                )
                return best_seed

        print(
            f"No non-negative seed found after {max_resamples} attempts. "
            f"Returning best overall seed: {overall_best_seed} "
            f"with score {overall_best_score:.4f}"
        )
        return overall_best_seed #if overall_best_seed is not None else tuple(map(int, coords[0]))


    def grow_surface_from_seeds(
            self,
            seed_ij,
            S=200,
            neighborhood=8,
            w_range=1.0,
            w_safe=1.0,
            w_explore=1.0,
            w_exploit=1.0,
            min_safe_dist_m=2.5,
            w_compactness=0.8,
    ):
        """
        Grow a connected surface (mask) of up to S cells starting from seed_ij.

        Returns:
          selected_mask: bool[H,W]
          idx_in_range_closest_to_selected: (i,j) or None
        """
        range_field, safe_field, explore_field, exploit_field = self.compute_score_fields()
        H, W = self.shape
        if S <= 0:
            return np.zeros((H, W), dtype=bool), None
        if neighborhood not in (4, 8):
            raise ValueError("neighborhood must be 4 or 8")

        # --- Initial feasibility mask ---
        candidates = (self.Initial_area & ~self.Collision_area) #& ~self.total_Covered_area
        covered= self.total_Covered_area.astype(bool)

        # --- Hard safety buffer from collision ---
        if min_safe_dist_m > 0.0:
            dist_to_collision_m = distance_transform_edt(~self.Collision_area.astype(bool)) * float(self.resolution)
            candidates = candidates & (dist_to_collision_m >= float(min_safe_dist_m))
            if not candidates.any():
                return np.zeros((H, W), dtype=bool), None

            if not candidates[seed_ij]:
                coords = np.argwhere(candidates)
                d2 = (coords[:, 0] - seed_ij[0]) ** 2 + (coords[:, 1] - seed_ij[1]) ** 2
                k = int(np.argmin(d2))
                seed_ij = (int(coords[k, 0]), int(coords[k, 1]))

        # Optional frontier bonus map

        def compactness_bonus(i, j):
            cnt = 0
            total = 0
            for ni, nj in iter_neighbors(i, j):
                total += 1
                if selected[ni, nj]:
                    cnt += 1
            return cnt / total if total > 0 else 0.0

        feasible_mask = candidates

        # --- Neighborhood offsets ---
        neigh4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neigh8 = neigh4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        offsets = neigh8 if neighborhood == 8 else neigh4

        def iter_neighbors(i, j):
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    yield ni, nj

        def has_selected_neighbor(i, j):
            for ni, nj in iter_neighbors(i, j):
                if selected[ni, nj]:
                    return True
            return False

        selected = np.zeros((H, W), dtype=bool)
        selected[seed_ij] = True
        selected_count = 1

        heap = []
        best_score = np.full((H, W), -np.inf, dtype=np.float32)

        def push(i, j):
            if selected[i, j]:
                return
            if not feasible_mask[i, j]:
                return
            s = self.calculate_score(i, j, w_range, w_safe, w_explore, w_exploit,range_field, safe_field, explore_field, exploit_field)
            s += w_compactness*compactness_bonus(i,j)
            if s > best_score[i, j]:
                best_score[i, j] = s
                heapq.heappush(heap, (-s, i, j))

        # Seed frontier: heap contains neighbors of selected cells
        for ni, nj in iter_neighbors(*seed_ij):
            push(ni, nj)

        target = int(S)
        while selected_count < target and heap:
            neg_s, i, j = heapq.heappop(heap)

            if selected[i, j] or not feasible_mask[i, j]:
                continue

            current_s = self.calculate_score(i, j, w_range, w_safe, w_explore, w_exploit,range_field, safe_field, explore_field, exploit_field)
            current_s += w_compactness*compactness_bonus(i,j)

            # Skip if score is significantly worse than best known
            if current_s < best_score[i, j] - 1e-9:
                continue

            # Rule: must have a selected neighbor (for connectivity)
            if not has_selected_neighbor(i, j):
                continue

            selected[i, j] = True
            selected_count += 1
            print(f"selected_count: {selected_count}")

            # Heap contains neighbors of each selected cell
            for ni, nj in iter_neighbors(i, j):
                push(ni, nj)

        '''neighbors = (
                np.roll(selected, 1, 0) + np.roll(selected, -1, 0) +
                np.roll(selected, 1, 1) + np.roll(selected, -1, 1) +
                np.roll(np.roll(selected, 1, 0), 1, 1) +
                np.roll(np.roll(selected, 1, 0), -1, 1) +
                np.roll(np.roll(selected, -1, 0), 1, 1) +
                np.roll(np.roll(selected, -1, 0), -1, 1)
        )

        # avoid adding infeasible cells
        grow_mask = (~selected) & feasible_mask & (neighbors >= 4)
        selected[grow_mask] = True '''

        # Closest point in range to the selected region
        idx = None
        inside = selected & self.Within_range.astype(bool)
        if inside.any():
            idx = tuple(np.argwhere(inside)[0])
        elif self.Within_range.any():
            dist_map = distance_transform_edt(~self.Within_range.astype(bool))
            masked = np.where(selected, dist_map, np.inf)
            idx_min = np.unravel_index(np.argmin(masked), masked.shape)
            if np.isfinite(masked[idx_min]):
                idx = idx_min

        return selected, idx


    def grow_surface_from_seed(
            self,
            seed_ij,
            S=200,
            neighborhood=8,
            w_range=1.0,
            w_safe=1.0,
            w_explore=1.0,
            w_exploit=1.0,
            min_safe_dist_m=2.5,
            w_compactness=0.8,
    ):
        H, W = self.shape
        if S <= 0:
            return np.zeros((H, W), dtype=bool), None
        if neighborhood not in (4, 8):
            raise ValueError("neighborhood must be 4 or 8")

        range_field, safe_field, explore_field, exploit_field = self.compute_score_fields()

        feasible = (self.Initial_area & ~self.Collision_area) & ~self.total_Covered_area

        if min_safe_dist_m > 0:
            dist = distance_transform_edt(~self.Collision_area.astype(bool)) * float(self.resolution)
            feasible &= dist >= float(min_safe_dist_m)

        if not feasible.any():
            return np.zeros((H, W), dtype=bool), None

        if not feasible[seed_ij]:
            coords = np.argwhere(feasible)
            d2 = (coords[:, 0] - seed_ij[0]) ** 2 + (coords[:, 1] - seed_ij[1]) ** 2
            k = int(np.argmin(d2))
            seed_ij = tuple(coords[k])

        neigh4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neigh8 = neigh4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        offsets = neigh8 if neighborhood == 8 else neigh4

        def iter_neighbors(i, j):
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    yield ni, nj

        selected = np.zeros((H, W), dtype=bool)
        selected[seed_ij] = True
        selected_count = 1

        def compactness_bonus(i, j):
            total = 0
            count = 0
            for ni, nj in iter_neighbors(i, j):
                total += 1
                if selected[ni, nj]:
                    count += 1
            return count / total if total else 0.0

        def score(i, j):
            s = self.calculate_score(
                i, j,
                w_range, w_safe, w_explore, w_exploit,
                range_field, safe_field, explore_field, exploit_field
            )
            return s + w_compactness * compactness_bonus(i, j)

        heap = []
        best_score = np.full((H, W), -np.inf, dtype=float)

        def push(i, j):
            if selected[i, j] or not feasible[i, j]:
                return
            s = score(i, j)
            if s > best_score[i, j]:
                best_score[i, j] = s
                heapq.heappush(heap, (-s, i, j))

        for ni, nj in iter_neighbors(*seed_ij):
            push(ni, nj)

        while selected_count < int(S) and heap:
            _, i, j = heapq.heappop(heap)

            if selected[i, j] or not feasible[i, j]:
                continue

            if not any(selected[ni, nj] for ni, nj in iter_neighbors(i, j)):
                continue

            s = score(i, j)
            if s < best_score[i, j] - 1e-9:
                continue

            selected[i, j] = True
            selected_count += 1

            for ni, nj in iter_neighbors(i, j):
                push(ni, nj)

        inside = selected & self.Within_range.astype(bool)
        idx = tuple(np.argwhere(inside)[0]) if inside.any() else tuple(np.argwhere(selected)[0])

        return selected, idx

    # ------------------------------------------------
    # Metrics
    # ------------------------------------------------
    def coverage_ratio(self):
        allowed = self.available_area()
        if not allowed.any():
            return 0.0
        return self.total_Covered_area.sum() / allowed.sum()

    def distance_to_collision(self):
        """
        Distance (meters) from available area to nearest collision cell
        """
        dist = distance_transform_edt(~self.Collision_area) * self.resolution
        return float(dist[self.available_area()].min())

    # ------------------------------------------------
    # Update functions
    # ------------------------------------------------
    def mark_covered(self, mask):
        """
        Mark cells as covered

        mask : boolean grid (same shape)
        """
        self.Covered_area |= mask
        self.total_Covered_area |= mask

    def update_coverage(self, mask):
        """
        Update coverage with new mask receive through communication
        """
        self.total_Covered_area |= mask

    def add_collision_area(self, mask):
        """
        Add forbidden area
        """
        self.Collision_area |= mask

    def add_within_range(self, mask):
        """
        Expand operational range
        """
        self.Within_range |= mask

    # ------------------------------------------------
    # World <-> grid conversion
    # ------------------------------------------------
    def world_to_grid(self, N, E):
        i = int((N - self.origin[0]) / self.resolution)
        j = int((E - self.origin[1]) / self.resolution)
        return i, j

    def grid_to_world(self, i, j):
        N = self.origin[0] + (i+0.5) * self.resolution
        E = self.origin[1] + (j+0.5) * self.resolution
        return N, E

    def grid_to_world_mask(self, mask: np.ndarray):
        ii, jj = np.where(mask)

        if ii.size == 0:
            return np.array([]), np.array([])

        N = self.origin[0] + ii * self.resolution
        E = self.origin[1] + jj * self.resolution

        return N, E

class Heatmap:
    def __init__(self, shape):
        self.shape = shape
        self.data = np.zeros(shape, dtype=np.float32)

    def update(self):
        rows, cols = self.shape
        self.data[:] = 0.0

        # Grid of coordinates
        y, x = np.mgrid[0:rows, 0:cols]

        # Random number of blobs
        n_blobs = np.random.randint(3, 10)

        for _ in range(n_blobs):
            # Random blob center
            cx = np.random.uniform(0, cols)
            cy = np.random.uniform(0, rows)

            # Random blob size
            sigma_x = np.random.uniform(cols * 0.05, cols * 0.15)
            sigma_y = np.random.uniform(rows * 0.05, rows * 0.15)

            # Random blob intensity
            amplitude = np.random.uniform(0.4, 1.0)

            # Gaussian blob
            blob = amplitude * np.exp(
                -(((x - cx) ** 2) / (2 * sigma_x ** 2) +
                  ((y - cy) ** 2) / (2 * sigma_y ** 2))
            )

            self.data += blob

        # Normalize to [0, 1]
        self.data -= self.data.min()
        if self.data.max() > 0:
            self.data /= self.data.max()

        self.data = self.data.astype(np.float32)






def main():
    # Create and update heatmap
    heatmap = Heatmap((600, 500))
    heatmap.update()

    # Create white → orange colormap
    cmap = LinearSegmentedColormap.from_list(
        "white_orange",
        ["white", "orange"]
    )

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap.data, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(label="Intensity")
    plt.title("Random Heatmap")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()

'''
def main():
    # -----------------------------
    # 1) Build a test environment
    # -----------------------------
    env = Environment(shape=(120, 160), resolution=1.0, origin=(0.0, 0.0))
    H, W = env.shape

    env.Initial_area[:, :] = True

    env.Collision_area[20:35, 30:55] = True
    env.Collision_area[55:80, 85:110] = True
    env.Collision_area[35:50, 120:145] = True

    env.Covered_area[5:10, 5:10] = True

    # -----------------------------
    # 2) Animation setup
    # -----------------------------
    T = 10
    S = 250
    neighborhood = 8

    cmap = ListedColormap([
        (0.00, 0.00, 0.00, 1.0),
        (0.10, 0.10, 0.90, 0.35),
        (0.00, 0.80, 0.00, 0.70),
        (0.80, 0.00, 0.00, 0.90),
        (1.00, 0.85, 0.10, 0.85),
    ])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Seed + Within_range + Collision + Covered + Grown surface")
    ax.set_xlabel("j (E / cols)")
    ax.set_ylabel("i (N / rows)")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_aspect("equal")
    ax.grid(False)

    img = np.zeros((H, W), dtype=np.uint8)
    im = ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=4, interpolation="nearest")

    seed_scatter = ax.scatter([], [], s=80, marker="x", c="white", linewidths=2, label="seed")
    idx_scatter  = ax.scatter([], [], s=60, marker="o", facecolors="none", edgecolors="white", linewidths=2, label="idx")

    # NEW: path overlay objects (create once, update each frame)
    path_line, = ax.plot([], [], linewidth=1.5, label="coverage path")
    path_start_scatter = ax.scatter([], [], marker="x", s=80, c="white", linewidths=2)

    ax.legend(loc="upper right")

    start_ij = (8, 8)

    # -----------------------------
    # 3) Helpers
    # -----------------------------
    def update_within_range(center_ij, radius_cells):
        ci, cj = center_ij
        ii, jj = np.indices((H, W))
        env.Within_range = ((ii - ci) ** 2 + (jj - cj) ** 2) <= (radius_cells ** 2)

    def build_viz(surface_mask):
        out = np.zeros((H, W), dtype=np.uint8)
        out[env.Within_range] = 1
        out[env.Covered_area] = 2
        out[env.Collision_area] = 3
        out[surface_mask] = 4
        out[env.Collision_area] = 3
        return out

    # -----------------------------
    # 4) Animation step
    # -----------------------------
    def step(frame):
        nonlocal start_ij

        center_i = int(20 + frame * 3) % H
        center_j = int(20 + frame * 4) % W
        update_within_range((center_i, center_j), radius_cells=25)

        seed = env.select_seed_ij(start_location=start_ij)

        surface_mask, idx = env.grow_surface_from_seed(
            seed_ij=seed,
            S=S,
            neighborhood=neighborhood,
            w_range=0.3,
            w_safe=1.0,
            w_explore=0.00,
            w_exploit=0.990,
            min_safe_dist_m=3.0,
            w_compactness=0.1,
        )
       
        planner = DirectionalBoustrophedonPlanner(occ_map=surface_mask, neighborhood=8)

        best_angle, lanes_rot, path, order = planner.plan(
            start=seed,
            auto_angles_deg=list(range(0, 180, 10)),
            connect=True
        )
        
        planner = BoustrophedonCoveragePlanner(occ_map=surface_mask, neighborhood=8)
        lanes_rot, path, order = planner.plan(
            start=seed,
             sweep_axis="row",
            connect=True
        )

        env.mark_covered(surface_mask)

        if idx is not None:
            start_ij = (int(idx[0]), int(idx[1]))
        else:
            start_ij = (int(seed[0]), int(seed[1]))

        img2 = build_viz(surface_mask)
        im.set_data(img2)

        seed_scatter.set_offsets([[seed[1], seed[0]]])

        if idx is not None:
            idx_scatter.set_offsets([[idx[1], idx[0]]])
        else:
            idx_scatter.set_offsets([])

        # NEW: add/update the path plotting snippet (in an animation-friendly way)
        if path:
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            path_line.set_data(xs, ys)
            path_start_scatter.set_offsets([[xs[0], ys[0]]])
        else:
            path_line.set_data([], [])
            path_start_scatter.set_offsets([])

        ax.set_title(
            f"Iter {frame+1}/{T}  | seed={seed} | start_ij={start_ij} | covered={int(env.Covered_area.sum())}"
        )

        return im, seed_scatter, idx_scatter, path_line, path_start_scatter

    anim = FuncAnimation(fig, step, frames=T, interval=450, blit=False, repeat=False)
    plt.show()


if __name__ == "__main__":

    main()

'''