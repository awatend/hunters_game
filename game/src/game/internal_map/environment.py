from __future__ import annotations
from skimage import measure
import numpy as np
import heapq
from scipy.ndimage import distance_transform_edt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import random
try:
    from .cell_geometry import CellGeometry
except ImportError:  # Direct execution from game/internal_map
    from cell_geometry import CellGeometry


class Environment:
    def __init__(self, geometry: "CellGeometry"):
        """
        Parameters
        ----------
        geometry : CellGeometry
            Cell geometry object defining the spatial layout.
            Swap this for a different CellGeometry subclass to change cell shape.
        """
        self.geom       = geometry
        # ── backward-compat shims (do not remove – used widely) ────────────
        self.shape      = geometry.shape
        self.resolution = geometry.cell_spacing_m
        self.origin     = geometry.origin
        # Keep self.data as an alias for self.prior for backward compatibility
        # ───────────────────────────────────────────────────────────────────
        self.target = None   # treasure location

        shape = self.shape
        self.prior = np.full(shape, 0.01, dtype=np.float32)
        # --- Core regions ---
        self.Initial_area          = np.zeros(shape, dtype=bool)
        self.Collision_area        = np.zeros(shape, dtype=bool)
        self.static_Collision_area = np.zeros(shape, dtype=bool)
        self.dynamic_Collision_area = np.zeros(shape, dtype=bool)
        self.Covered_area          = np.zeros(shape, dtype=bool)
        self.total_Covered_area    = np.zeros(shape, dtype=bool)
        self.overlap_ratio         = 0.0

    @property
    def data(self):
        """Backward-compat alias for self.prior."""
        return self.prior

    @data.setter
    def data(self, value):
        self.prior = value


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
        self.Collision_area = self.static_Collision_area | self.dynamic_Collision_area



    def update(self, prior_data):
        """
        Set the prior belief of where the target is.

        Parameters
        ----------
        prior_data : np.ndarray, optional
            External prior field with the same shape as the grid, e.g. the
            ``biomass`` or ``temperature`` array from
            ``load_prior.SinmodPrior``.  NaN values (outside the data hull)
            are set to zero.  The array is normalised to [0, 1] and stored
            in ``self.prior``.

            If *None* (default), a synthetic prior of random Gaussian blobs
            is generated.

        Examples
        --------
        # Synthetic random prior
        env.update()

        # SINMOD biomass prior
        from load_prior import SinmodPrior
        sinmod = SinmodPrior()
        env.update(prior_data=sinmod.biomass)
        """
        if prior_data is not None:
            # ── SINMOD / external prior ───────────────────────────────────
            p = np.array(prior_data, dtype=np.float32)
            if p.shape != self.shape:
                raise ValueError(
                    f"prior_data shape {p.shape} does not match "
                    f"environment shape {self.shape}."
                )
            np.nan_to_num(p, nan=0.0, copy=False)   # NaN → 0 (outside hull)
            p -= p.min()
            if p.max() > 0:
                p /= p.max()
            self.prior = p
        else:
            # ── Random Gaussian-blob prior ────────────────────────────────
            rows, cols = self.shape
            self.prior[:] = 0.0
            y, x = np.mgrid[0:rows, 0:cols]
            n_blobs = np.random.randint(3, 10)
            for _ in range(n_blobs):
                cx        = np.random.uniform(0, cols)
                cy        = np.random.uniform(0, rows)
                sigma_x   = np.random.uniform(cols * 0.05, cols * 0.15)
                sigma_y   = np.random.uniform(rows * 0.05, rows * 0.15)
                amplitude = np.random.uniform(0.4, 1.0)
                blob = amplitude * np.exp(
                    -(((x - cx) ** 2) / (2 * sigma_x ** 2) +
                      ((y - cy) ** 2) / (2 * sigma_y ** 2))
                )
                self.prior += blob
            self.prior -= self.prior.min()
            if self.prior.max() > 0:
                self.prior /= self.prior.max()
            self.prior = self.prior.astype(np.float32)

        return

    def compute_score_fields(self):
        covered = self.total_Covered_area.astype(bool)
        collision = self.Collision_area.astype(bool)
        #sigma_explore_m = 50.0
        sigma_exploit_m = 2.0
        sigma_safe_m = 80.0

        dist_to_covered_m = distance_transform_edt(~covered) * self.geom.cell_spacing_m
        explore_field =self.prior  #np.exp(-(dist_to_covered_m / sigma_exploit_m) ** 2).astype(np.float32) * (1 - self.prior)
        exploit_field = np.exp(-(dist_to_covered_m / sigma_exploit_m) ** 2).astype(np.float32)




        if collision.any():
            dist_to_collision_m = distance_transform_edt(~collision) * self.geom.cell_spacing_m
            safe_field = 1- np.exp(-(dist_to_collision_m / sigma_safe_m) ** 2).astype(np.float32)
            safe_field[collision] = 0.0
            explore_field[collision] = 0.0
            exploit_field[collision] = 0.0
        else:
            safe_field = np.ones(self.shape, dtype=np.float32)  # All cells are safe if no collision

        if covered.any():
            exploit_field[covered] = 0.0
            explore_field[covered] = 0.0


        # Normalize all fields to [0, 1]
        def normalize_field(field):
            min_val = field.min()
            max_val = field.max()
            if max_val - min_val < 1e-9:
                return np.ones_like(field, dtype=np.float32) * 0.5
            return ((field - min_val) / (max_val - min_val)).astype(np.float32)

        safe_field = normalize_field(safe_field)
        explore_field = normalize_field(explore_field)
        exploit_field = normalize_field(exploit_field)

        return  safe_field, explore_field, exploit_field

    def calculate_score(self, i, j, w_safety, w_explore, w_exploit,
                        ):
        """
        Calculate score for a cell (i, j).
        All field components should be normalized to [0, 1].
        """
        safe_field,explore_field,exploit_field=self.compute_score_fields()
        return (w_explore * explore_field[i, j] + w_exploit * exploit_field[i, j])

    def get_neighborhood_mask_fast(self, i, j, size=3):
        """
        Returns a boolean mask of cells within 'size' topology steps of (i, j).
        Delegates to CellGeometry.neighborhood_mask so the shape is respected.
        """
        return self.geom.neighborhood_mask(i, j, size)


    def select_seed_ijs(self, start_location=None, size=4, N_samples=10,):

        H, W = self.shape

        if start_location is None:
            print("Provide a start location")
            return None

        #  useful maps
        covered_or_unsafe = (self.Collision_area.astype(bool)| self.total_Covered_area.astype(bool))
        # Distance from obstacles / covered regions
        dist_field = distance_transform_edt(~covered_or_unsafe)


        # Clamp start point

        si = max(0, min(H - 1, int(start_location[0])))
        sj = max(0, min(W - 1, int(start_location[1])))

        while True:
            neighborhood_mask = self.get_neighborhood_mask_fast(si, sj,size)
            coords = np.argwhere(neighborhood_mask & ~ covered_or_unsafe)
            if coords.size > 0 or size> H/2:
                break
            size += 1
        if coords.size == 0:
            print("No seed possible")
            return None
        # Randomly sample candidates
        rng = np.random.default_rng()
        n_select = min(N_samples, len(coords))
        sampled_idx = rng.choice(len(coords), size=n_select, replace=False)
        sampled_coords = coords[sampled_idx]

        # Evaluate candidates
        best_score = -np.inf
        best_seed = None

        for i, j in sampled_coords:
            local_mask = self.get_neighborhood_mask_fast(i, j)
            covered_count = np.count_nonzero(local_mask & self.total_Covered_area )
            collision_count = np.count_nonzero(local_mask & self.Collision_area)
            total_count = np.count_nonzero(local_mask)

            if total_count == 0:
                continue
            # Lower covered/collision density is better
            free_ratio = 1.0 - ( covered_count + 2*collision_count) / total_count

            # Exploration bonus
            distance_score = dist_field[i,j]

            score = (2.0 * free_ratio +1.0 * distance_score)

            if score > best_score:
                best_score = score
                best_seed = (i, j)
        # Fallback
        if best_seed is None:
            feasible_coords = np.argwhere(~ covered_or_unsafe)
            if feasible_coords.size == 0:
                return None
            distances = dist_field[feasible_coords[:,0],feasible_coords[:,1]]
            i, j = feasible_coords[np.argmax(distances)]
            best_seed = (int(i), int(j))
        return best_seed


    def select_seed_ij(self, start_location=None, size=2, N_samples=10, min_safe_dist_m=2.5,w_safety=1.0, w_explore=1.0, w_exploit=1.0):

        H, W = self.shape

        if start_location is None:
            print("Provide a start location")
            return None


        # Clamp start point

        si = max(0, min(H - 1, int(start_location[0])))
        sj = max(0, min(W - 1, int(start_location[1])))

        safe_mask=distance_transform_edt(~self.Collision_area) * self.geom.cell_spacing_m >= float(min_safe_dist_m)
        while True:
            neighborhood_mask = self.get_neighborhood_mask_fast(si, sj, size)
            safe_neighborhood=neighborhood_mask & safe_mask
            uncovered_save_neighborhood=safe_neighborhood & ~self.total_Covered_area

            coords = np.argwhere (uncovered_save_neighborhood)
            if len(coords) > 0 or size > max(H, W):
                break
            size += 1
        if coords.size == 0:
            print("No seed possible")
            return None
        # Randomly sample candidates
        rng = np.random.default_rng()
        n_select = min(N_samples, len(coords))
        sampled_idx = rng.choice(len(coords), size=n_select, replace=False)
        sampled_coords = coords[sampled_idx]

        # Evaluate candidates
        best_score = -np.inf
        best_seed = None

        for i, j in sampled_coords:
            initial_mask = self.get_neighborhood_mask_fast(i, j)
            local_mask = initial_mask & safe_mask & ~self.total_Covered_area

            total_count = np.count_nonzero(initial_mask)
            #for each element in the local mask, of coordinate mi,mj, calculate the score based on safety, exploration, and exploitation
            score=0
            for mi, mj in np.argwhere(local_mask):
                score+= self.calculate_score(mi, mj, w_safety, w_explore, w_exploit,)
            score=score/total_count
            if score > best_score:
                best_score = score
                best_seed = (i, j)
        return best_seed
    #Grow a surface from a seed point.

    def grow_surface_from_seed(self, seed_ij, S=10, neighborhood=8,
            w_safe=1.0, w_explore=1.0, w_exploit=1.0,
            min_safe_dist_m=2.5, w_compactness=0.8):
        start=seed_ij

        seed_ij = self.select_seed_ij(start, size=3, N_samples=5, min_safe_dist_m=min_safe_dist_m,
                                       w_safety=w_safe, w_explore=w_explore, w_exploit=w_exploit)

        #distance from start to seed
        distance= np.sqrt((seed_ij[0]-start[0])**2+(seed_ij[1]-start[1])**2)
        print(f"distance from start to seed: {distance}")

        H, W = self.shape
        if S <= 0:
            return np.zeros((H, W), dtype=bool), None

        # Pre-calculate the static part of the score
        s_f, ex_f, et_f = self.compute_score_fields()
        base_scores = w_safe * s_f *(
                       w_explore * ex_f + w_exploit * et_f)


        # Calculate feasibility map
        feasible = (self.Initial_area & ~self.Collision_area)
        if min_safe_dist_m > 0:
            dist = distance_transform_edt(~self.Collision_area) * self.geom.cell_spacing_m
            feasible &= (dist >= float(min_safe_dist_m))
        good_candidate=(feasible & ~self.total_Covered_area)
        if not good_candidate.any():
            return np.zeros((H, W), dtype=bool), None

        # 1. Ensure seed is good_candidate before starting expansion
        if not good_candidate[seed_ij]:
            print("Seed already covered")
            coords = np.argwhere(feasible)
            d2 = (coords[:, 0] - seed_ij[0]) ** 2 + (coords[:, 1] - seed_ij[1]) ** 2
            seed_ij = tuple(coords[np.argmin(d2)])
            print(f"new seed {seed_ij}")

        # Neighbors are defined by the geometry — no hard-coded offsets needed.
        # The `neighborhood` parameter is kept for API compatibility but is
        # superseded by the connectivity encoded in self.geom.
        _geom = self.geom

        selected = np.zeros((H, W), dtype=bool)
        selected[seed_ij] = True
        selected_count = 1
        print(f"SEED {seed_ij}")

        heap = []
        # Tracks the best score currently in the heap for a cell to avoid massive redundancy
        best_heap_score = np.full((H, W), -np.inf, dtype=float)

        def get_current_score(i, j):
            """Calculates score including dynamic compactness bonus."""
            neighbor_count = 0
            total_possible = 0
            for ni, nj in _geom.neighbors(i, j):
                total_possible += 1
                if selected[ni, nj]:
                    neighbor_count += 1
            compactness = neighbor_count / total_possible if total_possible > 0 else 0
            return base_scores[i, j] + w_compactness * compactness

        def try_push(i, j):
            if selected[i, j]:
                return
            if not feasible[i, j]:
                return
            if not good_candidate[i, j] and len(heap) >4:
                return
            s = get_current_score(i, j)
            heapq.heappush(heap, (-s, i, j))

        # Initial neighbors of the (potentially snapped) seed
        for ni, nj in _geom.neighbors(seed_ij[0], seed_ij[1]):
            try_push(ni, nj)

        quality= False
        iter=0
        while selected_count < int(S) and quality is False: # Solve case where selected_count<<S
            if heap:
                neg_s, i, j = heapq.heappop(heap)
            else:
                print(f"heap empty   {selected_count}")
                break


            if selected[i, j]:
                continue

            # 2. FIX: Lazy Update
            # Re-check the score because a neighbor might have been added since this was pushed
            current_s = get_current_score(i, j)
            if  current_s > neg_s+ 1e-7:
                # If current score is better than what we popped, re-push and try again
                neg_s, i, j =heapq.heappushpop(heap, (-current_s, i, j))

            selected[i, j] = True
            #print(f"new point addedd {i} {j}")
            selected_count += 1

            # Push new neighbors into the heap
            for ni, nj in _geom.neighbors(i, j):
                try_push(ni, nj)

            if selected_count< int(S):
                continue
            else:
                selection_quality= np.count_nonzero(selected & ~self.total_Covered_area)

                if selection_quality > int (0.8*S):
                    print(f"good quality")
                    quality = True
                else:
                    print(f"poor coverage quality{selection_quality}")
                    selected=selected*False
                    selected_count=0
                    iter+=1
                    start=[seed_ij[0]+random.randint(-3*iter,3*iter),seed_ij[1]+random.randint(-3*iter,3*iter)]
                    new_seed=self.select_seed_ij(start,size=3+iter, N_samples=5+2*iter, min_safe_dist_m=min_safe_dist_m,w_safety=w_safe, w_explore=w_explore, w_exploit=w_exploit)
                    try_push(new_seed[0], new_seed[1])
                    print("high coverage region----recomputing...")

        return selected,



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
        overlap_area=mask & self.total_Covered_area
        if self.total_Covered_area.sum() > 0:
            self.overlap_ratio= float (overlap_area.sum() / self.total_Covered_area.sum())
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


    # ------------------------------------------------
    # World <-> grid conversion
    # ------------------------------------------------
    # ── World ↔ cell coordinate conversion ─────────────────────────────────
    # These are thin wrappers around self.geom so callers never need to know
    # which geometry is active.

    def world_to_grid(self, N, E):
        """Convert world (N, E) to nearest grid (row, col). Geometry-agnostic."""
        return self.geom.world_to_cell(float(N), float(E))

    def grid_to_world(self, i, j):
        """Convert grid (row, col) to world (N, E) cell centre. Geometry-agnostic."""
        return self.geom.cell_to_world(int(i), int(j))

    def grid_to_world_mask(self, mask: np.ndarray):
        """Batch-convert boolean mask to (N_array, E_array) of cell centres."""
        return self.geom.cells_to_world(mask)

    def save_mask_polygon(
            self,
            mask,
            filename="polygon.png",
            show_points=True,
            contour_level=0.5,
    ):

        #extract contours
        mask_float = mask.astype(float)
        contours = measure.find_contours(
            mask_float,
            level=contour_level
        )

        if len(contours) == 0:
            print("No contours found")
            return []


        fig, ax = plt.subplots(figsize=(8, 8))

        polygons_world = []


    # Optional raw mask points
    # ----------------------------------------------------------
        if show_points:
            pts = np.argwhere(mask)

            if pts.shape[0] < 3:
                print("Not enough points to create polygon")
                return None

            # Convert image indices -> world coordinates
            x, y = self.grid_to_world_mask(mask)
            ax.scatter(
                x,
                y,
                s=3,
                alpha=0.3,
                label="Mask points"
            )

        # Contour
        for k, contour in enumerate(contours):
            x_world = (
                    self.origin[0]
                    + contour[:, 0] * self.resolution
            )

            y_world = (
                    self.origin[1]
                    + contour[:, 1] * self.resolution
            )

            polygon_world = np.column_stack(
                (x_world, y_world)
            )

            polygons_world.append(polygon_world)

        # Optional raw points
        polygon_closed = np.vstack([
            polygon_world,
            polygon_world[0]
        ])

        ax.plot(
            polygon_closed[:, 0],
            polygon_closed[:, 1],
            linewidth=2,
            label=f"Contour {k}"
        )

        # Optional target
        if self.target is not None:
            ax.scatter(self.target[0],self.target[1],marker="x", s=100,label="Target")

        ax.set_aspect("equal")
        ax.set_title("Polygon from Mask")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.legend()

        plt.tight_layout()

        # Save
        plt.savefig(filename, dpi=300)
        plt.close(fig)
        print(f"Polygon saved to: {filename}")

class Heatmap:
    def __init__(self, shape):
        self.shape = shape
        self.prior = np.zeros(shape, dtype=np.float32)

    def update(self):
        rows, cols = self.shape
        self.prior[:] = 0.0

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

            self.prior += blob

        # Normalize to [0, 1]
        self.prior -= self.prior.min()
        if self.prior.max() > 0:
            self.prior /= self.prior.max()

        self.prior = self.prior.astype(np.float32)




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
