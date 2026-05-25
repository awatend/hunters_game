from __future__ import annotations
from skimage import measure
import numpy as np
import heapq
from scipy.ndimage import distance_transform_edt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import random
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
        self.data = np.full(shape, 0.01, dtype=np.float32) #Heatmap of prior knowldge f where the treasure is

        # --- Core regions ---
        self.Initial_area   = np.zeros(shape, dtype=bool)
        self.Within_range   = np.zeros(shape, dtype=bool)
        self.Collision_area = np.zeros(shape, dtype=bool)
        self.static_Collision_area = np.zeros(shape, dtype=bool)
        self.dynamic_Collision_area = np.zeros(shape, dtype=bool)
        self.Covered_area   = np.zeros(shape, dtype=bool)
        self.total_Covered_area = np.zeros(shape, dtype=bool)
        self.overlap_ratio= 0.0


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
        permeable_range=within &~collision
        sigma_explore_m = 50.0
        sigma_exploit_m = 12
        sigma_range_m = 200 #large sigma slow decay
        sigma_safe_m = 80.0

        #dist_to_covered_m = distance_transform_edt(~covered) * float(self.resolution)
        explore_field =1 - self.data  #np.exp(-(dist_to_covered_m / sigma_exploit_m) ** 2).astype(np.float32) * (1 - self.data)
        exploit_field = self.data


        if within.any():
            dist_to_range_m = distance_transform_edt(~within) * float(self.resolution)
            range_field = np.exp(-(dist_to_range_m / sigma_range_m) ** 2).astype(np.float32)
            range_field[permeable_range] = 1.0
        else:
            range_field = np.zeros(self.shape, dtype=np.float32)

        if collision.any():
            dist_to_collision_m = distance_transform_edt(~collision) * float(self.resolution)
            safe_field = 1- np.exp(-(dist_to_collision_m / sigma_safe_m) ** 2).astype(np.float32)
            safe_field[collision] = 0.0
            explore_field[collision] = 0.0
            exploit_field[collision] = 0.0
            range_field[collision] = 0.0
        else:
            safe_field = np.ones(self.shape, dtype=np.float32)  # All cells are safe if no collision

        if covered.any():
            exploit_field[covered] = 0.0
            explore_field[covered] = 0.0
            range_field[covered] = 0.0


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
             w_safety * safe_field[i, j]
            * (w_explore * explore_field[i, j]
               +w_range * range_field[i, j]
            + w_exploit * exploit_field[i, j])
        )

    def get_neighborhood_mask_fast(self,i, j, size):
        """Returns boolean mask using numpy slicing (vectorized)."""
        H,W=self.shape
        nb_mask = np.zeros((H, W), dtype=bool)
        i_min = max(0, i - size)
        i_max = min(H, i + size + 1)
        j_min = max(0, j - size + 1)
        j_max = min(W, j + size + 1)
        nb_mask[i_min:i_max, j_min:j_max] = True
        nb_mask[i, j] = False
        return nb_mask
    '''
    def select_seed_ijss(
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

        N_samples = 100
        max_resamples = 10
        H, W = self.shape
        rng = np.random.default_rng()


        candidates = (
                self.Initial_area.astype(bool)
                & ~self.Collision_area.astype(bool)
                & ~self.total_Covered_area.astype(bool)
        )

        if start_location is None:
            print("Provide a start location")

        else:
            si = max(0, min(H - 1, int(start_location[0])))
            sj = max(0, min(W - 1, int(start_location[1])))

            #r = self.coverage_ratio()
            #size = int(max(H, W) * 3 * (np.exp(10 * r) - 1) / (np.exp(2) - 1) + 1)
            size = 5# Neighborhood where to sample
            location_neighborhood = self.get_neighborhood_mask_fast(si, sj, size)
            candidates = candidates & location_neighborhood
            #print(f"size: {size}, ratio: {r}")

        if not candidates.any():
            return (0, 0)

        coords = np.argwhere(candidates)
        n = min(N_samples, np.count_nonzero(candidates))

        overall_best_seed = None
        overall_best_score = -np.inf

        for attempt in range(max_resamples):
            sampled_idx = rng.choice(len(coords), size=n, replace=False)

            best_seed = None
            best_score = -np.inf

            for k in sampled_idx:
                i, j = map(int, coords[k])

                # Vectorized neighborhood scoring
                i_min, i_max = max(0, i - 2), min(H, i + 2)
                j_min, j_max = max(0, j - 1), min(W, j + 2)

                scores = (
                    w_range * range_field[i_min:i_max, j_min:j_max]
                    + w_safe * safe_field[i_min:i_max, j_min:j_max]
                    + w_explore * explore_field[i_min:i_max, j_min:j_max]
                    + w_exploit * exploit_field[i_min:i_max, j_min:j_max]
                ).ravel()

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

                )
                return best_seed

        print(
            f"No non-negative seed found after {max_resamples} attempts. "
            f"Returning best overall seed: {overall_best_seed} "
            f"with score {overall_best_score:.4f}"
        )
        return overall_best_seed #if overall_best_seed is not None else tuple(map(int, coords[0]))
    '''

    '''
    def select_seed_ijs(
            self,
            start_location=None, size=5
    ):


        def get_best_seed(start_location):
            si = max(0, min(H - 1, int(start_location[0])))
            sj = max(0, min(W - 1, int(start_location[1])))

             # Neighborhood where to sample
            good_score = 3/size
            location_neighborhood = self.get_neighborhood_mask_fast(si, sj, size)
            #sampled_idx = rng.choice(len(location_neighborhood), size=N_samples, replace=False)

            values=1/(1+np.count_nonzero(location_neighborhood & self.Covered_area )+
                      np.count_nonzero(location_neighborhood)* np.count_nonzero(location_neighborhood & self.Collision_area))

            if values>good_score:
                return (si, sj)
            else:
                mask = self.get_neighborhood_mask_fast(si, sj, size)
                coords = np.argwhere(mask)

                if coords.size > 0:

                    non_feasible =self.Collision_area.astype(bool) |self.total_Covered_area.astype(bool)
                    dist_field = distance_transform_edt(~non_feasible)
                    distances = dist_field[coords[:, 0], coords[:, 1]]
                    i,j = coords[np.argmax(distances)]
                    furthest_pt=(int(i), int(j))
                else:
                    furthest_pt = None  # Handle empty mask cas
            return furthest_pt

        N_samples = 100
        max_resamples = 10
        H, W = self.shape
        if start_location is None:
            print("Provide a start location")
            return
        else:
            seed=get_best_seed(start_location)
            counter=1
            while seed is None:
                start_location[0]= random.randint(-counter, counter)
                start_location[1]= random.randint(-counter, counter)
                seed =get_best_seed(start_location)
                counter+=1

            return seed
    '''

    def select_seed_ij(
            self,
            start_location=None,
            size=5,
            N_samples=10,
    ):

        H, W = self.shape

        if start_location is None:
            print("Provide a start location")
            return None


        #  useful maps

        non_feasible = (
                self.Collision_area.astype(bool)
                | self.total_Covered_area.astype(bool)
        )

        # Distance from obstacles / covered regions
        dist_field = distance_transform_edt(~non_feasible)


        # Clamp start point

        si = max(0, min(H - 1, int(start_location[0])))
        sj = max(0, min(W - 1, int(start_location[1])))

        while True:
            neighborhood_mask = self.get_neighborhood_mask_fast(si, sj, size)
            coords = np.argwhere(neighborhood_mask & ~non_feasible)
            if coords.size > 0 or size> H/2:
                break
            size += 1
        if coords.size == 0:
            print("No seed possible")
            return None
        # Randomly sample candidates
        rng = np.random.default_rng()
        n_select = min(N_samples, len(coords))
        sampled_idx = rng.choice(
            len(coords),
            size=n_select,
            replace=False
        )
        sampled_coords = coords[sampled_idx]

        # Evaluate candidates
        best_score = -np.inf
        best_seed = None

        for i, j in sampled_coords:
            local_mask = self.get_neighborhood_mask_fast(i, j, size)
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
            feasible_coords = np.argwhere(~non_feasible)
            if feasible_coords.size == 0:
                return None
            distances = dist_field[feasible_coords[:,0],feasible_coords[:,1]]
            i, j = feasible_coords[np.argmax(distances)]
            best_seed = (int(i), int(j))
        return best_seed

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
            function=None
    ):
        H, W = self.shape
        if S <= 0:
            return np.zeros((H, W), dtype=bool), None

        # Pre-calculate the static part of the score
        r_f, s_f, ex_f, et_f = self.compute_score_fields()
        base_scores = w_safe * s_f *(w_range * r_f +
                       w_explore * ex_f + w_exploit * et_f)

        if function is "support":
            w_compactness=0.01
        # Calculate feasibility map
        feasible = (self.Initial_area & ~self.Collision_area)
        if min_safe_dist_m > 0:
            dist = distance_transform_edt(~self.Collision_area) * float(self.resolution)
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

        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if neighborhood == 8:
            offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

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
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
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
        for di, dj in offsets:
            ni, nj = seed_ij[0] + di, seed_ij[1] + dj
            if 0 <= ni < H and 0 <= nj < W:
                try_push(ni, nj)
                #print(f"pushed {seed_ij} {ni} {nj}")

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
            if -neg_s < current_s - 1e-7:
                # If current score is better than what we popped, re-push and try again
                neg_s, i, j =heapq.heappushpop(heap, (-current_s, i, j))

            selected[i, j] = True
            #print(f"new point addedd {i} {j}")
            selected_count += 1

            # Push new neighbors into the heap
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                        try_push(ni, nj)
                        #print("new neighbors pushed")

            if selected_count< S-1:
                continue
            else:
                selection_quality= np.argwhere(selected & ~self.total_Covered_area)
                if selection_quality.size > 0.80*S:
                    print(f"good qualuty")
                    quality = True
                else:
                    selected=selected*False
                    selected_count=0
                    iter+=1
                    start=[seed_ij[0]+random.randint(-3*iter,3*iter),seed_ij[1]+random.randint(-3*iter,3*iter)]
                    new_seed=self.select_seed_ij(start,size=3+iter, N_samples=5+2*iter)
                    try_push(new_seed[0], new_seed[1])
                    print("high coverage region----recomputing...")

        idx_list = np.argwhere(selected) if selected.any() else seed_ij
        if function is "hunter":
            dist_field = distance_transform_edt(~self.total_Covered_area.astype(bool))
            selected_distances = dist_field[idx_list[:, 0], idx_list[:, 1]]
            farthest_idx = np.argmax(selected_distances)
            pt_to_return = tuple(idx_list[farthest_idx])

        elif function is "support":
            if idx_list.size > 0:
                # Scores of selected points
                selected_scores = base_scores[idx_list[:, 0],idx_list[:, 1]]
                # Index of best point
                best_idx = np.argmax(selected_scores)
                print(f"best score in selected: {selected_scores[best_idx]:.4f}")
                # Coordinate with highest score
                i, j = idx_list[best_idx]

                pt_to_return = (int(i), int(j))
            else:
                pt_to_return = seed_ij

        return selected, pt_to_return



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

    def add_within_range(self, mask):
        """
        Expand operational range
        """
        self.Within_range |= mask

    # ------------------------------------------------
    # World <-> grid conversion
    # ------------------------------------------------
    def world_to_grid(self, N, E):
        H,W = self.shape
        i = int((N-0.5 - self.origin[0]) / self.resolution)
        j = int((E-0.5 - self.origin[1]) / self.resolution)
        i = np.clip(i, 0, H - 1).astype(int)
        j = np.clip(j, 0, W - 1).astype(int)
        return i, j

    def grid_to_world(self, i, j):
        H, W = self.shape
        i = np.clip(i, 0, H - 1).astype(int)
        j = np.clip(j, 0, W - 1).astype(int)
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