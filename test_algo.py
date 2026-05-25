"""
Test class for hunters game coverage planning simulation.
Contains all helper functions and methods for running the simulation.
"""

from environment import Environment
from vehicles import AUV, ASV

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
from skimage import measure
from skimage.draw import polygon
import fields2cover as f2c
import math


class HuntersGameSimulation:
    """Main simulation class for multi-vehicle coverage planning."""

    def __init__(self, shape, resolution, origin):
        """
        Initialize the simulation environment.

        Args:
            shape: Grid shape (height, width)
            resolution: Cell resolution in meters
            origin: Grid origin (N, E) in world coordinates
        """
        self.shape = shape
        self.resolution = resolution
        self.origin = origin

        # Environments for each vehicle
        self.env_marie = None
        self.env_thor = None
        self.env_grethe = None

        # Vehicles
        self.marie = None
        self.thor = None
        self.grethe = None

        # Simulation state
        self.treasure_location = None
        self.frames_hist = []
        self.coverage_hist = []
        self.marie_hist = []
        self.thor_hist = []
        self.overlap_r = []
        self.marie_overlap_r = []
        self.thor_overlap_r = []
        self.grethe_marie_dist_hist = []
        self.grethe_thor_dist_hist = []
        self.marie_cov_N = np.array([])
        self.marie_cov_E = np.array([])
        self.thor_cov_N = np.array([])
        self.thor_cov_E = np.array([])

        # Planning parameters (will be set during run)
        # Marie parameters
        self.marie_w_range = 0.1
        self.marie_w_safe = 0.5
        self.marie_w_explore = 1.0
        self.marie_w_exploit = 0.5
        self.marie_min_safe_dist = 6.0
        self.marie_seed_w_range = 0.5
        self.marie_seed_w_safe = 0.0
        self.marie_seed_w_explore = 0.9
        self.marie_seed_w_exploit = 0.5

        # Thor parameters
        self.thor_w_range = 0.2
        self.thor_w_safe = 1.0
        self.thor_w_explore = 1.0
        self.thor_w_exploit = 0.5
        self.thor_min_safe_dist = 6.0
        self.thor_seed_w_range = 0.5
        self.thor_seed_w_safe = 0.9
        self.thor_seed_w_explore = 0.9
        self.thor_seed_w_exploit = 0.5

        # Grethe parameters
        self.grethe_w_range = 1.0
        self.grethe_w_safe = 1.0
        self.grethe_w_explore = 0.01
        self.grethe_w_exploit = 0.09
        self.grethe_min_safe_dist = 6.0
        self.grethe_seed_w_range = 1.0
        self.grethe_seed_w_safe = 0.0
        self.grethe_seed_w_explore = 0.2
        self.grethe_seed_w_exploit = 0.01

    def setup_environments(self):
        """Create and configure environment instances."""
        self.env_marie = Environment(shape=self.shape, resolution=self.resolution, origin=self.origin)
        self.env_thor = Environment(shape=self.shape, resolution=self.resolution, origin=self.origin)
        self.env_grethe = Environment(shape=self.shape, resolution=self.resolution, origin=self.origin)

        self.env_grethe.update()
        for env in (self.env_marie, self.env_thor, self.env_grethe):
            env.Initial_area[:, :] = True

    def add_static_obstacles(self):
        """Add static obstacles to all environment maps."""
        for env in (self.env_marie, self.env_thor, self.env_grethe):
            env.static_Collision_area[40:60, 60:110] = True
            env.static_Collision_area[120:160, 160:190] = True
            env.static_Collision_area[70:95, 220:260] = True
            env.data = self.env_grethe.data

    def setup_vehicles(self, marie_pos, thor_pos, grethe_pos,
                       marie_safety_bubble, thor_safety_bubble, grethe_safety_bubble,
                       marie_comm_range, thor_comm_range, grethe_comm_range):
        """
        Create and configure vehicle instances.

        """
        self.marie = AUV("Marie", pos=marie_pos,
                         safety_bubble_radius=marie_safety_bubble,
                         comm_range_m=marie_comm_range)
        self.thor = AUV("Thor", pos=thor_pos,
                        safety_bubble_radius=thor_safety_bubble,
                        comm_range_m=thor_comm_range)
        self.grethe = ASV("Grethe", pos=grethe_pos,
                          safety_bubble_radius=grethe_safety_bubble,
                          comm_range_m=grethe_comm_range)

        self.marie.set_internal_map(self.env_marie)
        self.thor.set_internal_map(self.env_thor)
        self.grethe.set_internal_map(self.env_grethe)

    def set_treasure_location(self):
        """Randomly set treasure location in feasible area."""
        coords = np.argwhere(self.env_marie.Initial_area & ~self.env_marie.Collision_area)
        self.treasure_location = tuple(coords[np.random.randint(len(coords))])
        print(f"Treasure is at grid location: {self.treasure_location}")

    def mark_initial_covered(self):
        """Mark initial covered areas for AUVs."""
        for auv in (self.marie, self.thor):
            i, j = auv.internal_map.world_to_grid(auv.pos[0], auv.pos[1])
            auv.internal_map.Covered_area[i, j] = True

    def setup_plot(self):
        print("SETuo plots")
        """Setup matplotlib figure and subplots."""
        fig, ((ax, ax_metrics), (ax_dist, ax_overlap)) = plt.subplots(
            2, 2, figsize=(20, 8), gridspec_kw={"width_ratios": [1, 1], "height_ratios": [1, 1]}
        )

        extent = self.grid_extent()

        # Heatmap colormap: white → orange
        heat_cmap = LinearSegmentedColormap.from_list(
            "white_orange",
            ["white", "orange"],
        )

        # --- Main map subplot ---
        ax.set_title("Marie + Thor coverage, Grethe support (bubbles + comm range)")
        ax.set_xlabel("E [m]")
        ax.set_ylabel("N [m]")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(True)

        # --- Metrics subplot ---
        ax_metrics.set_title("Coverage Metrics")
        ax_metrics.set_xlabel("Iteration")
        ax_metrics.set_ylabel("Ratio")
        ax_metrics.set_ylim(0, 1.2)
        ax_metrics.set_xlim(0, 100)  # Default T=100
        ax_metrics.grid(True)

        # --- Distance subplot ---
        ax_dist.set_title("Distances to Grethe")
        ax_dist.set_xlabel("Iteration")
        ax_dist.set_ylabel("Distance [m]")
        ax_dist.set_xlim(0, 100)  # Default T=100
        ax_dist.axhline(self.grethe.comm_range_m, linestyle="--", linewidth=1.5, color="gray", label="Comm range")
        ax_dist.legend(loc="upper left")
        ax_dist.grid(True)

        ax_overlap.set_title("Overlap")
        ax_overlap.set_xlabel("Iteration")
        ax_overlap.set_ylabel("Overlap Ratio")
        ax_overlap.set_xlim(0, 100)
        ax_overlap.set_ylim(0, 2.0)
        ax_overlap.grid(True)

        return fig, ax, ax_metrics, ax_dist, ax_overlap

    def setup_plot_elements(self, ax, ax_metrics, ax_dist, ax_overlap):
        print("SETuo plots ELEMENTS")
        """Setup plot elements (lines, markers, patches)."""
        # Coverage lines
        cov_line, = ax_metrics.plot([], [], linewidth=2, label="Total coverage ratio")
        marie_line, = ax_metrics.plot([], [], "g-", linewidth=2, label="Marie contribution")
        thor_line, = ax_metrics.plot([], [], "c-", linewidth=2, label="Thor contribution")
        ax_metrics.legend(loc="upper left")

        overlap_ratio, = ax_overlap.plot([], [], linewidth=2, label="Overlap ratio")
        marie_overlap, = ax_overlap.plot([], [], "g-", linewidth=2, label="Marie overlap ratio")
        thor_overlap, = ax_overlap.plot([], [], "c-", linewidth=2, label="Thor overlap ratio")
        ax_overlap.legend(loc="upper left")

        # Distance lines
        gm_dist_line, = ax_dist.plot([], [], "m-", linewidth=2, label="Grethe-Marie distance")
        gt_dist_line, = ax_dist.plot([], [], "b-", linewidth=2, label="Grethe-Thor distance")
        ax_dist.legend(loc="upper left")

        # Overlay heatmap and obstacles
        heat = self.env_marie.data
        obstacle_img = self.env_marie.Collision_area.astype(float)
        extent = self.grid_extent()

        heat_cmap = LinearSegmentedColormap.from_list("white_orange", ["white", "orange"])
        ax.imshow(
            heat,
            origin="lower",
            extent=extent,
            cmap=heat_cmap,
            vmin=0,
            vmax=1,
            alpha=1.0,
            interpolation="bilinear"
        )

        if self.treasure_location is not None:
            treasure_N, treasure_E = self.env_marie.grid_to_world(
                self.treasure_location[0], self.treasure_location[1]
            )
            ax.plot(treasure_E, treasure_N, "y*", markersize=15, label="Treasure")
            #ax.legend(loc="upper right")

        ax.imshow(
            obstacle_img,
            origin="lower",
            extent=extent,
            cmap="Reds",
            alpha=0.6,
        )

        # Vehicle markers
        marie_dot, = ax.plot([], [], "go", markersize=8, label="Marie")
        thor_dot, = ax.plot([], [], "co", markersize=8, label="Thor")
        grethe_dot, = ax.plot([], [], "mo", markersize=10, label="Grethe")

        # Safety bubbles
        marie_bubble = Circle((0, 0), radius=5 * self.resolution, fill=False, linewidth=2)
        thor_bubble = Circle((0, 0), radius=5 * self.resolution, fill=False, linewidth=2)
        grethe_bubble = Circle((0, 0), radius=5 * self.resolution, fill=False, linewidth=2)

        # Communication range circle for grethe
        grethe_comm = Circle((0, 0), radius=self.grethe.comm_range_m, fill=False, linestyle="--", linewidth=2)

        ax.add_patch(marie_bubble)
        ax.add_patch(thor_bubble)
        ax.add_patch(grethe_bubble)
        ax.add_patch(grethe_comm)

        # Covered scatters (world points)
        marie_cov_sc = ax.scatter([], [], s=12, c="g", alpha=0.75, label="Marie covered")
        thor_cov_sc = ax.scatter([], [], s=12, c="c", alpha=0.75, label="Thor covered")

        ax.legend(loc="upper right")

        return (marie_dot, thor_dot, grethe_dot, marie_bubble, thor_bubble, grethe_bubble, grethe_comm,
                cov_line, marie_line, thor_line,
                overlap_ratio, marie_overlap, thor_overlap, gm_dist_line, gt_dist_line, marie_cov_sc, thor_cov_sc
                )

    # =====================================================================
    # Helper Functions (Static and Instance)
    # =====================================================================

    def grid_extent(self):
        """Get grid extent in world coordinates."""
        H, W = self.shape
        res = self.resolution
        N0, E0 = self.origin
        return [E0, E0 + W * res, N0, N0 + H * res]


    @staticmethod
    def covered_grid_to_world_points(env, covered_mask: np.ndarray):
        """Convert covered grid mask to world point coordinates."""
        ii, jj = np.where(covered_mask)
        if ii.size == 0:
            return np.array([]), np.array([])
        N0, E0 = env.origin
        res = env.resolution
        N = N0 + (ii + 0.5) * res
        E = E0 + (jj + 0.5) * res
        return N, E


    @staticmethod
    def disk_mask(env, center_NE, radius_m):
        """Create boolean disk mask in grid coordinates."""
        H, W = env.shape
        N0, E0 = env.origin
        res = env.resolution
        cN, cE = center_NE
        ci = (cN - N0) / res
        cj = (cE - E0) / res
        ii, jj = np.indices((H, W))
        r_cells = float(radius_m / res)
        return ((ii - ci) ** 2 + (jj - cj) ** 2) <= (r_cells ** 2)



    # =====================================================================
    # Vehicle Planning Methods
    # =====================================================================

    def update_marie_plan(self, S, avoid_thor, avoid_grethe, neighborhood=8):
        """Generate coverage plan for Marie."""
        env = self.marie.internal_map

        env.Within_range = self.marie.set_safety_bubble(self.grethe.pos, self.grethe.comm_range_m)

        bubble_other = np.zeros(env.shape, dtype=bool)
        bubble_other |= avoid_thor
        bubble_other |= avoid_grethe

        self.marie.internal_map.dynamic_Collision_area = bubble_other
        self.marie.internal_map.set_collision_area()

        start_ij = self.marie.internal_map.world_to_grid(self.marie.pos[0], self.marie.pos[1])
        seed = env.select_seed_ij(
            start_location=start_ij,
            w_range=self.marie_seed_w_range,
            w_safe=self.marie_seed_w_safe,
            w_explore=self.marie_seed_w_explore,
            w_exploit=self.marie_seed_w_exploit,
        )

        if seed is None:
            return None

        surface_mask, idx = env.grow_surface_from_seed(
            seed_ij=start_ij,
            S=S,
            neighborhood=neighborhood,
            w_range=self.marie_w_range,
            w_safe=self.marie_w_safe,
            w_explore=self.marie_w_explore,
            w_exploit=self.marie_w_exploit,
            min_safe_dist_m=self.marie_min_safe_dist
        )

        if idx is not None:
            endN, endE = env.grid_to_world(int(idx[0]), int(idx[1]))
            self.marie.pos[:] = [float(endN), float(endE)]
        else:
            endN, endE = env.grid_to_world(int(seed[0]), int(seed[1]))
            self.marie.pos[:] = [float(endN), float(endE)]

        return surface_mask

    def update_thor_plan(self, S, avoid_marie, avoid_grethe, neighborhood=8):
        """Generate coverage plan for Thor."""
        env = self.thor.internal_map

        env.Within_range = self.thor.set_safety_bubble(self.grethe.pos, self.grethe.comm_range_m)

        bubble_other = np.zeros(env.shape, dtype=bool)
        bubble_other |= avoid_marie
        bubble_other |= avoid_grethe

        self.thor.internal_map.dynamic_Collision_area = bubble_other
        self.thor.internal_map.set_collision_area()

        start_ij = self.thor.internal_map.world_to_grid(self.thor.pos[0], self.thor.pos[1])
        seed = env.select_seed_ij(
            start_location=start_ij,
            w_range=self.thor_seed_w_range,
            w_safe=self.thor_seed_w_safe,
            w_explore=self.thor_seed_w_explore,
            w_exploit=self.thor_seed_w_exploit,
        )

        if seed is None:
            return None

        surface_mask, idx = env.grow_surface_from_seed(
            seed_ij=start_ij,
            S=S,
            neighborhood=neighborhood,
            w_range=self.thor_w_range,
            w_safe=self.thor_w_safe,
            w_explore=self.thor_w_explore,
            w_exploit=self.thor_w_exploit,
            min_safe_dist_m=self.thor_min_safe_dist
        )

        self.thor.last_surface = surface_mask
        self.thor.last_idx = idx
        if idx is not None:
            endN, endE = env.grid_to_world(int(idx[0]), int(idx[1]))
            self.thor.pos[:] = [float(endN), float(endE)]
        else:
            endN, endE = env.grid_to_world(int(seed[0]), int(seed[1]))
            self.thor.pos[:] = [float(endN), float(endE)]

        return surface_mask

    def update_grethe_plan(self, S, avoid_marie, avoid_thor, neighborhood=8):
        """Generate coverage plan for Grethe."""
        env = self.grethe.internal_map

        env.Within_range = self.grethe.set_safety_bubble(self.marie.pos, self.marie.comm_range_m)
        env.Within_range |= self.grethe.set_safety_bubble(self.thor.pos, self.thor.comm_range_m)

        bubble_other = np.zeros(env.shape, dtype=bool)
        bubble_other |= avoid_marie
        bubble_other |= avoid_thor

        self.grethe.internal_map.dynamic_Collision_area = bubble_other
        self.grethe.internal_map.set_collision_area()

        start_ij = env.world_to_grid(self.grethe.pos[0], self.grethe.pos[1])
        seed = env.select_seed_ij(
            start_location=start_ij,
            w_range=self.grethe_seed_w_range,
            w_safe=self.grethe_seed_w_safe,
            w_explore=self.grethe_seed_w_explore,
            w_exploit=self.grethe_seed_w_exploit,
        )

        if seed is None:
            return None

        surface_mask, idx = env.grow_surface_from_seed(
            seed_ij=start_ij,
            S=S,
            neighborhood=neighborhood,
            w_range=self.grethe_w_range,
            w_safe=self.grethe_w_safe,
            w_explore=self.grethe_w_explore,
            w_exploit=self.grethe_w_exploit,
            min_safe_dist_m=self.grethe_min_safe_dist
        )

        self.grethe.last_surface = surface_mask
        self.grethe.last_idx = idx

        if idx is not None:
            endN, endE = env.grid_to_world(int(idx[0]), int(idx[1]))
            self.grethe.pos[:] = [float(endN), float(endE)]
        else:
            endN, endE = env.grid_to_world(int(seed[0]), int(seed[1]))
            self.grethe.pos[:] = [float(endN), float(endE)]

        return surface_mask

    # =====================================================================
    # Main Simulation Method
    # =====================================================================

    def run(self, T=100, S_marie=80, S_thor=80, S_grethe=20, neighborhood=8,
            # Marie parameters
            marie_w_range=0.1, marie_w_safe=0.5, marie_w_explore=1.0, marie_w_exploit=0.5,
            marie_min_safe_dist=6.0,
            marie_seed_w_range=0.5, marie_seed_w_safe=0.0, marie_seed_w_explore=0.9, marie_seed_w_exploit=0.5,
            # Thor parameters
            thor_w_range=0.2, thor_w_safe=1.0, thor_w_explore=1.0, thor_w_exploit=0.5,
            thor_min_safe_dist=6.0,
            thor_seed_w_range=0.5, thor_seed_w_safe=0.9, thor_seed_w_explore=0.9, thor_seed_w_exploit=0.5,
            # Grethe parameters
            grethe_w_range=1.0, grethe_w_safe=1.0, grethe_w_explore=0.01, grethe_w_exploit=0.09,
            grethe_min_safe_dist=6.0,
            grethe_seed_w_range=1.0, grethe_seed_w_safe=0.0, grethe_seed_w_explore=0.2, grethe_seed_w_exploit=0.01):
        """
        Run the full simulation.

        Args:
            T: Number of simulation iterations
            S_marie: Surface size for Marie
            S_thor: Surface size for Thor
            S_grethe: Surface size for Grethe
            neighborhood: Neighborhood type (4 or 8)
            marie_*: Marie's planning parameters
            thor_*: Thor's planning parameters
            grethe_*: Grethe's planning parameters
        """
        # Store planning parameters
        self.marie_w_range = marie_w_range
        self.marie_w_safe = marie_w_safe
        self.marie_w_explore = marie_w_explore
        self.marie_w_exploit = marie_w_exploit
        self.marie_min_safe_dist = marie_min_safe_dist
        self.marie_seed_w_range = marie_seed_w_range
        self.marie_seed_w_safe = marie_seed_w_safe
        self.marie_seed_w_explore = marie_seed_w_explore
        self.marie_seed_w_exploit = marie_seed_w_exploit

        self.thor_w_range = thor_w_range
        self.thor_w_safe = thor_w_safe
        self.thor_w_explore = thor_w_explore
        self.thor_w_exploit = thor_w_exploit
        self.thor_min_safe_dist = thor_min_safe_dist
        self.thor_seed_w_range = thor_seed_w_range
        self.thor_seed_w_safe = thor_seed_w_safe
        self.thor_seed_w_explore = thor_seed_w_explore
        self.thor_seed_w_exploit = thor_seed_w_exploit

        self.grethe_w_range = grethe_w_range
        self.grethe_w_safe = grethe_w_safe
        self.grethe_w_explore = grethe_w_explore
        self.grethe_w_exploit = grethe_w_exploit
        self.grethe_min_safe_dist = grethe_min_safe_dist
        self.grethe_seed_w_range = grethe_seed_w_range
        self.grethe_seed_w_safe = grethe_seed_w_safe
        self.grethe_seed_w_explore = grethe_seed_w_explore
        self.grethe_seed_w_exploit = grethe_seed_w_exploit

        # ...existing code...
        self.setup_environments()
        self.add_static_obstacles()
        self.setup_vehicles(
            marie_pos=[0.0, 0.0, 0.0],
            thor_pos=[5.0, 2.0, 0.0],
            grethe_pos=[2.0, 5.0, 0.0],
            marie_safety_bubble=1,
            thor_safety_bubble=1,
            grethe_safety_bubble=2,
            marie_comm_range=1,
            thor_comm_range=2,
            grethe_comm_range=5
        )
        self.set_treasure_location()
        self.mark_initial_covered()

        # Create plot
        fig, ax, ax_metrics, ax_dist, ax_overlap = self.setup_plot()
        plot_elements = self.setup_plot_elements(ax, ax_metrics, ax_dist, ax_overlap)
        marie_dot, thor_dot, grethe_dot, marie_bubble, thor_bubble, grethe_bubble, \
            grethe_comm, cov_line, marie_line, thor_line, overlap_ratio, marie_overlap, thor_overlap, \
            gm_dist_line, gt_dist_line, marie_cov_sc, thor_cov_sc = plot_elements

        fig.tight_layout()

        # Animation functions
        def init():
            print("INIT")
            marie_dot.set_data([self.marie.pos[1]], [self.marie.pos[0]])
            thor_dot.set_data([self.thor.pos[1]], [self.thor.pos[0]])
            grethe_dot.set_data([self.grethe.pos[1]], [self.grethe.pos[0]])

            cov_line.set_data([], [])
            marie_line.set_data([], [])
            thor_line.set_data([], [])
            overlap_ratio.set_data([], [])
            marie_overlap.set_data([], [])
            thor_overlap.set_data([], [])
            gm_dist_line.set_data([], [])
            gt_dist_line.set_data([], [])

            return (
                marie_dot, thor_dot, grethe_dot,
                marie_bubble, thor_bubble, grethe_bubble, grethe_comm,
                cov_line, marie_line, thor_line, overlap_ratio, marie_overlap, thor_overlap,
                gm_dist_line, gt_dist_line,marie_cov_sc, thor_cov_sc
            )

        def update(frame):

            avoid_thor = self.thor.set_safety_bubble(self.thor.pos, self.thor.safety_bubble_radius)
            avoid_marie = self.marie.set_safety_bubble(self.marie.pos, self.marie.safety_bubble_radius)
            avoid_grethe = self.grethe.set_safety_bubble(self.grethe.pos, self.grethe.safety_bubble_radius)

            # Marie coverage

            marie_surface = self.update_marie_plan(S_marie, avoid_thor, avoid_grethe, neighborhood)
            if marie_surface is not None:
                try:
                    N, E = self.covered_grid_to_world_points(self.marie.internal_map, marie_surface)
                    self.marie_cov_N = np.concatenate((self.marie_cov_N, N))
                    self.marie_cov_E = np.concatenate((self.marie_cov_E, E))
                        #self.marie_cov_pl.append(self.covered_grid_to_world_points (self.marie.internal_map, marie_surface))


                except Exception as e:
                    print(f"Error: {e}")
                print("UPDATE frame Thor")
            self.marie.internal_map.mark_covered(marie_surface)
            self.thor.internal_map.update_coverage(marie_surface)
            self.grethe.internal_map.update_coverage(marie_surface)

            # Thor coverage
            thor_surface = self.update_thor_plan(S_thor, avoid_marie, avoid_grethe, neighborhood)
            if thor_surface is not None:
                try:
                    N, E = self.covered_grid_to_world_points(self.thor.internal_map, thor_surface)
                    self.thor_cov_N = np.concatenate((self.thor_cov_N, N))
                    self.thor_cov_E = np.concatenate((self.thor_cov_E, E))

                    #self.thor_cov_pl.append(self.covered_grid_to_world_points (self.thor.internal_map, thor_surface))
                except Exception as e:
                    print(f"Error: {e}")

                self.thor.internal_map.mark_covered(thor_surface)
                self.marie.internal_map.update_coverage(thor_surface)
                self.grethe.internal_map.update_coverage(thor_surface)

                # Grethe coverage
                grethe_surface = self.update_grethe_plan(S_grethe, avoid_marie, avoid_thor, neighborhood)

            # Update vehicle positions
            marie_dot.set_data([self.marie.pos[1]], [self.marie.pos[0]])
            thor_dot.set_data([self.thor.pos[1]], [self.thor.pos[0]])
            grethe_dot.set_data([self.grethe.pos[1]], [self.grethe.pos[0]])

            # Update bubbles
            marie_bubble.center = (self.marie.pos[1], self.marie.pos[0])
            thor_bubble.center = (self.thor.pos[1], self.thor.pos[0])
            grethe_bubble.center = (self.grethe.pos[1], self.grethe.pos[0])
            grethe_comm.center = (self.grethe.pos[1], self.grethe.pos[0])

            # Update title
            ax.set_title(
                f"Iter {frame + 1}/{T} | "
                f"Marie covered={int(self.marie.internal_map.Covered_area.sum())} | "
                f"Thor covered={int(self.thor.internal_map.Covered_area.sum())}"
            )

            # Update metrics
            available_area = self.marie.internal_map.available_area()
            total_ratio = float(
                self.marie.internal_map.total_Covered_area.sum() / available_area.sum()
            )

            marie_ratio = float(
                self.marie.internal_map.Covered_area.sum() / available_area.sum()
            )

            thor_ratio = float(
                self.thor.internal_map.Covered_area.sum() / available_area.sum()
            )

            grethe_marie_dist = float(np.hypot(
                self.grethe.pos[0] - self.marie.pos[0],
                self.grethe.pos[1] - self.marie.pos[1]
            ))

            grethe_thor_dist = float(np.hypot(
                self.grethe.pos[0] - self.thor.pos[0],
                self.grethe.pos[1] - self.thor.pos[1]
            ))

            self.frames_hist.append(frame + 1)
            self.coverage_hist.append(total_ratio)
            self.marie_hist.append(marie_ratio)
            self.thor_hist.append(thor_ratio)
            self.overlap_r.append(
                float(self.marie.internal_map.overlap_ratio) + float(self.thor.internal_map.overlap_ratio))
            self.marie_overlap_r.append(self.marie.internal_map.overlap_ratio)
            self.thor_overlap_r.append(self.thor.internal_map.overlap_ratio)
            self.grethe_marie_dist_hist.append(grethe_marie_dist)
            self.grethe_thor_dist_hist.append(grethe_thor_dist)

            cov_line.set_data(self.frames_hist, self.coverage_hist)
            marie_line.set_data(self.frames_hist, self.marie_hist)
            thor_line.set_data(self.frames_hist, self.thor_hist)
            overlap_ratio.set_data(self.frames_hist, self.overlap_r)
            marie_overlap.set_data(self.frames_hist, self.marie_overlap_r)
            thor_overlap.set_data(self.frames_hist, self.thor_overlap_r)
            gm_dist_line.set_data(self.frames_hist, self.grethe_marie_dist_hist)
            gt_dist_line.set_data(self.frames_hist, self.grethe_thor_dist_hist)

            max_dist = max(
                max(self.grethe_marie_dist_hist, default=1.0),
                max(self.grethe_thor_dist_hist, default=1.0)
            )
            ax_dist.set_ylim(0, max_dist * 1.1)

            # Draw coverage scatters points

            if self.marie_cov_N.size > 0:
                marie_cov_sc.set_offsets(np.c_[self.marie_cov_E, self.marie_cov_N])

            if self.thor_cov_N.size > 0:
                thor_cov_sc.set_offsets(np.c_[self.thor_cov_E, self.thor_cov_N])

            return (
                marie_dot, thor_dot, grethe_dot,
                marie_bubble, thor_bubble, grethe_bubble, grethe_comm,
                cov_line, marie_line, thor_line,
                overlap_ratio, marie_overlap, thor_overlap,
                gm_dist_line, gt_dist_line,marie_cov_sc, thor_cov_sc
            )

        anim = FuncAnimation(fig, update, frames=T, init_func=init, interval=1500, blit=False, repeat=False)
        plt.show()

