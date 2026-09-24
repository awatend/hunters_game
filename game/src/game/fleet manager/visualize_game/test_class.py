"""
Test class for hunters game coverage planning simulation.
Contains all helper functions and methods for running the simulation.
"""
from environment import Environment
from vehicles import AUV, ASV
from cell_geometry import CellGeometry

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
import shapely.geometry as sg

from typing import Tuple
import math
import random
import os
from marcov import (
    Vehicle,
    SwathGenerator,
    OptimizeSwathLengths,
    RoutePlanner,
    RouteOrder,
    PathPlanner,
    TurnType,
    plot_area,
    plot_headland,
    plot_swaths,
    plot_route,
    plot_path,
    save_figure,
)

NED = Tuple[float, float, float]

class HuntersGameSimulation:
    """Main simulation class for multi-vehicle coverage planning."""

    def __init__(self, geometry: "CellGeometry"):
        """
        Initialize the simulation environment.

        Args:
            geometry: CellGeometry instance defining the grid layout and
                      cell shape.  Change only this argument (in test_config.py)
                      to switch between rectangular, hexagonal, or any future
                      cell shape.
        """
        self.geom       = geometry
        # Backward-compat shims used by plotting helpers below
        self.shape      = geometry.shape
        self.resolution = geometry.cell_spacing_m
        self.origin     = geometry.origin
        
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
        self.overlap_r= []
        self.marie_overlap_r= []
        self.thor_overlap_r= []
        self.grethe_marie_dist_hist = []
        self.grethe_thor_dist_hist = []
        self.marie_cov_pl = []
        self.thor_cov_pl = []
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

        # Vehicles INITIAL POSITIONS
        self.marie_init = [0, 0, 0]
        self.thor_init = [0, 0, 0]
        self.grethe_init = [0, 0, 0]

        self.prior_data=np.full(self.geom.shape, 0.01, dtype=np.float32)

    def setup_environments(self):
        """Create and configure environment instances."""
        self.env_marie  = Environment(self.geom)
        self.env_thor   = Environment(self.geom)
        self.env_grethe = Environment(self.geom)

        for env in (self.env_marie, self.env_thor, self.env_grethe):
            env.Initial_area[:, :] = True
            env.update(self.prior_data)

    def add_static_obstacles(self):
        """Add static obstacles to all environment maps."""
        for env in (self.env_marie, self.env_thor, self.env_grethe):
            env.static_Collision_area[40:60, 60:110] = True
            env.static_Collision_area[120:160, 160:190] = True
            env.static_Collision_area[70:95, 220:260] = True
            env.prior = self.env_grethe.prior

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
        """Setup matplotlib figure and subplots."""
        fig, ((ax, ax_metrics), (ax_dist, ax_overlap)) = plt.subplots(
            2, 2, figsize=(20, 8), gridspec_kw={"width_ratios": [ 1, 1], "height_ratios": [1, 1]}
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
        ax_overlap.set_ylim(0, 1.2)
        return fig, ax, ax_metrics, ax_dist, ax_overlap

    def setup_plot_elements(self, ax, ax_metrics, ax_dist, ax_overlap):
        """Setup plot elements (lines, markers, patches)."""
        # Coverage lines
        cov_line, = ax_metrics.plot([], [], linewidth=2, label="Total coverage ratio")
        marie_line, = ax_metrics.plot([], [], "g-", linewidth=2, label="Marie contribution")
        thor_line, = ax_metrics.plot([], [], "c-", linewidth=2, label="Thor contribution")
        ax_metrics.legend(loc="upper left")

        overlap_ratio, =ax_overlap.plot([], [], linewidth=2, label="Overlap ratio")
        marie_overlap, =ax_overlap.plot([], [], "g-", linewidth=2, label="Marie overlap ratio")
        thor_overlap, =ax_overlap.plot([], [], "c-", linewidth=2, label="Thor overlap ratio")
        ax_overlap.legend(loc="upper left")

        # Distance lines
        gm_dist_line, = ax_dist.plot([], [], "m-", linewidth=2, label="Grethe-Marie distance")
        gt_dist_line, = ax_dist.plot([], [], "b-", linewidth=2, label="Grethe-Thor distance")
        ax_dist.legend(loc="upper left")

        # Overlay heatmap and obstacles
        heat = self.env_marie.prior
        obstacle_img = self.env_marie.static_Collision_area.astype(float)
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
            ax.legend(loc="upper right")

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

        ax.legend(loc="upper right")

        return (marie_dot, thor_dot, grethe_dot, marie_bubble, thor_bubble, grethe_bubble, grethe_comm,
                cov_line, marie_line, thor_line,
                 overlap_ratio, marie_overlap, thor_overlap,gm_dist_line, gt_dist_line
                )

    # =====================================================================
    # Helper Functions (Static and Instance)
    # =====================================================================

    def grid_extent(self):
        """Get grid extent in world coordinates. Delegates to CellGeometry."""
        return list(self.geom.world_extent())

    @staticmethod
    def closest_point(pos, start, end):
        """Find closest point between position and two endpoints."""
        d_start = (start.getY() - pos[0]) ** 2 + (start.getX() - pos[1]) ** 2
        d_end = (end.getY() - pos[0]) ** 2 + (end.getX() - pos[1]) ** 2
        return start if d_start < d_end else end

    @staticmethod
    def covered_grid_to_world_points(env, covered_mask: np.ndarray):
        """Convert covered grid mask to world point coordinates."""
        return env.geom.cells_to_world(covered_mask)

    @staticmethod
    def polygon_world_to_grid(geom, polygon_NE):
        """
        Convert a world-space Shapely polygon to a boolean cell mask.

        Parameters
        ----------
        geom : CellGeometry
            Active geometry (determines rasterisation method).
        polygon_NE : shapely geometry
            Polygon or MultiPolygon in world coordinates.

        Returns
        -------
        mask : np.ndarray[bool]
        coords : np.ndarray  shape (K, 2) of (N, E) polygon vertices
        """
        mask = geom.polygon_to_mask(polygon_NE, geom.shape)

        if polygon_NE is None or polygon_NE.is_empty:
            return mask, np.empty((0, 2), dtype=float)

        # Keep visualization compatible with Polygon / MultiPolygon coverage.
        if isinstance(polygon_NE, sg.MultiPolygon):
            polygon_NE = max(polygon_NE.geoms, key=lambda polygon: polygon.area)

        coords = np.array([(float(y), float(x)) for x, y in polygon_NE.exterior.coords], dtype=float)
        return mask, coords

    @staticmethod
    def disk_mask(env, center_NE, radius_m):
        """
        Create boolean mask of all cells whose centres are within radius_m
        of center_NE.  Uses the geometry's cell_to_world for correct distances.
        """
        H, W = env.shape
        cN, cE = center_NE
        mask = np.zeros((H, W), dtype=bool)
        ci, cj = env.geom.world_to_cell(cN, cE)
        # Upper bound on cells to check: radius / min_cell_spacing + 2
        r_steps = int(radius_m / env.geom.cell_spacing_m) + 2
        for ni, nj in env.geom.bubble_cells(ci, cj, r_steps):
            nN, nE = env.geom.cell_to_world(ni, nj)
            if (nN - cN) ** 2 + (nE - cE) ** 2 <= radius_m ** 2:
                mask[ni, nj] = True
        # Also check center itself
        ccN, ccE = env.geom.cell_to_world(ci, cj)
        if (ccN - cN) ** 2 + (ccE - cE) ** 2 <= radius_m ** 2:
            mask[ci, cj] = True
        return mask


    @staticmethod
    def route_planner(polygon, output_dir="pictures",):
        """
        Generate a complete coverage plan for a Shapely polygon.

        Parameters
        ----------
        polygon : shapely.geometry.Polygon
            Survey area.

        output_dir : str, optional
            Directory where figures are saved. If None, figures are not
            automatically saved.

        Returns
        -------
        tuple
            ``connections, start_pt, end_pt, covered_area``

            where ``covered_area`` is a Shapely Polygon or MultiPolygon.
        """

        if polygon is None or polygon.is_empty:
            return None

        if isinstance(polygon, sg.MultiPolygon):
            polygon = max(polygon.geoms, key=lambda geom: geom.area)

        if not polygon.is_valid:
            polygon = polygon.buffer(0)
            if polygon.is_empty:
                return None

        # ------------------------------------------------------------------
        # 1. Vehicle
        # ------------------------------------------------------------------

        vehicle = Vehicle(width=4.0, cov_width=25.0, cruise_speed=2.0, turn_speed=1.0, max_curvature=1 / 5.0, name="USV-Survey",)

        # 2. Swath generation
        generator = SwathGenerator(vehicle)
        swath_result = generator.generate(polygon, angle=OptimizeSwathLengths(step=math.pi / 36.0,), headland_distance=7.0,)

        if not swath_result.swaths:
            print("No valid swaths generated for this polygon.")
            return None



        # 3. Route planning
        route_planner = RoutePlanner()
        route_result = route_planner.plan(swath_result, order=RouteOrder.SPIRAL,)

        if not route_result.swaths:
            print("No valid route generated.")
            return None

        # 4. Path planning
        print("planing path")
        path_planner = PathPlanner(vehicle, turn_type=TurnType.DUBINS, step=1.0,)
        path = path_planner.plan(route_result, headland=polygon,)
        print("planed path")

        if not path.states:
            print("No valid vehicle path generated.")
            return None

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        fig_all, ax = plt.subplots(figsize=(10, 9))
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        plot_headland(polygon, swath_result, ax=ax)
        plot_route(route_result, ax=ax, show_arrows=True, show_order=True)
        plot_path(path, ax=ax)

        if output_dir is not None:
            save_figure(fig_all, os.path.join(output_dir, f"out_{len(path.states)}.png"))

        # 5. Covered area

        covered_parts = [swath.buffer(vehicle.cov_width / 2.0) for swath in route_result.swaths if not swath.is_empty]
        if covered_parts:
            from shapely.ops import unary_union
            covered_area = unary_union(covered_parts)
            # Do not let the coverage extend outside the survey polygon.
            covered_area = covered_area.intersection(polygon)

            # Optional small geometric cleanup.
            covered_area = covered_area.buffer(0)
        else:
            covered_area = polygon.__class__()

        # 6. Start / end
        start_state = path.states[0]
        end_state = path.states[-1]

        start_pt = (start_state.x, start_state.y)
        end_pt = (end_state.x, end_state.y)

        # 7. Visualization
        if output_dir is not None:
            from marcov import plot_coverage

            fig_overview = plot_coverage(
                polygon,
                swath_result=swath_result,
                route_result=route_result,
                path=path,
                show_headland=True,
                show_arrows=True,
                show_order=True,
                title="Maritime Coverage Plan",
            )

            save_figure(fig_overview, os.path.join(output_dir, f"output{len(path.states)}.png",),)

        print("\nPlanning successful")

        return (start_pt, end_pt, covered_area)

    @staticmethod
    def is_treasure_founded(mask, treasure_location):
        """Check if treasure is found in covered area."""
        if treasure_location is None:
            return False
        i, j = treasure_location
        return mask[i, j]

    # =====================================================================
    # Vehicle Planning Methods
    # =====================================================================

    def update_marie_plan(self, S, avoid_thor, avoid_grethe, neighborhood=8):
        """Generate coverage plan for Marie."""
        env = self.marie.internal_map

        self.marie.internal_map.Within_range = self.marie.set_safety_bubble(self.grethe.pos, self.grethe.comm_range_m)

        bubble_other = np.zeros(self.marie.internal_map.shape, dtype=bool)
        bubble_other |= avoid_thor
        bubble_other |= avoid_grethe

        self.marie.internal_map.dynamic_Collision_area = bubble_other
        self.marie.internal_map.set_collision_area()

        start_ij = env.world_to_grid(self.marie.pos[0], self.marie.pos[1])

        seed = env.select_seed_ij(
            start_location=start_ij
        )

        if seed is None:
            return None

        surface_mask, idx = env.grow_surface_from_seed(
            seed_ij=seed,
            S=S,
            neighborhood=neighborhood,
            w_range=self.marie_w_range,
            w_safe=self.marie_w_safe,
            w_explore=self.marie_w_explore,
            w_exploit=self.marie_w_exploit,
            min_safe_dist_m=self.marie_min_safe_dist,
            function="hunter"
        )

        #print("Marie plan updated: surface size =", np.sum(surface_mask))
        return surface_mask, idx

    def update_thor_plan(self, S, avoid_marie, avoid_grethe, neighborhood=8):
        """Generate coverage plan for Thor."""
        env = self.thor.internal_map
        self.thor.internal_map.Within_range = self.thor.set_safety_bubble(self.grethe.pos, self.grethe.comm_range_m)

        bubble_other = np.zeros(self.thor.internal_map.shape, dtype=bool)
        bubble_other |= avoid_marie
        bubble_other |= avoid_grethe

        self.thor.internal_map.dynamic_Collision_area = bubble_other
        self.thor.internal_map.set_collision_area()

        start_ij = self.thor.internal_map.world_to_grid(self.thor.pos[0], self.thor.pos[1])

        seed = env.select_seed_ij(
            start_location=start_ij

        )

        if seed is None:
            return None

        surface_mask, idx = env.grow_surface_from_seed(
            seed_ij=seed,
            S=S,
            neighborhood=neighborhood,
            w_range=self.thor_w_range,
            w_safe=self.thor_w_safe,
            w_explore=self.thor_w_explore,
            w_exploit=self.thor_w_exploit,
            min_safe_dist_m=self.thor_min_safe_dist,
            function="hunter"
        )

        self.thor.last_surface = surface_mask
        self.thor.last_idx = idx

        return surface_mask, idx

    def update_grethe_plan(self, S, avoid_marie, avoid_thor, neighborhood=8):
        """Generate coverage plan for Grethe."""
        env = self.grethe.internal_map

        self.grethe.internal_map.Within_range = self.grethe.set_safety_bubble(self.marie.pos, self.marie.comm_range_m)
        self.grethe.internal_map.Within_range |= self.grethe.set_safety_bubble(self.thor.pos, self.thor.comm_range_m)

        bubble_other = np.zeros(self.grethe.internal_map.shape, dtype=bool)
        bubble_other |= avoid_marie
        bubble_other |= avoid_thor

        self.grethe.internal_map.dynamic_Collision_area = bubble_other
        self.grethe.internal_map.set_collision_area()

        start_ij = env.world_to_grid(self.grethe.pos[0], self.grethe.pos[1])

        seed = env.select_seed_ij(
            start_location=start_ij
        )

        if seed is None:
            return None

        surface_mask, idx = env.grow_surface_from_seed(
            seed_ij=seed,
            S=S,
            neighborhood=neighborhood,
            w_range=self.grethe_w_range,
            w_safe=self.grethe_w_safe,
            w_explore=self.grethe_w_explore,
            w_exploit=self.grethe_w_exploit,
            min_safe_dist_m=self.grethe_min_safe_dist,
            function="support"
        )

        self.grethe.last_surface = surface_mask
        self.grethe.last_idx = idx
        if idx is not None:
            endN, endE = env.grid_to_world(int(idx[0]), int(idx[1]))
            self.grethe.pos[:] = [float(endN), float(endE)]

        return surface_mask, idx

    # =====================================================================
    # Main Simulation Method
    # =====================================================================

    def run(self, marie_init,thor_init, grethe_init, T=100, S_marie=80, S_thor=80, S_grethe=20, neighborhood=8,
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
            grethe_seed_w_range=1.0, grethe_seed_w_safe=0.0, grethe_seed_w_explore=0.2, grethe_seed_w_exploit=0.01 ,
            prior_data=np.full((200, 300), 0.0, dtype=np.float32)):
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
            marie_pos=marie_init,
            thor_pos=thor_init,
            grethe_pos=grethe_init,
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
        marie_dot, thor_dot, grethe_dot,marie_bubble, thor_bubble, grethe_bubble, \
             grethe_comm, cov_line, marie_line, thor_line,overlap_ratio, marie_overlap, thor_overlap, \
            gm_dist_line, gt_dist_line = plot_elements

        fig.tight_layout()

        # Animation functions
        def init():
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
                gm_dist_line, gt_dist_line
            )

        def update(frame):
            avoid_thor = self.thor.set_safety_bubble(self.thor.pos, self.thor.safety_bubble_radius)
            avoid_marie = self.marie.set_safety_bubble(self.marie.pos, self.marie.safety_bubble_radius)
            avoid_grethe = self.grethe.set_safety_bubble(self.grethe.pos, self.grethe.safety_bubble_radius)

            # Marie coverage
            print(f" vehicles coordinates: GRETHE {self.grethe.pos}, MARIE {self.marie.pos}, THOR {self.thor.pos}")
            result=None
            while (result is None):
                marie_surface, end_pose = self.update_marie_plan(S_marie, avoid_thor, avoid_grethe, neighborhood)
                if marie_surface is not None:
                    try:
                        survey_polygon = self.geom.mask_to_polygon(marie_surface)
                        result = self.route_planner(survey_polygon)
                        self.marie.internal_map.save_mask_polygon(marie_surface, filename=f"marie_interest{T}.png")
                    except Exception as e:
                        print(f"Error: {e}")
                if result is None:
                    self.marie.pos[:] = [self.marie.pos[0]+random.randint(-3,3)*self.marie.internal_map.resolution,
                                        self.marie.pos[1]+random.randint(-3,3)*self.marie.internal_map.resolution]
                    print(f"marie plan failed, changing position {self.marie.pos}")

            #if result is not None:
            #print("marie results non empty")
            m_start, m_end, polygon_NE_marie = result
            marie_cov_surface, covered_NE = self.polygon_world_to_grid(
                self.geom, polygon_NE_marie
            )
            self.marie.internal_map.mark_covered(marie_cov_surface)
            self.marie.last_surface = marie_cov_surface
            #self.marie.internal_map.save_mask_polygon(marie_cov_surface, filename=f"marie_covered{T}.png")

            if self.is_treasure_founded(marie_cov_surface, self.treasure_location):
                print("Treasure Founded by Marie!")
                return

            '''
            N, E=self.marie.internal_map.grid_to_world(end_pose[0], end_pose[1])
            marie_pose=[N,E]
            if self.closest_point(marie_pose, m_start, m_end) == m_end:
                self.marie.pos[0] = m_end.getY()
                self.marie.pos[1] = m_end.getX()
            else:
                self.marie.pos[0] = m_start.getY()
                self.marie.pos[1] = m_start.getX()
            '''
            endN, endE = self.marie.internal_map.grid_to_world(end_pose[0], end_pose[1])
            self.marie.pos[:] = [endN, endE]

            #print(f"Marie POSE {self.marie.pos}")
            #print(f"Marie POSE before conversion {end_pose}")

            self.thor.internal_map.update_coverage(marie_cov_surface)
            self.grethe.internal_map.update_coverage(marie_cov_surface)

            self.marie_cov_pl.append(covered_NE)
            #print(f"Marie polygon {covered_NE}")

                #print(f"polygon appended {covered_NE}")



            # Thor coverage
            result=None
            while (result is None):
                print("result not none")
                thor_surface, end_pose = self.update_thor_plan(S_thor, avoid_marie, avoid_grethe, neighborhood)

                if thor_surface is not None:
                    '''
                    print(f"polygon of interest")

                    for i in range(99):
                        for j in range(179):
                            if thor_surface[i, j] == True:
                                print(f"{i} {j}")
                    '''
                    survey_polygon = self.geom.mask_to_polygon(thor_surface)
                    result = self.route_planner(survey_polygon)
                    self.thor.internal_map.save_mask_polygon(thor_surface, filename=f"thor_interest{T}.png")
                if result is None:
                    self.thor.pos[:] = [self.thor.pos[0]+random.uniform(-3,3)*self.thor.internal_map.resolution,
                                        self.thor.pos[1]+random.uniform(-3,3)*self.thor.internal_map.resolution]
                    print(f"thor plan failed, changing position {self.thor.pos}")

            t_start, t_end, polygon_NE_thor = result
            thor_cov_surface, thor_covered_NE = self.polygon_world_to_grid(
                self.geom, polygon_NE_thor
            )

            self.thor.internal_map.mark_covered(thor_cov_surface)
            self.thor.last_surface = thor_cov_surface
            # Visualizing the actually covered surface
            print(f"thor covered surface visualization")
            self.thor.internal_map.save_mask_polygon(thor_cov_surface, filename=f"thor_covered{T}.png")


            if self.is_treasure_founded(thor_cov_surface, self.treasure_location):
                print("Treasure Founded by Thor!")
                return
            '''
            N, E=self.thor.internal_map.grid_to_world(end_pose[0], end_pose[1])
            thor_pose=[N,E]
            if self.closest_point(thor_pose, t_start, t_end) == t_end:
                self.thor.pos[0] =t_end.getY()
                self.thor.pos[1] = t_end.getX()
            else:
                self.thor.pos[0] = t_start.getY()
                self.thor.pos[1] = t_start.getX()
            '''
            endN, endE = self.thor.internal_map.grid_to_world (end_pose[0],end_pose[1])
            self.thor.pos[:] = [endN, endE]

            #print(f"thor POSE {self.thor.pos}")
            print(f"thor POSE before conversion {end_pose}")

            self.marie.internal_map.update_coverage(thor_cov_surface)
            self.grethe.internal_map.update_coverage(thor_cov_surface)

            self.thor_cov_pl.append(thor_covered_NE)
            #print(f"Thor polygon {thor_covered_NE}")


                # Grethe coverage
            print("Updating gRETHE")
            #grethe_surface, position = self.update_grethe_plan(S_grethe, avoid_marie, avoid_thor, neighborhood)
            #self.grethe.pos[0], self.grethe.pos[1]=self.grethe.internal_map.grid_to_world(position[0], position[1])
            #print(f"grethe position {self.grethe.pos}, position {position}")

            result = None
            while (result is None):
                grethe_surface, end_pose = self.update_grethe_plan(S_marie, avoid_thor, avoid_marie, neighborhood)
                if grethe_surface is not None:
                    try:
                        survey_polygon = self.geom.mask_to_polygon(grethe_surface)
                        result = self.route_planner(survey_polygon)
                        self.grethe.internal_map.save_mask_polygon(grethe_surface, filename=f"grethe_interest{T}.png")
                    except Exception as e:
                        print(f"Error: {e}")
                if result is None:
                    self.grethe.pos[:] = [self.grethe.pos[0] + random.randint(-3, 3) * self.grethe.internal_map.resolution,
                                         self.grethe.pos[1] + random.randint(-3, 3) * self.grethe.internal_map.resolution]
                    print(f"marie plan failed, changing position {self.grethe.pos}")

            # if result is not None:
            # print("marie results non empty")
            m_start, m_end, polygon_NE_grethe = result
            grethe_cov_surface, covered_NE = self.polygon_world_to_grid(
                self.geom, polygon_NE_grethe
            )
            self.grethe.internal_map.mark_covered(grethe_cov_surface)
            self.grethe.last_surface = grethe_cov_surface
            self.grethe.internal_map.save_mask_polygon(grethe_cov_surface, filename=f"grethe_covered{T}.png")

            if self.is_treasure_founded(grethe_cov_surface, self.treasure_location):
                print("Treasure Founded by Marie!")
                return

            endN, endE = self.grethe.internal_map.grid_to_world(end_pose[0], end_pose[1])
            self.grethe.pos[:] = [endN, endE]

            # print(f"Marie POSE {self.marie.pos}")
            # print(f"Marie POSE before conversion {end_pose}")

            self.marie.internal_map.update_coverage(grethe_cov_surface)
            #self.thor.internal_map.update_coverage(grethe_cov_surface)




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
            self.overlap_r.append(float (self.marie.internal_map.overlap_ratio)+ float(self.thor.internal_map.overlap_ratio))
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

            # Draw coverage polygons
            for poly in self.marie_cov_pl:
                if len(poly) == 0:
                    continue

                Nm = [p[0] for p in poly]
                Em = [p[1] for p in poly]

                patch = ax.fill(Em, Nm, color='green', alpha=1.0)[0]
                #print("plotted")

            for poly in self.thor_cov_pl:
                if len(poly) == 0:
                    continue

                Nt = poly[:, 0]
                Et = poly[:, 1]

                patch = ax.fill(Et, Nt, color='c', alpha=1.0)[0]

            return (
                marie_dot, thor_dot, grethe_dot,
                marie_bubble, thor_bubble, grethe_bubble, grethe_comm,
                cov_line, marie_line, thor_line,
                overlap_ratio, marie_overlap, thor_overlap,
                gm_dist_line, gt_dist_line
            )

        anim = FuncAnimation(fig, update, frames=T, init_func=init, interval=1500, blit=False, repeat=False)
        plt.show()
