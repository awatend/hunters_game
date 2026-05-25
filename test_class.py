"""
Test class for hunters game coverage planning simulation.
Contains all helper functions and methods for running the simulation.
"""
from debugpy.adapter.servers import connections

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
from typing import Tuple
import math
import random
import cv2
NED = Tuple[float, float, float]

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
        heat = self.env_marie.data
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
        """Get grid extent in world coordinates."""
        H, W = self.shape
        res = self.resolution
        n0, e0 = self.origin
        return [e0, e0 + W * res, n0, n0 + H * res]

    @staticmethod
    def closest_point(pos, start, end):
        """Find closest point between position and two endpoints."""
        d_start = (start.getY() - pos[0]) ** 2 + (start.getX() - pos[1]) ** 2
        d_end = (end.getY() - pos[0]) ** 2 + (end.getX() - pos[1]) ** 2
        return start if d_start < d_end else end

    @staticmethod
    def covered_grid_to_world_points(env, covered_mask: np.ndarray):
        """Convert covered grid mask to world point coordinates."""
        ii, jj = np.where(covered_mask)
        if ii.size == 0:
            return np.array([]), np.array([])
        n0, e0 = env.origin
        res = env.resolution
        N = n0 + (ii + 0.5) * res
        E = e0 + (jj + 0.5) * res
        return N, E

    @staticmethod
    def polygon_world_to_grid(shape, origin, resolution, polygon_NE):
        """Convert F2C polygon to grid mask."""
        H, W = shape
        n0, e0 = origin
        res = resolution

        contour = polygon_NE.getExteriorRing()
        coordinates = []
        for p in range(contour.size()):
            coordinates.append((contour.getGeometry(p).getY(), contour.getGeometry(p).getX()))

        coords = np.array(coordinates)
        ii = (coords[:, 0] - n0) / res
        jj = (coords[:, 1] - e0) / res
        #ii = H - ((coords[:, 0] - n0) / res)
        #jj = (W - 1)-  ((coords[:, 1] - e0) / res)
        #ii=ii[::-1] #reverse element to aaount for mirror distortion induced by the contour.GetGeometry operation
        ii = np.clip(ii, 0, H-1).astype(int)
        jj = np.clip(jj, 0, W-1).astype(int)

        # Rasterize polygon
        rr, cc = polygon(ii, jj, shape=(H, W))
        mask = np.zeros((H, W), dtype=bool)
        mask[rr, cc] = True

        return mask, coords

    @staticmethod
    def disk_mask(env, center_NE, radius_m):
        """Create boolean disk mask in grid coordinates."""
        H, W = env.shape
        n0, e0 = env.origin
        res = env.resolution
        cN, cE = center_NE
        ci = (cN - n0) / res
        cj = (cE - e0) / res
        ii, jj = np.indices((H, W))
        r_cells = float(radius_m / res)
        return ((ii - ci) ** 2 + (jj - cj) ** 2) <= (r_cells ** 2)

    @staticmethod
    def mask_to_f2c_cell(mask, scale=1.0):
        """Convert boolean mask to Fields2Cover Cell."""
        '''
        contours = measure.find_contours(mask, 0.5)
        #print(f'contours: {contours}')

        if not contours:
            raise ValueError("No polygon found in the mask.")

        main_contour = max(contours, key=len)
        tolerance = 1.5
        simplified_contour = measure.approximate_polygon(main_contour, tolerance=tolerance)
        if len(simplified_contour) < 4:
            simplified_contour = main_contour

        ring = f2c.LinearRing()

        for pt in main_contour:
            x_m = float(pt[1]+0.5) * scale
            y_m = float(pt[0]+0.5) * scale
            ring.addPoint(x_m, y_m)

        start_pt = main_contour[0]
        ring.addPoint(float(start_pt[1]+0.5) * scale, float(start_pt[0]+0.5) * scale)
        #print(f"check coordinates contour {simplified_contour} ")

        cell = f2c.Cell()
        cell.addRing(ring)
        simplified_cell = cell.simplify(1.0)
        '''
        binary_mask = (mask * 255).astype(np.uint8)

        # 2. Find contours
        # RETR_EXTERNAL: only finds the outer boundaries
        # CHAIN_APPROX_SIMPLE: removes redundant points to save memory
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #biggest_area_contour = max(contours, key=lambda c: cv2.arcLength(c, True))
        #print(f'contours: {contours}')
        cell = f2c.Cell()
        for cnt in contours:
            if len(cnt) < 3:  # Skip noise/lines that can't form a polygon
                continue

            ring = f2c.LinearRing()

            for pt in cnt:
                # Flatten the [[[x, y]]] structure
                # OpenCV provides (x, y), which matches (Easting, Northing)
                x_pixel = float(pt[0][0])
                y_pixel = float(pt[0][1])

                x_m = x_pixel * scale
                y_m = y_pixel * scale
                ring.addPoint(x_m, y_m)

            # Explicitly close the ring using the first point
            first_pt = cnt[0][0]
            ring.addPoint(float(first_pt[0])  * scale,
                          float(first_pt[1]) * scale)

            cell.addRing(ring)

        # Simplify the entire cell (handles geometry cleaning)
        simplified_cell = cell.simplify(1.0)

        #print(f"Extracted contour with")
        f2c.Visualizer.figure()
        f2c.Visualizer.plot(cell)
        #f2c.Visualizer.plot(simplified_cell.convexHull())
        #f2c.Visualizer.plot(simplified_cell)
        f2c.Visualizer.save(f"cell_and simplified{len(contours[0])}.png")

        return simplified_cell

    @staticmethod
    def route_planner(cell):
        """Generate route plan for coverage cell."""
        robot = f2c.Robot(2.0, 2.6)
        robot.setMinTurningRadius(2.5)
        robot.setMaxCurv(0.5)
        robot.setTurnVel(1.0)
        robotsensorSwath = 20

        cells_container = f2c.Cells()
        cells_container.addGeometry(cell)

        const_hl = f2c.HG_Const_gen()
        headland = const_hl.generateHeadlands(cells_container, 6.0)
        mid_headland = const_hl.generateHeadlands(cells_container, 3.0)
        if headland.size() == 0:
            #print("Skipping cell: Cell is too small to have a headland.")
            f2c.Visualizer.figure()
            f2c.Visualizer.plot(cells_container)
            f2c.Visualizer.save(f"cell too small_{len(cells_container)}.png")

            return

        n_swath = f2c.OBJ_NSwath() # Minimize the number of swaths
        swath_gen = f2c.SG_BruteForce()
        swaths = swath_gen.generateBestSwaths(n_swath , robotsensorSwath, headland)
        f_swaths = f2c.Swaths()
        for i in range(swaths.at(0).size()):
            swath = swaths.getSwath(i)
            if swath.length() >= 15.0:
                f_swaths.push_back(swath)

        filtered_swaths = f2c.SwathsByCells()
        filtered_swaths.push_back(f_swaths)

        if filtered_swaths.size() == 0 or filtered_swaths.at(0).size() == 0:
            #print("No valid swaths generated for this cell. Skipping route planning.")
            f2c.Visualizer.figure()
            f2c.Visualizer.plot(filtered_swaths)
            f2c.Visualizer.save(f"cell too small_{filtered_swaths.size()}.png")
            return

        r_planner = f2c.RP_RoutePlannerBase()
        route = r_planner.genRoute(mid_headland, filtered_swaths)

        path_planner = f2c.PP_PathPlanning()
        dubins = f2c.PP_DubinsCurves()
        final_path = path_planner.planPath(robot, route, dubins)
        #final_path.reduce(0.5)
        #print(f"Total path length: {final_path.length():.2f} meters {final_path.size()} ")
        f2c.Visualizer.figure()
        f2c.Visualizer.plot(cell)
        f2c.Visualizer.plot(mid_headland)
        f2c.Visualizer.plot(route)
        #print("route planning done.")
        #print(route)
        f2c.Visualizer.save(f"planned_path{final_path.size()}.png")
        path=final_path.reduce(10.0)
        connections=path.getStates()
        start_pt=path.atStart()
        end_pt=path.atEnd()
        size=connections.size()
        #s_connections=connections.simplify(robotsensorSwath - 1)
        '''
        for i in range (size-1):
            print(f"{connections[i+1].atEnd()}")
        print(f"start_pt: {start_pt}i+1, end_pt: {end_pt}, size: {size}")
        '''

        if final_path.size() < 0:
            return

        combined_cells = f2c.Cells()
        buffer_dist = robot.getWidth() / 2.0

        for swaths_section in filtered_swaths:
            for i in range(swaths_section.size()):
                swath = swaths_section.at(i)
                swath_poly = swath.areaCovered()
                combined_cells = combined_cells.unionOp(swath_poly)

        #f2c.Visualizer.figure()
        #f2c.Visualizer.plot(combined_cells)
        simplified_cells = combined_cells.simplify(0.1)
        simplified_cell = simplified_cells.getGeometry(0)
        s_area=simplified_cell.area()
        for i in range (simplified_cells.size()):
            #print(f"simplified cell size {simplified_cells.size()}")
            cell = simplified_cells.getGeometry(i)
            if cell.area() > s_area:
                simplified_cell = cell


        f2c.Visualizer.figure()
        f2c.Visualizer.plot(simplified_cell)
        f2c.Visualizer.save(f"covered_{final_path.size()}.png")

        print("planning successful")
       # return final_path, final_path.atStart(), final_path.atEnd(), simplified_cells.convexHull()
        return connections, start_pt, end_pt, simplified_cell

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
            grethe_seed_w_range=1.0, grethe_seed_w_safe=0.0, grethe_seed_w_explore=0.2, grethe_seed_w_exploit=0.01 ):
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
                        f2c_cell = self.mask_to_f2c_cell(marie_surface, scale=self.marie.internal_map.resolution)
                        result = self.route_planner(f2c_cell)
                        self.marie.internal_map.save_mask_polygon(marie_surface, filename=f"marie_interest{T}.png")
                    except Exception as e:
                        print(f"Error: {e}")
                if result is None:
                    self.marie.pos[:] = [self.marie.pos[0]+random.randint(-3,3)*self.marie.internal_map.resolution,
                                        self.marie.pos[1]+random.randint(-3,3)*self.marie.internal_map.resolution]
                    print(f"marie plan failed, changing position {self.marie.pos}")

            #if result is not None:
            #print("marie results non empty")
            m_path, m_start, m_end, polygon_NE_marie = result
            marie_cov_surface, covered_NE = self.polygon_world_to_grid(
                self.marie.internal_map.shape,
                self.marie.internal_map.origin,
                self.marie.internal_map.resolution,
                polygon_NE_marie
                )
            self.marie.internal_map.mark_covered(marie_cov_surface)
            self.marie.last_surface = marie_cov_surface
            self.marie.internal_map.save_mask_polygon(marie_cov_surface, filename=f"marie_covered{T}.png")

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
                thor_surface, end_pose = self.update_thor_plan(S_thor, avoid_marie, avoid_grethe, neighborhood)

                if thor_surface is not None:
                    '''
                    print(f"polygon of interest")

                    for i in range(99):
                        for j in range(179):
                            if thor_surface[i, j] == True:
                                print(f"{i} {j}")
                    '''
                    f2c_cell = self.mask_to_f2c_cell(thor_surface, scale=self.thor.internal_map.resolution)
                    result = self.route_planner(f2c_cell)
                    self.thor.internal_map.save_mask_polygon(thor_surface, filename=f"thor_interest{T}.png")
                if result is None:
                    self.thor.pos[:] = [self.thor.pos[0]+random.uniform(-3,3)*self.thor.internal_map.resolution,
                                        self.thor.pos[1]+random.uniform(-3,3)*self.thor.internal_map.resolution]
                    print(f"thor plan failed, changing position {self.thor.pos}")

            t_path, t_start, t_end, polygon_NE_thor = result
            thor_cov_surface, thor_covered_NE = self.polygon_world_to_grid(
                self.thor.internal_map.shape,
                self.thor.internal_map.origin,
                self.thor.internal_map.resolution,
                 polygon_NE_thor
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
           # print(f"thor POSE before conversion {end_pose}")

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
                        f2c_cell = self.mask_to_f2c_cell(grethe_surface, scale=self.grethe.internal_map.resolution)
                        result = self.route_planner(f2c_cell)
                        self.grethe.internal_map.save_mask_polygon(grethe_surface, filename=f"grethe_interest{T}.png")
                    except Exception as e:
                        print(f"Error: {e}")
                if result is None:
                    self.grethe.pos[:] = [self.grethe.pos[0] + random.randint(-3, 3) * self.grethe.internal_map.resolution,
                                         self.grethe.pos[1] + random.randint(-3, 3) * self.grethe.internal_map.resolution]
                    print(f"marie plan failed, changing position {self.grethe.pos}")

            # if result is not None:
            # print("marie results non empty")
            m_path, m_start, m_end, polygon_NE_grethe = result
            grethe_cov_surface, covered_NE = self.polygon_world_to_grid(
                self.grethe.internal_map.shape,
                self.grethe.internal_map.origin,
                self.grethe.internal_map.resolution,
                polygon_NE_grethe
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

