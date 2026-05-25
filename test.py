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

def grid_extent(env):
    H, W = env.shape
    res = env.resolution
    N0, E0 = env.origin
    return [E0, E0 + W * res, N0, N0 + H * res]


def closest_point(pos, start, end):
    d_start = (start.getY()- pos[0])**2 + (start.getX() - pos[1])**2
    d_end   = (end.getY()   - pos[0])**2 + (end.getX()   - pos[1])**2

    return start if d_start < d_end else end

def covered_grid_to_world_points(env, covered_mask: np.ndarray):
    ii, jj = np.where(covered_mask)
    if ii.size == 0:
        return np.array([]), np.array([])
    N0, E0 = env.origin
    res = env.resolution
    N = N0 + (ii+0.5) * res
    E = E0 + (jj+0.5) * res
    return N, E

def polygon_world_to_grid(shape, origin,resolution, polygon_NE):
    H, W = shape
    N0, E0 = origin
    res = resolution

    contour = polygon_NE.getExteriorRing()
    coordinates=[]
    for p in range(contour.size()):
        coordinates.append ((contour.getGeometry(p).getY(), contour.getGeometry(p).getX()))   # (N,E) ← (y,x)

    coords = np.array(coordinates)
    ii = (coords[:,0] - N0) / res
    jj = (coords[:, 1] - E0) / res

    ii = np.clip(ii, 0, H - 1)
    jj = np.clip(jj, 0, W - 1)

    # Rasterize polygon
    rr, cc = polygon(ii, jj, shape=(H, W))
    mask = np.zeros((H, W), dtype=bool)
    mask[rr, cc] = True
    #print(f"testing function: contours{contour} polygon{coords}")

    return mask,coords


def disk_mask(env, center_NE, radius_m):
    """Boolean disk mask in grid coords."""
    H, W = env.shape
    N0, E0 = env.origin
    res = env.resolution
    cN, cE = center_NE
    ci = (cN - N0) / res
    cj = (cE - E0) / res
    ii, jj = np.indices((H, W))
    r_cells = float(radius_m / res)
    return ((ii - ci) ** 2 + (jj - cj) ** 2) <= (r_cells ** 2)


def mask_to_f2c_cell(mask, scale=1.0):
    """
    Converts a boolean mask to a Fields2Cover Cell.
    :param mask: numpy array (dtype=bool)
    :param scale: meters per pixel (important for kinematics!)
    :return: f2c.F2CCell
    """
    # 1. Extract contours as ordered coordinates [y, x]
    # Fully vectorized and handles complex shapes
    contours = measure.find_contours(mask, 0.5)

    if not contours:
        raise ValueError("No polygon found in the mask.")

    # 2. Pick the largest contour (the field boundary)
    main_contour = max(contours, key=len)
    #simplify contour to reduce points (tolerance in pixels)
    tolerance = 1.5
    simplified_contour = measure.approximate_polygon(main_contour, tolerance=tolerance)
    if len(simplified_contour) < 4:
        # Option A: Revert to the original contour to keep the shape
        simplified_contour = main_contour
    # 3. Build the F2C Geometry
    ring = f2c.LinearRing()

    for pt in simplified_contour:
        # pt is [row, col]. We map: col -> x, row -> y
        # We apply scale to convert pixels to meters
        y_m = float(pt[0]) * scale
        x_m = float(pt[1]) * scale
        ring.addPoint(x_m, y_m)

    # 4. Explicitly close the ring (F2C requirement)

    start_pt = simplified_contour[0]
    ring.addPoint(float(start_pt[1]) * scale, float(start_pt[0]) * scale)

    # 5. Create the Cell
    cell = f2c.Cell()
    cell.addRing(ring)
    print (f"Extracted contour with {len(main_contour)} points, simplified to {len(simplified_contour)}")

    return cell

def route_planner(cell):
    robot = f2c.Robot(2.0, 2.6)  # Width: 2.0m, Operating width: 2.6m
    robot.setMinTurningRadius(2.5)  # 1.5m minimum turn radius
    robot.setMaxCurv(0.5)  # 1.5m minimum turn radius
    robot.setTurnVel(1.0)
    robotsensorSwath = 5 # Sensor swath width: 5.0m


    # 1. Create a Cells container (notice the plural 's')
    cells_container = f2c.Cells()

    # 2. Add your existing cell to that container
    cells_container.addGeometry(cell)

    # CELL DECOMPOSITION STEP
    #decomp = f2c.DECOMP_TrapezoidalDecomp()
    # Split vertically (0.5 * pi). This creates simple trapezoids.
    #decomp.setSplitAngle(0.5 * math.pi)
    #decomposed_cells = decomp.decompose(cells_container)

    # 3. Now generate the headland
    const_hl = f2c.HG_Const_gen()
    # Note: Use the container as the argument
    headland = const_hl.generateHeadlands(cells_container, 5.0) # or cell container  directly
    mid_headland= const_hl.generateHeadlands(cells_container, 2.5) # to avoid overshooting planned path
    if headland.size() == 0:
        print("Skipping cell: Cell is too small to have a headland.")
        return

    #4. Generate Swaths inside the field (excluding headlands)
    # SG_BruteForce finds the best angle to minimize turns or distance
    swath_gen = f2c.SG_BruteForce()
    #swaths = swath_gen.generateBestSwaths(f2c.OBJ_NSwath(), robotsensorSwath, headland.getGeometry(0)) #for snake pattern
    swaths = swath_gen.generateSwaths(math.pi*0.7, robotsensorSwath, headland)


    f_swaths = f2c.Swaths()

    for i in range(swaths.at(0).size()):

        swath = swaths.getSwath(i)
        # Check length in meters


        if swath.length() >= 10.0:  # Only keep swaths longer than 5 meters
            f_swaths.push_back(swath)
    filtered_swaths = f2c.SwathsByCells()
    filtered_swaths.push_back(f_swaths)
    # 3. Sort Swaths into a Route (Snake pattern)
    #careful because it can lead the robot into obstacles,
    #r_planner = f2c.RP_Snake()
    #route = r_planner.genSortedSwaths(swaths)

    if filtered_swaths.size() == 0 or filtered_swaths.at(0).size() == 0:
        print("No valid swaths generated for this cell. Skipping route planning.")
        return
    r_planner = f2c.RP_RoutePlannerBase()
    route = r_planner.genRoute(mid_headland,filtered_swaths) #
    #print(f"Generated route with {swaths.at(0).size()} segments.")

    # 4.final Path using Dubins Kinematics
    path_planner = f2c.PP_PathPlanning()
    dubins = f2c.PP_DubinsCurves()
    final_path = path_planner.planPath(robot, route, dubins)
    #final_path.reduce(1.5) # Reduce points to simplify the path (tolerance in meters)

    # Print some results
    print(f"Total path length: {final_path.length():.2f} meters {final_path.size()} ")
    f2c.Visualizer.figure()
    #f2c.Visualizer.plot(cell)
    #f2c.Visualizer.save(f"coverage_plan_{final_path.length():.2f}.png")

    f2c.Visualizer.figure()
    f2c.Visualizer.plot(cell)
    f2c.Visualizer.plot(final_path)
    f2c.Visualizer.save(f"main_contour{final_path.length():.2f}.png")

    if final_path.size() < 0:
        return
    else:
        combined_cells = f2c.Cells()  # Use F2CCells prefix for safety
        buffer_dist = robot.getWidth() / 2.0

        for swaths_section in filtered_swaths:
            for i in range(swaths_section.size()):
                swath = swaths_section.at(i)

                # Convert Swath (Line) to Cells (Polygon)
                swath_poly = swath.areaCovered()#.buffer(buffer_dist)

                # Now you can append a Cell/Cells to Cells
                combined_cells = combined_cells.unionOp(swath_poly)

        # Correct Visualization logic
        f2c.Visualizer.figure()
        f2c.Visualizer.plot(combined_cells)  # Plot the buffered swath area
        f2c.Visualizer.save(f"coverage_map{final_path.size():.2f}.png")
        #print(f"Combined coverage area: {combined_cells.area()} m2")
        simplified_cells = combined_cells.simplify(0.1)

        # 3. Visualization
        f2c.Visualizer.figure()
        f2c.Visualizer.plot(simplified_cells.convexHull())  # Plot simplified coverage
        f2c.Visualizer.save(f"simplified_cell_{final_path.size():.2f}.png")

        # 4. Print results
        #print(f"Combined coverage area: {simplified_cells.area():.2f} m2")
        return final_path, final_path.atStart(), final_path.atEnd() ,simplified_cells.convexHull()

def is_treasure_founded(mask, treasure_location ):
    if treasure_location is None:
        return False
    i, j = treasure_location
    return mask[i, j]

def is_treasure_foundeds(mask, treasure_location, resolution, threshold_m=5.0):
    if treasure_location is None:
        return False

    ti, tj = int(treasure_location[0]), int(treasure_location[1])

    # Distance (in cells) to nearest True in mask
    dist_cells = distance_transform_edt(~mask)

    # Convert to meters
    dist_m = dist_cells[ti, tj] * resolution

    return dist_m < threshold_m

def main():
    # -----------------------
    # Global geometry
    # -----------------------
    #np.random.seed(0)
    shape = (
        130, 220)
    res = 5.0
    origin = (0.0, 0.0)




    # Global parameters
    # -----------------------
    T = 100
    S_marie = 80
    S_thor = 80
    S_grethe = 20
    neighborhood = 8


    # Each vehicle has its own internal map (separate instances)
    env_marie = Environment(shape=shape, resolution=res, origin=origin)
    env_thor = Environment(shape=shape, resolution=res, origin=origin)
    env_grethe = Environment(shape=shape, resolution=res, origin=origin)  # not used for coverage, but exists
    env_grethe.update()
    for env in (env_marie, env_thor, env_grethe):
        env.Initial_area[:, :] = True

    # Add same static obstacles to all internal maps (example)
    def add_static_obstacles(env):
       env.Collision_area[40:60, 60:110] = True
       env.Collision_area[120:160, 160:190] = True
       env.Collision_area[70:95, 220:260] = True
        #pass

    for env in (env_marie, env_thor, env_grethe):
        add_static_obstacles(env)
        env.data=env_grethe.data

    # -----------------------
    # Vehicles
    # -----------------------
    marie = AUV("Marie", pos=[0.0, 0.0, 0.0], safety_bubble_radius=1, comm_range_m=1)  # AUV
    thor = AUV("Thor", pos=[5.0, 2.0, 0.0], safety_bubble_radius=1, comm_range_m=2)  # AUV
    grethe = ASV("Grethe", pos=[2.0, 5.0, 0.0], safety_bubble_radius=2, comm_range_m=5)  # ASV support

    marie.set_internal_map(env_marie)
    thor.set_internal_map(env_thor)
    grethe.set_internal_map(env_grethe)

    #Define treasure location
    coords = np.argwhere(env_marie.Initial_area & ~env_marie.Collision_area)
    treasure_location = tuple(coords[np.random.randint(len(coords))])
    print(f"Treasuuuuuuuuuuuuuuure is at grid location: {treasure_location}")
    # Mark initial covered on each AUV map
    for auv in (marie, thor):
        i, j = auv.internal_map.world_to_grid(auv.pos[0], auv.pos[1])
        auv.internal_map.Covered_area[i, j] = True

    # -----------------------
    # Plot setup
    # -----------------------
    fig, (ax, ax_metrics, ax_dist) = plt.subplots(
        1, 3, figsize=(20, 8), gridspec_kw={"width_ratios": [2.2, 1, 1]}
    )

    extent = grid_extent(env_marie)

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
    ax_metrics.set_xlim(0, T)
    ax_metrics.grid(True)

    # --- Distance subplot ---
    ax_dist.set_title("Distances to Grethe")
    ax_dist.set_xlabel("Iteration")
    ax_dist.set_ylabel("Distance [m]")
    ax_dist.set_xlim(0, T)
    ax_dist.axhline(grethe.comm_range_m, linestyle="--", linewidth=1.5, color="gray", label="Comm range")
    ax_dist.legend(loc="upper left")
    #ax_dist.set_ylim(0, 600)
    ax_dist.grid(True)

    # Histories
    frames_hist = []
    coverage_hist = []
    marie_hist = []
    thor_hist = []
    grethe_marie_dist_hist = []
    grethe_thor_dist_hist = []
    marie_cov_pl=[]
    thor_cov_pl=[]

    # Coverage
    cov_line, = ax_metrics.plot([], [], linewidth=2, label="Total coverage ratio")
    marie_line, = ax_metrics.plot([], [], "g-", linewidth=2, label="Marie contribution")
    thor_line, = ax_metrics.plot([], [], "c-", linewidth=2, label="Thor contribution")
    ax_metrics.legend(loc="upper left")

    # Distance lines
    gm_dist_line, = ax_dist.plot([], [], "m-", linewidth=2, label="Grethe-Marie distance")
    gt_dist_line, = ax_dist.plot([], [], "b-", linewidth=2, label="Grethe-Thor distance")
    ax_dist.legend(loc="upper left")

    # Overlay
    heat = env_marie.data
    obstacle_img = env_marie.Collision_area.astype(float)

    ax.imshow(
        heat,
        origin="lower",
        extent=extent,
        cmap=heat_cmap,
        vmin=0,
        vmax=1,
        alpha=1.0,  # transparency so obstacles remain visible
        interpolation="bilinear"
    )
    if treasure_location is not None:
        treasure_N, treasure_E = env_marie.grid_to_world(treasure_location[0], treasure_location[1])
        ax.plot(treasure_E, treasure_N, "y*", markersize=15, label="Treasure")
        ax.legend(loc="upper right")

    ax.imshow(
        obstacle_img,
        origin="lower",
        extent=extent,
        cmap="Reds",
        alpha=0.6,  # very transparent
    )

    # Covered polygons (world points)
    for poly in marie_cov_pl:
        if len(poly) == 0:
            continue

        N = [p[0] for p in poly]
        E = [p[1] for p in poly]
        #print(f"polygon to draw {poly}")

        ax.fill(E, N, color='green', alpha=0.3, label="Marie coverage")

    for poly in thor_cov_pl:
        if len(poly) == 0:
            continue

        N = [p[0] for p in poly]
        E = [p[1] for p in poly]

        ax.fill(E, N, color='green', alpha=0.3, label = "Thor coverage")

    # Vehicle markers
    marie_dot, = ax.plot([], [], "go", markersize=8, label="Marie")
    thor_dot, = ax.plot([], [], "co", markersize=8, label="Thor")
    grethe_dot, = ax.plot([], [], "mo", markersize=10, label="Grethe")

    # Safety bubbles (circles in meters)
    marie_bubble = Circle((0, 0), radius=5 * res, fill=False, linewidth=2)
    thor_bubble = Circle((0, 0), radius=5 * res, fill=False, linewidth=2)
    grethe_bubble = Circle((0, 0), radius=5 * res, fill=False, linewidth=2)

    # Communication range circle for grethe
    grethe_comm = Circle((0, 0), radius=grethe.comm_range_m, fill=False, linestyle="--", linewidth=2)

    ax.add_patch(marie_bubble)
    ax.add_patch(thor_bubble)
    ax.add_patch(grethe_bubble)
    ax.add_patch(grethe_comm)

    ax.legend(loc="upper right")

    fig.tight_layout()
    # -----------------------
    # Simulation parameters
    # -----------------------



    def update_marie_plan(marie, S, surface_thor, surface_grethe):
        print("update_marie_plan")
        env = marie.internal_map

        # 1) Within_range is grethe comm disk (each AUV "sees" range around grethe)
        env.Within_range = marie.set_safety_bubble(grethe.pos, grethe.comm_range_m)

        # 2) Dynamic collision: other vehicles' safety bubbles
        bubble_other = np.zeros(env.shape, dtype=bool)
        bubble_other |= surface_thor
        bubble_other |= surface_grethe

        # Add dynamic collision: other vehicles' safety bubbles
        marie.internal_map.dynamic_Collision_area = bubble_other
        marie.internal_map.set_collision_area()
        # 3) Pick seed near current position (grid)
        start_ij = env.world_to_grid(marie.pos[0], marie.pos[1])
        seed = env.select_seed_ij(start_location=start_ij, w_range=.5,
            w_safe= 0.0,
            w_explore=0.9,
            w_exploit=0.5,)
        if seed is None:
            return
        # 4) Grow surface and mark covered
        surface_mask, idx = env.grow_surface_from_seed(
            seed_ij=seed,
            S=S,
            neighborhood=neighborhood,
            w_range=.1,
            w_safe= 0.5,
            w_explore=1.0,
            w_exploit=0.5,
            min_safe_dist_m=6.0

        )
        '''
        marie.last_surface = surface_mask
        marie.last_idx = idx

        env.mark_covered(surface_mask)

        # 5) Move AUV: teleport to idx (closest in range), else stay near seed
        if idx is not None:
            endN, endE = env.grid_to_world(int(idx[0]), int(idx[1]))
            marie.pos[:] = [float(endN), float(endE)]
        else:
            endN, endE = env.grid_to_world(int(seed[0]), int(seed[1]))
            marie.pos[:] = [float(endN), float(endE)]

        f2c_cell = None
        try:
            f2c_cell = mask_to_f2c_cell(surface_mask, scale=env.resolution)
            print(f"Cell Area: {f2c_cell.area():.2f} m²")

        except Exception as e:
            print(f"Error: {e}")
            '''
        return surface_mask

    def update_thor_plan(thor, S, surface_marie, surface_grethe):
        env = thor.internal_map

        # 1) Within_range is grethe comm disk (each AUV "sees" range around grethe)
        env.Within_range = thor.set_safety_bubble(grethe.pos, grethe.comm_range_m)

        # 2) Dynamic collision: other vehicles' safety bubbles
        bubble_other = np.zeros(env.shape, dtype=bool)
        bubble_other |= surface_marie
        bubble_other |= surface_grethe

        # Add dynamic collision: other vehicles' safety bubbles
        thor.internal_map.dynamic_Collision_area = bubble_other
        thor.internal_map.set_collision_area()
        # 3) Pick seed near current position (grid)
        start_ij = thor.internal_map.world_to_grid(thor.pos[0], thor.pos[1])
        seed = env.select_seed_ij(start_location=start_ij, w_range=.5,
            w_safe= 0.9,
            w_explore=0.9,
            w_exploit=0.5,)

        if seed is None:
            return
        # 4) Grow surface and mark covered
        surface_mask, idx = env.grow_surface_from_seed(
            seed_ij=seed,
            S=S,
            neighborhood=neighborhood,
            w_range=.2,
            w_safe= 1.0,
            w_explore=1.0,
            w_exploit=0.5,
            min_safe_dist_m=6.0
        )
        thor.last_surface = surface_mask
        thor.last_idx = idx


        return surface_mask

    def update_grethe_plan(grethe, S, surface_marie, surface_thor):
        env = grethe.internal_map

        # 1) Within_range is grethe comm disk (each AUV "sees" range around grethe)
        #env.Within_range = disk_mask(env, center_NE=(thor.pos[0], thor.pos[1]), radius_m=thor.comm_range_m)
        #env.Within_range |= disk_mask(env, center_NE=(marie.pos[0], marie.pos[1]), radius_m=marie.comm_range_m)

        env.Within_range = grethe.set_safety_bubble(marie.pos, marie.comm_range_m)
        env.Within_range |= grethe.set_safety_bubble(thor.pos, thor.comm_range_m)

        # 2) Dynamic collision: other vehicles' safety bubbles
        bubble_other = np.zeros(env.shape, dtype=bool)
        bubble_other |= surface_marie
        bubble_other |= surface_thor

        # Add dynamic collision: other vehicles' safety bubbles
        grethe.internal_map.dynamic_Collision_area = bubble_other
        grethe.internal_map.set_collision_area()
        # 3) Pick seed near current position (grid)
        start_ij = env.world_to_grid(grethe.pos[0], grethe.pos[1])
        seed = env.select_seed_ij(start_location=start_ij, w_range=1.0,
            w_safe= 0.0,
            w_explore=0.2,
            w_exploit=0.01,)

        if seed is None:
            return
        # 4) Grow surface and mark covered
        surface_mask, idx = env.grow_surface_from_seed(
            seed_ij=seed,
            S=S,
            neighborhood=neighborhood,
            w_range=1.0,
            w_safe= 1.0,
            w_explore=0.01,
            w_exploit=0.09,
            min_safe_dist_m=6.0
        )
        grethe.last_surface = surface_mask
        grethe.last_idx = idx


        # 5) Move AUV: teleport to idx (closest in range), else stay near seed
        if idx is not None:
            endN, endE = env.grid_to_world(int(idx[0]), int(idx[1]))
            grethe.pos[:] = [float(endN), float(endE)]
        else:
            endN, endE = env.grid_to_world(int(seed[0]), int(seed[1]))
            grethe.pos[:] = [float(endN), float(endE)]
        return surface_mask
    # -----------------------
    # Animation update
    # -----------------------
    def init():
        marie_dot.set_data([marie.pos[1]], [marie.pos[0]])
        thor_dot.set_data([thor.pos[1]], [thor.pos[0]])
        grethe_dot.set_data([grethe.pos[1]], [grethe.pos[0]])

        cov_line.set_data([], [])
        marie_line.set_data([], [])
        thor_line.set_data([], [])
        gm_dist_line.set_data([], [])
        gt_dist_line.set_data([], [])


        return (
            marie_cov_pl, thor_cov_pl,
            marie_dot, thor_dot, grethe_dot,
            marie_bubble, thor_bubble, grethe_bubble, grethe_comm,
            cov_line, marie_line, thor_line,
            gm_dist_line, gt_dist_line
        )


    def update(frame):
        print("start update")
        avoid_thor=thor.set_safety_bubble(thor.pos,thor.safety_bubble_radius)
        avoid_marie=marie.set_safety_bubble(marie.pos,marie.safety_bubble_radius)
        avoid_grethe=grethe.set_safety_bubble(grethe.pos,grethe.safety_bubble_radius)

        # AUVs perform coverage using their own internal maps
        marie_surface=update_marie_plan(marie, S_marie, avoid_thor, avoid_grethe)
        f2c_cell = None
        try:
            f2c_cell = mask_to_f2c_cell(marie_surface, scale=marie.internal_map.resolution)
            #print(f"Cell Area: {f2c_cell.area():.2f} m²")

        except Exception as e:
            print(f"Error: {e}")

        if f2c_cell is not None:
            print("Planning for marie")
            result = route_planner(f2c_cell)
            if result is not None:
                m_path, m_start, m_end, polygon_NE_marie=result
                #print(f"polygon: {polygon_NE_marie}")
                #print(type(polygon_NE_marie))
                marie_cov_surface, covered_NE_marie=polygon_world_to_grid(marie.internal_map.shape,marie.internal_map.origin,marie.internal_map.resolution, polygon_NE_marie)
                marie.internal_map.mark_covered(marie_cov_surface)
                marie.last_surface = marie_cov_surface
                if is_treasure_founded(marie_cov_surface, treasure_location):
                    print("Treasure Founded by Marie!")
                    return

                if closest_point(marie.pos,m_start,m_end)== m_start:
                    marie.pos[0] = float(m_end.getY())
                    marie.pos[1] = float(m_end.getX())
                else:
                    marie.pos[0] = float(m_start.getY())
                    marie.pos[1] = float(m_start.getX())

                thor.internal_map.update_coverage(marie_cov_surface)
                grethe.internal_map.update_coverage(marie_cov_surface)

                marie_cov_pl.append(covered_NE_marie)
                print(f"polygon appended{covered_NE_marie}")

        thor_surface=update_thor_plan(thor, S_thor, avoid_marie, avoid_grethe)
        print("planning for thor")
        f2c_cell = None
        try:
            f2c_cell = mask_to_f2c_cell(thor_surface, scale=thor.internal_map.resolution)
            #print(f"Cell Area: {f2c_cell.area():.2f} m²")

        except Exception as e:
            print(f"Error: {e}")
        if f2c_cell is not None:
            result =route_planner(f2c_cell)
            if result is not None:
                t_path, t_start, t_end, polygon_NE_thor = result
                thor_cov_surface,thor_covered_NE=polygon_world_to_grid(thor.internal_map.shape,thor.internal_map.origin,thor.internal_map.resolution, polygon_NE_thor)
                thor.internal_map.mark_covered(thor_cov_surface)
                thor.last_surface = thor_cov_surface
                if is_treasure_founded(thor_cov_surface, treasure_location):
                    print("Treasure Founded by Thor!")
                    return

                if closest_point(thor.pos,t_start,t_end) == t_start:
                    thor.pos[0] = float(t_end.getY())
                    thor.pos[1] = float(t_end.getX())
                else:
                    thor.pos[0] = float (t_start.getY())
                    thor.pos[1] = float(t_start.getX())

                marie.internal_map.update_coverage(thor_cov_surface) #Assume the vehicle communicate
                grethe.internal_map.update_coverage(thor_cov_surface)
                thor_cov_pl.append(thor_covered_NE)

            grethe_surface = update_grethe_plan(grethe, S_grethe, avoid_thor,avoid_marie)


            # Update covered scatter (each from its own internal map)


        # Update vehicle dots (x=E, y=N)
        marie_dot.set_data([marie.pos[1]], [marie.pos[0]])
        thor_dot.set_data([thor.pos[1]], [thor.pos[0]])
        grethe_dot.set_data([grethe.pos[1]], [grethe.pos[0]])

        # Update bubbles (circle centers are (E,N))
        marie_bubble.center = (marie.pos[1], marie.pos[0])
        thor_bubble.center = (thor.pos[1], thor.pos[0])
        grethe_bubble.center = (grethe.pos[1], grethe.pos[0])
        grethe_comm.center = (grethe.pos[1], grethe.pos[0])

        # (optional) show seeds in title
        ax.set_title(
            f"Iter {frame+1}/{T} | "
            f"Marie covered={int(marie.internal_map.Covered_area.sum())} | "
            f"Thor covered={int(thor.internal_map.Covered_area.sum())}"
        )

        # ---- Metrics ----
        available_area = marie.internal_map.available_area()  # Should be same for both if they have same static map
        total_ratio = float(
                marie.internal_map.total_Covered_area.sum()
                / available_area.sum()
        )

        marie_ratio = float(
                marie.internal_map.Covered_area.sum()
                / available_area.sum()
        )

        thor_ratio = float(
                thor.internal_map.Covered_area.sum()
                / available_area.sum()
        )

        # Distances to Grethe
        grethe_marie_dist = float(np.hypot(
            grethe.pos[0] - marie.pos[0],
            grethe.pos[1] - marie.pos[1]
        ))

        grethe_thor_dist = float(np.hypot(
            grethe.pos[0] - thor.pos[0],
            grethe.pos[1] - thor.pos[1]
        ))

        frames_hist.append(frame + 1)
        coverage_hist.append(total_ratio)
        marie_hist.append(marie_ratio)
        thor_hist.append(thor_ratio)
        grethe_marie_dist_hist.append(grethe_marie_dist)
        grethe_thor_dist_hist.append(grethe_thor_dist)

        cov_line.set_data(frames_hist, coverage_hist)
        marie_line.set_data(frames_hist, marie_hist)
        thor_line.set_data(frames_hist, thor_hist)
        gm_dist_line.set_data(frames_hist, grethe_marie_dist_hist)
        gt_dist_line.set_data(frames_hist, grethe_thor_dist_hist)

        max_dist = max(
            max(grethe_marie_dist_hist, default=1.0),
            max(grethe_thor_dist_hist, default=1.0)
        )
        ax_dist.set_ylim(0, max_dist * 1.1)

        for poly in marie_cov_pl:
            if len(poly) == 0:
                continue

            Nm = [p[0] for p in poly]
            Em = [p[1] for p in poly]

            patch = ax.fill(Em, Nm, color='green', alpha=1.0)[0]

        for poly in thor_cov_pl:
            if len(poly) == 0:
                continue

            Nt = poly[:,0]
            Et = poly[:,1]
            #print(f"polygon to draw {poly}")

            patch = ax.fill(Et, Nt, color='blue', alpha=1.0)[0]
            #print("DONNNE")

        return (
            marie_cov_pl, thor_cov_pl,
            marie_dot, thor_dot, grethe_dot,
            marie_bubble, thor_bubble, grethe_bubble, grethe_comm,
            cov_line, marie_line, thor_line,
            gm_dist_line, gt_dist_line
        )

    anim = FuncAnimation(fig, update, frames=T, init_func=init, interval=500, blit=False, repeat=False)
    plt.show()


if __name__ == "__main__":
    main()
    print("Game over")

