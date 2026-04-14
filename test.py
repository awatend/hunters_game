from environment import Environment
from vehicles import AUV, ASV


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
from skimage import measure
import fields2cover as f2c
import math

def grid_extent(env):
    H, W = env.shape
    res = env.resolution
    N0, E0 = env.origin
    return [E0, E0 + W * res, N0, N0 + H * res]


def covered_grid_to_world_points(env, covered_mask: np.ndarray):
    ii, jj = np.where(covered_mask)
    if ii.size == 0:
        return np.array([]), np.array([])
    N0, E0 = env.origin
    res = env.resolution
    N = N0 + (ii+0.5) * res
    E = E0 + (jj+0.5) * res
    return N, E


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
    tolerance = 2.0
    simplified_contour = measure.approximate_polygon(main_contour, tolerance=tolerance)

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

    # cELL DECOMPOSITION STEP
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

    # Print some results
    print(f"Total path length: {final_path.length():.2f} meters {final_path.size()} ")
    f2c.Visualizer.figure()
    f2c.Visualizer.plot(cell)
    f2c.Visualizer.plot(final_path)
    f2c.Visualizer.save(f"coverage_plan_{final_path.length():.2f}.png")




def main():
    # -----------------------
    # Global geometry
    # -----------------------
    #np.random.seed(0)
    shape = (
        130, 220)
    res = 20.0
    origin = (0.0, 0.0)

    # Global parameters
    # -----------------------
    T = 10
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
    ax.imshow(
        obstacle_img,
        origin="lower",
        extent=extent,
        cmap="Reds",
        alpha=0.6,  # very transparent
    )

    # Covered scatters (world points)
    marie_cov_sc = ax.scatter([], [], s=12, c="g", alpha=0.9, label="Marie covered")
    thor_cov_sc = ax.scatter([], [], s=12, c="c", alpha=0.9, label="Thor covered")

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
        return surface_mask,f2c_cell

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

        env.mark_covered(surface_mask)

        # 5) Move AUV: teleport to idx (closest in range), else stay near seed
        if idx is not None:
            endN, endE = env.grid_to_world(int(idx[0]), int(idx[1]))
            thor.pos[:] = [float(endN), float(endE)]
        else:
            endN, endE = env.grid_to_world(int(seed[0]), int(seed[1]))
            thor.pos[:] = [float(endN), float(endE)]

        f2c_cell = None
        try:
            f2c_cell = mask_to_f2c_cell(surface_mask, scale=env.resolution)
            print(f"Cell Area: {f2c_cell.area():.2f} m²")

        except Exception as e:
            print(f"Error: {e}")
        return surface_mask, f2c_cell

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

        f2c_cell = None
        try:
            f2c_cell = mask_to_f2c_cell(surface_mask, scale=env.resolution)
            print(f"Cell Area: {f2c_cell.area():.2f} m²")

        except Exception as e:
            print(f"Error: {e}")
        return surface_mask, f2c_cell
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
            marie_cov_sc, thor_cov_sc,
            marie_dot, thor_dot, grethe_dot,
            marie_bubble, thor_bubble, grethe_bubble, grethe_comm,
            cov_line, marie_line, thor_line,
            gm_dist_line, gt_dist_line
        )


    def update(frame):
        avoid_thor=thor.set_safety_bubble(thor.pos,thor.safety_bubble_radius)
        avoid_marie=marie.set_safety_bubble(marie.pos,marie.safety_bubble_radius)
        avoid_grethe=grethe.set_safety_bubble(grethe.pos,grethe.safety_bubble_radius)
        # AUVs perform coverage using their own internal maps
        marie_surface, marie_cell=update_marie_plan(marie, S_marie, avoid_thor, avoid_grethe)
        thor.internal_map.update_coverage(marie_surface)
        grethe.internal_map.update_coverage(marie_surface)

        thor_surface, thor_cell=update_thor_plan(thor, S_thor, avoid_marie, avoid_grethe)
        marie.internal_map.update_coverage(thor_surface) #Assume the vehicle communicate
        grethe.internal_map.update_coverage(thor_surface)

        grethe_surface, grethe_cell = update_grethe_plan(grethe, S_grethe, avoid_thor,avoid_marie)

        route_planner(marie_cell)
        route_planner(thor_cell)
        #route_planner(grethe_cell)

        # Update covered scatter (each from its own internal map)
        mN, mE = covered_grid_to_world_points(env_marie,marie.internal_map.Covered_area)
        tN, tE = covered_grid_to_world_points(env_thor,thor.internal_map.Covered_area)
        marie_cov_sc.set_offsets(np.column_stack([mE, mN]) if mN.size else np.empty((0, 2)))
        thor_cov_sc.set_offsets(np.column_stack([tE, tN]) if tN.size else np.empty((0, 2)))

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



        return (
            marie_cov_sc, thor_cov_sc,
            marie_dot, thor_dot, grethe_dot,
            marie_bubble, thor_bubble, grethe_bubble, grethe_comm,
            cov_line, marie_line, thor_line,
            gm_dist_line, gt_dist_line
        )

    anim = FuncAnimation(fig, update, frames=T, init_func=init, interval=2000, blit=False, repeat=False)
    plt.show()


if __name__ == "__main__":
    main()

