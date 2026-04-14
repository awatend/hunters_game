from environment import Environment
from vehicles import AUV, ASV

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle

# Assumes you already have:
# - Environment class (with select_seed_ij, grow_surface_from_seed, mark_covered, world_to_grid, grid_to_world)
# - OR paste your Environment above this main.
#
# This script is self-contained for animation logic; it does not require your AUV/ASV classes.


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
    N = N0 + (ii + 0.5) * res
    E = E0 + (jj + 0.5) * res
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







def main():
    # -----------------------
    # Global geometry
    # -----------------------
    np.random.seed(0)
    shape = (220, 300)
    res = 2.0
    origin = (0.0, 0.0)

    # Each vehicle has its own internal map (separate instances)
    env_marie = Environment(shape=shape, resolution=res, origin=origin)
    env_thor = Environment(shape=shape, resolution=res, origin=origin)
    env_grethe = Environment(shape=shape, resolution=res, origin=origin)  # not used for coverage, but exists

    for env in (env_marie, env_thor, env_grethe):
        env.Initial_area[:, :] = True

    # Add same static obstacles to all internal maps (example)
    def add_static_obstacles(env):
        env.Collision_area[40:60, 60:110] = True
        env.Collision_area[120:160, 160:190] = True
        env.Collision_area[70:95, 220:260] = True

    for env in (env_marie, env_thor, env_grethe):
        add_static_obstacles(env)

    # -----------------------
    # Vehicles
    # -----------------------
    marie = AUV("Marie", pos=[60.0, 80.0, 0.0], safety_bubble_radius=6)  # AUV
    thor = AUV("Thor", pos=[80.0, 100.0, 0.0], safety_bubble_radius=6)  # AUV
    grethe = ASV("Grethe", pos=[120.0, 140.0, 0.0], safety_bubble_radius=8, comm_range_m=100.0)  # ASV support

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
    extent = grid_extent(env_marie)

    # background (light gray) + optional obstacle overlay (red) drawn as imshow
    base_cmap = ListedColormap([
        (0.95, 0.95, 0.95, 1.0),  # free background
        (0.85, 0.10, 0.10, 0.95)  # obstacles
    ])
    obstacle_img = env_marie.Collision_area.astype(int)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_title("Marie + Thor coverage, Grethe support (bubbles + comm range)")
    ax.set_xlabel("E [m]")
    ax.set_ylabel("N [m]")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.grid(True)

    ax.imshow(obstacle_img, origin="lower", extent=extent, cmap=base_cmap, vmin=0, vmax=1, interpolation="nearest")

    # Covered scatters (world points)
    marie_cov_sc = ax.scatter([], [], s=12, c="g", alpha=0.75, label="Marie covered")
    thor_cov_sc = ax.scatter([], [], s=12, c="c", alpha=0.75, label="Thor covered")

    # Vehicle markers
    marie_dot, = ax.plot([], [], "go", markersize=8, label="Marie")
    thor_dot, = ax.plot([], [], "co", markersize=8, label="Thor")
    grethe_dot, = ax.plot([], [], "mo", markersize=10, label="Grethe")

    # Safety bubbles (circles in meters)
    marie_bubble = Circle((0, 0), radius=marie.safety_bubble_radius * res, fill=False, linewidth=2)
    thor_bubble = Circle((0, 0), radius=thor.safety_bubble_radius * res, fill=False, linewidth=2)
    grethe_bubble = Circle((0, 0), radius=grethe.safety_bubble_radius * res, fill=False, linewidth=2)

    # Communication range circle for grethe
    grethe_comm = Circle((0, 0), radius=grethe.comm_range_m, fill=False, linestyle="--", linewidth=2)

    ax.add_patch(marie_bubble)
    ax.add_patch(thor_bubble)
    ax.add_patch(grethe_bubble)
    ax.add_patch(grethe_comm)

    ax.legend(loc="upper right")

    # -----------------------
    # Simulation parameters
    # -----------------------
    T = 100
    S_marie = 220
    S_thor = 180
    S_grethe = 150
    neighborhood = 8

    # Simple motion for grethe (support vessel): a slow drift
    def move_grethe(k):
        grethe.pos[0] += 3.0   # N
        grethe.pos[1] += 2.0   # E

        # keep inside map bounds
        Nmin, Emax = origin[0], origin[1] + shape[1] * res
        Nmax, Emin = origin[0] + shape[0] * res, origin[1]
        grethe.pos[0] = np.clip(grethe.pos[0], Nmin + 5.0, Nmax - 5.0)
        grethe.pos[1] = np.clip(grethe.pos[1], Emin + 5.0, Emax - 5.0)

    def step_grethe_coverage(grethe, S):
        env = grethe.internal_map

        # 1) Within_range is grethe comm disk (each AUV "sees" range around grethe)
        env.Within_range = disk_mask(env, center_NE=(grethe.pos[0], grethe.pos[1]), radius_m=grethe.comm_range_m)

        # 2) Dynamic collision: other vehicles' safety bubbles
        bubble_other = np.zeros(env.shape, dtype=bool)
        bubble_other |= grethe.set_safety_bubble( marie.pos, marie.safety_bubble_radius)
        bubble_other |= grethe.set_safety_bubble(thor.pos, thor.safety_bubble_radius)

        # Keep static obstacles too:

        # 3) Pick seed near current position (grid)
        start_ij = env.world_to_grid(grethe.pos[0], grethe.pos[1])
        seed = env.select_seed_ij(start_location=start_ij)

        # 4) Grow surface and mark covered
        surface_mask, idx = env.grow_surface_from_seed(
            seed_ij=seed,
            S=S,
            neighborhood=neighborhood,
            w_range=0,
            w_safe= 1.8,
            w_explore=1.0,
            w_exploit=0.1,
            min_safe_dist_m=4.0,
            w_compactness=0.1,
        )
        grethe.last_surface = surface_mask
        grethe.last_idx = idx

        env.mark_covered(surface_mask)

        # 5) Move AUV: teleport to idx (closest in range), else stay near seed
        if idx is not None:
            endN, endE = env.grid_to_world(int(idx[0]), int(idx[1]))
            grethe.pos[:] = [float(endN), float(endE)]
        else:
            endN, endE = env.grid_to_world(int(seed[0]), int(seed[1]))
            grethe.pos[:] = [float(endN), float(endE)]

        return surface_mask

    def step_thor_coverage(thor, S):
        env = thor.internal_map

        # 1) Within_range is grethe comm disk (each AUV "sees" range around grethe)
        env.Within_range = disk_mask(env, center_NE=(grethe.pos[0], grethe.pos[1]), radius_m=grethe.comm_range_m)

        # 2) Dynamic collision: other vehicles' safety bubbles
        bubble_other = np.zeros(env.shape, dtype=bool)
        bubble_other |= thor.set_safety_bubble( marie.pos, marie.safety_bubble_radius)
        bubble_other |= thor.set_safety_bubble(grethe.pos, grethe.safety_bubble_radius)

        # Keep static obstacles too:

        # 3) Pick seed near current position (grid)
        start_ij = env.world_to_grid(thor.pos[0], thor.pos[1])
        seed = env.select_seed_ij(start_location=start_ij)

        # 4) Grow surface and mark covered
        surface_mask, idx = env.grow_surface_from_seed(
            seed_ij=seed,
            S=S,
            neighborhood=neighborhood,
            w_range=.2,
            w_safe= 1.8,
            w_explore=0.5,
            w_exploit=0.1,
            min_safe_dist_m=4.0,
            w_compactness=1.0,
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

        return surface_mask

    def step_marie_coverage(vehicle, S):
        env = marie.internal_map

        # 1) Within_range is grethe comm disk (each AUV "sees" range around grethe)
        env.Within_range = disk_mask(env, center_NE=(grethe.pos[0], grethe.pos[1]), radius_m=grethe.comm_range_m)

        # 2) Dynamic collision: other vehicles' safety bubbles
        bubble_other = np.zeros(env.shape, dtype=bool)
        bubble_other |= marie.set_safety_bubble( grethe.pos, grethe.safety_bubble_radius)
        bubble_other |= marie.set_safety_bubble(thor.pos, thor.safety_bubble_radius)

        # Keep static obstacles too:

        # 3) Pick seed near current position (grid)
        start_ij = env.world_to_grid(marie.pos[0], marie.pos[1])
        seed = env.select_seed_ij(start_location=start_ij)

        # 4) Grow surface and mark covered
        surface_mask, idx = env.grow_surface_from_seed(
            seed_ij=seed,
            S=S,
            neighborhood=neighborhood,
            w_range=.2,
            w_safe= 1.8,
            w_explore=0.5,
            w_exploit=0.1,
            min_safe_dist_m=4.0,
            w_compactness=1.0,
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

        return surface_mask

    # -----------------------
    # Animation update
    # -----------------------
    def init():
        marie_dot.set_data([marie.pos[1]], [marie.pos[0]])
        thor_dot.set_data([thor.pos[1]], [thor.pos[0]])
        grethe_dot.set_data([grethe.pos[1]], [grethe.pos[0]])

        return (
            marie_cov_sc, marie_cov_sc,
            marie_dot, thor_dot, grethe_dot,
            marie_bubble, thor_bubble, grethe_bubble, grethe_comm
        )

    def update(frame):
        # Move support vessel
        move_grethe(frame)

        # AUVs perform coverage using their own internal maps
        surface_marie=step_marie_coverage(marie, S=S_marie)
        thor.internal_map.update_coverage(surface_marie)
        grethe.internal_map.update_coverage(surface_marie)
        surface_thor=step_thor_coverage(thor, S=S_thor)
        marie.internal_map.update_coverage(surface_thor) #Assume the vehicle communicate
        grethe.internal_map.update_coverage(surface_thor)
        surface_grethe = step_grethe_coverage(thor, S=S_grethe)

        # Update covered scatter (each from its own internal map)
        mN, mE = covered_grid_to_world_points(marie.internal_map, marie.internal_map.Covered_area)
        tN, tE = covered_grid_to_world_points(thor.internal_map, thor.internal_map.Covered_area)
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

        return (
            marie_cov_sc, thor_cov_sc,
            marie_dot, thor_dot, grethe_dot,
            marie_bubble, thor_bubble, grethe_bubble, grethe_comm
        )

    anim = FuncAnimation(fig, update, frames=T, init_func=init, interval=550, blit=False, repeat=False)
    plt.show()


if __name__ == "__main__":
    main()

