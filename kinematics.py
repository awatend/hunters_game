import math
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

NED = Tuple[float, float, float]

def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class SimpleAUV:
    def __init__(
        self,
        pos: NED = (0.0, 0.0, 0.0),
        trajectory: List[NED] | None = None,
        speed_mps: float = 1.0,
        yaw_rate_max_rps: float = math.radians(15.0),
        wp_radius_m: float = 1.0,
        depth_rate_max_mps: float | None = None,   # set to e.g. 0.2 to limit; None snaps to target depth
        accel_max_mps2: float | None = None,        # set to e.g. 0.3 to limit; None means instant speed
    ):
        self.pos = pos
        self.yaw = 0.0
        self.v = 0.0
        self.trajectory = trajectory or []
        self.traj_index = 0

        self.speed_mps = speed_mps
        self.yaw_rate_max_rps = yaw_rate_max_rps
        self.wp_radius_m = wp_radius_m
        if depth_rate_max_mps is not None:
            self.depth_rate_max_mps = depth_rate_max_mps
        if accel_max_mps2 is not None:
            self.accel_max_mps2 = accel_max_mps2


    def follow_trajectory(self, dt: float = 1.0) -> NED:
        """Waypoint follower with simple AUV kinematics (speed + turn-rate + depth-rate limits)."""
        if not self.trajectory:
            return self.pos

        tN, tE, tD = self.trajectory[self.traj_index]
        N, E, D = self.pos

        # --- guidance: desired yaw to waypoint ---
        dx = tN - N
        dy = tE - E
        dist_xy = math.hypot(dx, dy)

        # waypoint acceptance radius (meters)
        wp_radius = getattr(self, "wp_radius_m", 1.0)

        # If we're close enough in XY, advance waypoint
        if dist_xy <= wp_radius:
            self.traj_index = min(self.traj_index + 1, len(self.trajectory) - 1)
            # keep moving smoothly; don't "snap" position
            return self.pos

        desired_yaw = math.atan2(dy, dx)  # NED: atan2(East, North) if using (dy, dx) with N,E
        # NOTE: since we used dy=tE-E and dx=tN-N, atan2(dy, dx) gives yaw where 0=North, +pi/2=East.

        # --- yaw kinematics: limit turn rate ---
        yaw_rate_max = getattr(self, "yaw_rate_max_rps", math.radians(15.0))  # rad/s
        yaw_err = wrap_pi(desired_yaw - getattr(self, "yaw", 0.0))
        yaw_step = clamp(yaw_err, -yaw_rate_max * dt, yaw_rate_max * dt)
        self.yaw = wrap_pi(getattr(self, "yaw", 0.0) + yaw_step)

        # --- speed (optional accel limit) ---
        v_cmd = getattr(self, "speed_mps", 1.0)
        if hasattr(self, "accel_max_mps2"):
            v = getattr(self, "v", v_cmd)
            amax = self.accel_max_mps2
            v = v + clamp(v_cmd - v, -amax * dt, amax * dt)
            self.v = max(0.0, v)
        else:
            self.v = v_cmd

        # --- integrate horizontal motion using yaw ---
        # N_dot = v*cos(yaw), E_dot = v*sin(yaw)
        N += self.v * math.cos(self.yaw) * dt
        E += self.v * math.sin(self.yaw) * dt

        # --- depth tracking: limit vertical rate (optional) ---
        depth_rate_max = getattr(self, "depth_rate_max_mps", None)
        if depth_rate_max is None:
            D = tD  # your original "snap depth" behavior
        else:
            # move depth toward target at limited rate
            d_err = tD - D
            D += clamp(d_err, -depth_rate_max * dt, depth_rate_max * dt)

        self.pos = (N, E, D)
        return self.pos


def animate_auv_3d(path_orig, occ_map=None, resolution_m=1.0, depth_m=10.0, step_per_frame=1):
    """
    Animate vehicle motion in 3D from a boustrophedon grid path.

    Parameters
    ----------
    path_orig : list[(i,j)]
        Output path from planner (original grid coords).
    occ_map : np.ndarray[H,W] bool or None
        If provided, can be used to set plot bounds (True=free).
    resolution_m : float
        Meters per grid cell.
    depth_m : float
        Constant depth used for animation (D positive down).
    step_per_frame : int
        How many path points to advance each animation frame (speed-up).
    """
    if not path_orig:
        raise ValueError("path_orig is empty")

    pts = np.asarray(path_orig, dtype=float)  # (i,j)
    # grid -> meters: N ~ i*res, E ~ j*res
    Ns = pts[:, 0] * resolution_m
    Es = pts[:, 1] * resolution_m
    Zs = np.full_like(Ns, -float(depth_m))  # plot Up = -Depth

    # bounds
    pad = 5.0 * resolution_m
    if occ_map is not None and np.any(occ_map):
        free = np.argwhere(occ_map.astype(bool))
        Nf = free[:, 0] * resolution_m
        Ef = free[:, 1] * resolution_m
        x_min, x_max = float(Nf.min()) - pad, float(Nf.max()) + pad
        y_min, y_max = float(Ef.min()) - pad, float(Ef.max()) + pad
    else:
        x_min, x_max = float(Ns.min()) - pad, float(Ns.max()) + pad
        y_min, y_max = float(Es.min()) - pad, float(Es.max()) + pad

    z_min, z_max = float(Zs.min()) - pad, float(max(0.0, Zs.max())) + pad

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Boustrophedon Path Animation (N, E, -D)")
    ax.set_xlabel("North (m)")
    ax.set_ylabel("East (m)")
    ax.set_zlabel("Up (m) = -Depth")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Optional: show free-space footprint as a faint plane of points (can be heavy if huge map)
    if occ_map is not None:
        free = np.argwhere(occ_map.astype(bool))
        if len(free) <= 20000:  # safety cap
            Nf = free[:, 0] * resolution_m
            Ef = free[:, 1] * resolution_m
            Zf = np.full_like(Nf, 0.0)  # draw at "surface" plane
            ax.scatter(Nf, Ef, Zf, s=1)

    (path_line,) = ax.plot([], [], [])
    (auv_pt,) = ax.plot([], [], [], marker="o")
    (head_line,) = ax.plot([], [], [])

    trailN, trailE, trailZ = [], [], []
    idx = {"k": 0}

    def init():
        path_line.set_data([], [])
        path_line.set_3d_properties([])
        auv_pt.set_data([], [])
        auv_pt.set_3d_properties([])
        head_line.set_data([], [])
        head_line.set_3d_properties([])
        return path_line, auv_pt, head_line

    def update(_frame):
        k = idx["k"]
        if k >= len(Ns):
            anim.event_source.stop()
            return path_line, auv_pt, head_line

        N, E, Z = float(Ns[k]), float(Es[k]), float(Zs[k])

        trailN.append(N)
        trailE.append(E)
        trailZ.append(Z)

        path_line.set_data(trailN, trailE)
        path_line.set_3d_properties(trailZ)

        auv_pt.set_data([N], [E])
        auv_pt.set_3d_properties([Z])

        # heading from current -> next (or previous if at end)
        k2 = min(k + 1, len(Ns) - 1)
        dN = float(Ns[k2] - Ns[k])
        dE = float(Es[k2] - Es[k])
        yaw = math.atan2(dE, dN) if (dN != 0.0 or dE != 0.0) else 0.0

        L = 2.0 * resolution_m
        hx = N + L * math.cos(yaw)
        hy = E + L * math.sin(yaw)
        head_line.set_data([N, hx], [E, hy])
        head_line.set_3d_properties([Z, Z])

        idx["k"] += int(step_per_frame)
        return path_line, auv_pt, head_line

    anim = FuncAnimation(fig, update, init_func=init, interval=30, blit=False)
    plt.show()

if __name__ == "__main__":
    animate_auv_3d()