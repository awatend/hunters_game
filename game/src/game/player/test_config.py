"""
Parameters and execution script for the Hunters Game simulation.

TO SWITCH CELL SHAPES: change only the CELL_GEOMETRY line.
"""

from game.player.test_class import HuntersGameSimulation
from game.internal_map.cell_geometry import RectangularGeometry
from game.player.load_prior import SinmodPrior


# ---------------------------------------------------------------------------
# Load SINMOD prior once at module level so both CELL_GEOMETRY and the
# field arrays (temperature, biomass) share the same object.
# ---------------------------------------------------------------------------
_SINMOD_PRIOR = SinmodPrior(
    # filepath  = ...,          # override if archive lives elsewhere
    x_min   = 500_000.0,
    x_max   = 509_000.0,
    y_min   = 7_060_000.0,
    y_max   = 7_065_000.0,
    spacing = 100.0,
    utm_zone = 32,
)


class SimulationParameters:
    """Container for all simulation parameters."""

    # =========================================================================
    # CELL GEOMETRY — change this single line to switch between shapes.
    # All grid parameters (shape, spacing, origin) live inside the geometry.
    # =========================================================================

    # Geometry derived from the real SINMOD domain (hexagonal, 50 m spacing):
    #CELL_GEOMETRY = _SINMOD_PRIOR.geometry
    #ORIGIN=_SINMOD_PRIOR.geometry.origin
    #PRIOR_DATA = _SINMOD_PRIOR._biomass

    # ── Alternative geometries (uncomment one to switch) ─────────────────────
        #Rectangular grid
    CELL_GEOMETRY = RectangularGeometry(
         shape=(100, 180),
         resolution=20.0,
         origin=(0.0, 0.0),
         connectivity=8,
     )
    PRIOR_DATA = None  # no prior data for rectangular grid
    # Hand-crafted hexagonal grid:
    # CELL_GEOMETRY = HexagonalGeometry(
    #     shape=(116, 181),
    #     spacing=20.0,
    #     origin=(0.0, 0.0),
    # )

    # Simulation parameters
    NUM_ITERATIONS = 20  # T: Number of time steps

    # Coverage surface sizes (in grid cells)
    S_MARIE  = 50
    S_THOR   = 50
    S_GRETHE = 200

    # Vehicle positions (initial) — world coordinates [N, E, z]
    MARIE_INITIAL_POS  = [CELL_GEOMETRY.origin[0]+10.0, CELL_GEOMETRY.origin[1]+0.0, 0.0]
    THOR_INITIAL_POS   = [CELL_GEOMETRY.origin[0]+0.0,  CELL_GEOMETRY.origin[1]+10.0, 0.0]
    GRETHE_INITIAL_POS = [CELL_GEOMETRY.origin[0]+5, CELL_GEOMETRY.origin[1]+5, 0.0]

    # Safety bubble radii (topology steps)
    MARIE_SAFETY_BUBBLE  = 3
    THOR_SAFETY_BUBBLE   = 3
    GRETHE_SAFETY_BUBBLE = 3

    # Communication ranges (metres)
    MARIE_COMM_RANGE  = 5
    THOR_COMM_RANGE   = 5
    GRETHE_COMM_RANGE = 5

    # ── Coverage Planning Weights — Marie ────────────────────────────────────
    MARIE_W_RANGE      = 0.0
    MARIE_W_SAFE       = 0.2
    MARIE_W_EXPLORE    = 1.0
    MARIE_W_EXPLOIT    = 0.0
    MARIE_MIN_SAFE_DIST = 6.0

    # ── Coverage Planning Weights — Thor ─────────────────────────────────────
    THOR_W_RANGE      = 0.5
    THOR_W_SAFE       = 0.2
    THOR_W_EXPLORE    = 1.0
    THOR_W_EXPLOIT    = 0.0
    THOR_MIN_SAFE_DIST = 6.0

    # ── Coverage Planning Weights — Grethe ───────────────────────────────────
    GRETHE_W_RANGE      = 1.0
    GRETHE_W_SAFE       = 1.0
    GRETHE_W_EXPLORE    = 0.0
    GRETHE_W_EXPLOIT    = 0.0
    GRETHE_MIN_SAFE_DIST = 8.0

    # ── Seed Selection Weights ────────────────────────────────────────────────
    MARIE_SEED_W_RANGE   = 0.0
    MARIE_SEED_W_SAFE    = 0.0
    MARIE_SEED_W_EXPLORE = 1.0
    MARIE_SEED_W_EXPLOIT = 0.0

    THOR_SEED_W_RANGE   = 0.5
    THOR_SEED_W_SAFE    = 0.0
    THOR_SEED_W_EXPLORE = 1.0
    THOR_SEED_W_EXPLOIT = 0.0

    GRETHE_SEED_W_RANGE   = 1.0
    GRETHE_SEED_W_SAFE    = 1.0
    GRETHE_SEED_W_EXPLORE = 0.2
    GRETHE_SEED_W_EXPLOIT = 0.1


def run_simulation():
    """Run the simulation with specified parameters."""

    p = SimulationParameters

    print("=" * 70)
    print("HUNTERS GAME - MULTI-VEHICLE COVERAGE PLANNING SIMULATION")
    print("=" * 70)
    geom = p.CELL_GEOMETRY
    print(f"  Cell geometry : {type(geom).__name__}")
    print(f"  Grid shape    : {geom.shape}")
    print(f"  Cell spacing  : {geom.cell_spacing_m} m")
    print(f"  Iterations    : {p.NUM_ITERATIONS}")
    print()

    sim = HuntersGameSimulation(geometry=p.CELL_GEOMETRY)

    try:
        sim.run(
            marie_init=p.MARIE_INITIAL_POS,
            thor_init=p.THOR_INITIAL_POS,
            grethe_init=p.GRETHE_INITIAL_POS,
            T=p.NUM_ITERATIONS,
            S_marie=p.S_MARIE,
            S_thor=p.S_THOR,
            S_grethe=p.S_GRETHE,
            # Marie parameters
            marie_w_range=p.MARIE_W_RANGE,
            marie_w_safe=p.MARIE_W_SAFE,
            marie_w_explore=p.MARIE_W_EXPLORE,
            marie_w_exploit=p.MARIE_W_EXPLOIT,
            marie_min_safe_dist=p.MARIE_MIN_SAFE_DIST,
            marie_seed_w_range=p.MARIE_SEED_W_RANGE,
            marie_seed_w_safe=p.MARIE_SEED_W_SAFE,
            marie_seed_w_explore=p.MARIE_SEED_W_EXPLORE,
            marie_seed_w_exploit=p.MARIE_SEED_W_EXPLOIT,
            # Thor parameters
            thor_w_range=p.THOR_W_RANGE,
            thor_w_safe=p.THOR_W_SAFE,
            thor_w_explore=p.THOR_W_EXPLORE,
            thor_w_exploit=p.THOR_W_EXPLOIT,
            thor_min_safe_dist=p.THOR_MIN_SAFE_DIST,
            thor_seed_w_range=p.THOR_SEED_W_RANGE,
            thor_seed_w_safe=p.THOR_SEED_W_SAFE,
            thor_seed_w_explore=p.THOR_SEED_W_EXPLORE,
            thor_seed_w_exploit=p.THOR_SEED_W_EXPLOIT,
            # Grethe parameters
            grethe_w_range=p.GRETHE_W_RANGE,
            grethe_w_safe=p.GRETHE_W_SAFE,
            grethe_w_explore=p.GRETHE_W_EXPLORE,
            grethe_w_exploit=p.GRETHE_W_EXPLOIT,
            grethe_min_safe_dist=p.GRETHE_MIN_SAFE_DIST,
            grethe_seed_w_range=p.GRETHE_SEED_W_RANGE,
            grethe_seed_w_safe=p.GRETHE_SEED_W_SAFE,
            grethe_seed_w_explore=p.GRETHE_SEED_W_EXPLORE,
            grethe_seed_w_exploit=p.GRETHE_SEED_W_EXPLOIT,
            prior_data=p.PRIOR_DATA,
        )
        print("=" * 70)
        print("Simulation completed successfully!")
        print("=" * 70)
    except Exception as e:
        print(f"Error during simulation: {e}")
        raise


if __name__ == "__main__":
    run_simulation()
    print("Game over")