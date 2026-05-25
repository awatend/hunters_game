"""
Parameters and execution script for the Hunters Game simulation.
This file contains all configuration parameters and runs the simulation.
"""

from test_class import HuntersGameSimulation


class SimulationParameters:
    """Container for all simulation parameters."""

    # Grid parameters
    GRID_SHAPE = (100, 180)  # (height, width) in cells
    GRID_RESOLUTION = 20.0    # meters per cell
    GRID_ORIGIN = (0.0, 0.0) # (N, E) in world coordinates

    # Simulation parameters
    NUM_ITERATIONS = 6 # T: Number of time steps

    # Coverage surface sizes
    S_MARIE = 100             # Number of cells for Marie's coverage
    S_THOR = 100              # Number of cells for Thor's coverage
    S_GRETHE = 200

    # Neighborhood type
    NEIGHBORHOOD = 8         # 4 or 8-neighborhood connectivity

    # Vehicle positions (initial)
    MARIE_INITIAL_POS = [300.0, 1222.0, 0.0]      # [N, E, z]
    THOR_INITIAL_POS = [400.0, 500.0, 0.0]       # [N, E, z]
    GRETHE_INITIAL_POS = [800.0, 2230.0, 0.0]     # [N, E, z]

    # Safety bubble radii (in grid cells)
    MARIE_SAFETY_BUBBLE = 2
    THOR_SAFETY_BUBBLE = 2
    GRETHE_SAFETY_BUBBLE = 2

    # Communication ranges (in meters)
    MARIE_COMM_RANGE = 5
    THOR_COMM_RANGE = 5
    GRETHE_COMM_RANGE = 5

    # =====================================================================
    # Coverage Planning Weights - Marie (grow_surface_from_seed)
    # =====================================================================
    MARIE_W_RANGE = 0.5     # Weight for being within communication range
    MARIE_W_SAFE = 0.2      # Weight for safety distance from obstacles
    MARIE_W_EXPLORE = 1.0  # Weight for exploring frontier areas
    MARIE_W_EXPLOIT = .0   # Weight for compact/connected coverage
    MARIE_MIN_SAFE_DIST = 6.0  # Minimum safe distance from obstacles (meters)

    # =====================================================================
    # Coverage Planning Weights - Thor (grow_surface_from_seed)
    # =====================================================================
    THOR_W_RANGE = 0.5      # Weight for being within communication range
    THOR_W_SAFE = 0.2        # Weight for safety distance from obstacles
    THOR_W_EXPLORE = 1.0     # Weight for exploring frontier areas
    THOR_W_EXPLOIT = .0    # Weight for compact/connected coverage
    THOR_MIN_SAFE_DIST = 6.0   # Minimum safe distance from obstacles (meters)

    # =====================================================================
    # Coverage Planning Weights - Grethe (grow_surface_from_seed)
    # =====================================================================

    GRETHE_W_RANGE = 1.0     # Weight for being within communication range
    GRETHE_W_SAFE = 0.9     # Weight for safety distance from obstacles
    GRETHE_W_EXPLORE = 0.0  # Weight for exploring frontier areas
    GRETHE_W_EXPLOIT = 0.0  # Weight for compact/connected coverage
    GRETHE_MIN_SAFE_DIST = 8.0  # Minimum safe distance from obstacles (meters)

    # =====================================================================
    # Seed Selection Weights (used in select_seed_ij)
    # =====================================================================
    # Marie seed selection
    MARIE_SEED_W_RANGE = 0.0
    MARIE_SEED_W_SAFE = 0.0
    MARIE_SEED_W_EXPLORE = 1.0
    MARIE_SEED_W_EXPLOIT = 0.0

    # Thor seed selection
    THOR_SEED_W_RANGE = 0.0
    THOR_SEED_W_SAFE = 0.0
    THOR_SEED_W_EXPLORE = 1.0
    THOR_SEED_W_EXPLOIT = 0.0

    # Grethe seed selection
    GRETHE_SEED_W_RANGE = 1.0
    GRETHE_SEED_W_SAFE = 1.0
    GRETHE_SEED_W_EXPLORE = 0.2
    GRETHE_SEED_W_EXPLOIT = 0.1


def run_simulation():
    """Run the simulation with specified parameters."""

    print("=" * 70)
    print("HUNTERS GAME - MULTI-VEHICLE COVERAGE PLANNING SIMULATION")
    print("=" * 70)
    print()

    print("Configuration:")
    print(f"  Grid Shape: {SimulationParameters.GRID_SHAPE}")
    print(f"  Grid Resolution: {SimulationParameters.GRID_RESOLUTION} m/cell")
    print(f"  Simulation Iterations: {SimulationParameters.NUM_ITERATIONS}")
    print(f"  Neighborhood: {SimulationParameters.NEIGHBORHOOD}")
    print()

    print("Coverage Sizes:")
    print(f"  Marie: {SimulationParameters.S_MARIE} cells")
    print(f"  Thor: {SimulationParameters.S_THOR} cells")
    print(f"  Grethe: {SimulationParameters.S_GRETHE} cells")
    print()

    print("Vehicle Parameters:")
    print(f"  Marie - Initial Pos: {SimulationParameters.MARIE_INITIAL_POS}, "
          f"Safety Bubble: {SimulationParameters.MARIE_SAFETY_BUBBLE}, "
          f"Comm Range: {SimulationParameters.MARIE_COMM_RANGE}")
    print(f"  Thor - Initial Pos: {SimulationParameters.THOR_INITIAL_POS}, "
          f"Safety Bubble: {SimulationParameters.THOR_SAFETY_BUBBLE}, "
          f"Comm Range: {SimulationParameters.THOR_COMM_RANGE}")
    print(f"  Grethe - Initial Pos: {SimulationParameters.GRETHE_INITIAL_POS}, "
          f"Safety Bubble: {SimulationParameters.GRETHE_SAFETY_BUBBLE}, "
          f"Comm Range: {SimulationParameters.GRETHE_COMM_RANGE}")
    print()

    print("=" * 70)
    print("Starting simulation...")
    print("=" * 70)
    print()

    # Create simulation instance
    sim = HuntersGameSimulation(
        shape=SimulationParameters.GRID_SHAPE,
        resolution=SimulationParameters.GRID_RESOLUTION,
        origin=SimulationParameters.GRID_ORIGIN
    )

    # Run simulation
    try:
        sim.run(
            marie_init=SimulationParameters.MARIE_INITIAL_POS,
            thor_init=SimulationParameters.THOR_INITIAL_POS,
            grethe_init=SimulationParameters.GRETHE_INITIAL_POS,
            T=SimulationParameters.NUM_ITERATIONS,
            S_marie=SimulationParameters.S_MARIE,
            S_thor=SimulationParameters.S_THOR,
            S_grethe=SimulationParameters.S_GRETHE,
            neighborhood=SimulationParameters.NEIGHBORHOOD,
            # Marie parameters
            marie_w_range=SimulationParameters.MARIE_W_RANGE,
            marie_w_safe=SimulationParameters.MARIE_W_SAFE,
            marie_w_explore=SimulationParameters.MARIE_W_EXPLORE,
            marie_w_exploit=SimulationParameters.MARIE_W_EXPLOIT,
            marie_min_safe_dist=SimulationParameters.MARIE_MIN_SAFE_DIST,
            marie_seed_w_range=SimulationParameters.MARIE_SEED_W_RANGE,
            marie_seed_w_safe=SimulationParameters.MARIE_SEED_W_SAFE,
            marie_seed_w_explore=SimulationParameters.MARIE_SEED_W_EXPLORE,
            marie_seed_w_exploit=SimulationParameters.MARIE_SEED_W_EXPLOIT,
            # Thor parameters
            thor_w_range=SimulationParameters.THOR_W_RANGE,
            thor_w_safe=SimulationParameters.THOR_W_SAFE,
            thor_w_explore=SimulationParameters.THOR_W_EXPLORE,
            thor_w_exploit=SimulationParameters.THOR_W_EXPLOIT,
            thor_min_safe_dist=SimulationParameters.THOR_MIN_SAFE_DIST,
            thor_seed_w_range=SimulationParameters.THOR_SEED_W_RANGE,
            thor_seed_w_safe=SimulationParameters.THOR_SEED_W_SAFE,
            thor_seed_w_explore=SimulationParameters.THOR_SEED_W_EXPLORE,
            thor_seed_w_exploit=SimulationParameters.THOR_SEED_W_EXPLOIT,
            # Grethe parameters
            grethe_w_range=SimulationParameters.GRETHE_W_RANGE,
            grethe_w_safe=SimulationParameters.GRETHE_W_SAFE,
            grethe_w_explore=SimulationParameters.GRETHE_W_EXPLORE,
            grethe_w_exploit=SimulationParameters.GRETHE_W_EXPLOIT,
            grethe_min_safe_dist=SimulationParameters.GRETHE_MIN_SAFE_DIST,
            grethe_seed_w_range=SimulationParameters.GRETHE_SEED_W_RANGE,
            grethe_seed_w_safe=SimulationParameters.GRETHE_SEED_W_SAFE,
            grethe_seed_w_explore=SimulationParameters.GRETHE_SEED_W_EXPLORE,
            grethe_seed_w_exploit=SimulationParameters.GRETHE_SEED_W_EXPLOIT
        )
        print()
        print("=" * 70)
        print("Simulation completed successfully!")
        print("=" * 70)
    except Exception as e:
        print()
        print("=" * 70)
        print(f"Error during simulation: {e}")
        print("=" * 70)
        raise


if __name__ == "__main__":
    run_simulation()
    print("Game over")

