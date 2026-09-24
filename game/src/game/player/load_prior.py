"""
load_prior.py
-------------
Loads SINMOD oceanographic prior data from a .npz archive,
reprojects to UTM, clips to a bounding box, and builds an
interpolated hexagonal field grid.

The main class is ``SinmodPrior``.  Its ``.geometry`` property
returns a ``HexagonalGeometry`` instance that is drop-in
compatible with ``CELL_GEOMETRY`` in test_config.py.

Standalone usage::

    python load_prior.py           # load defaults and plot both fields
    python load_prior.py --field temperature
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from pyproj import Transformer
from scipy.interpolate import griddata

from game.internal_map.cell_geometry import HexagonalGeometry

# ---------------------------------------------------------------------------
# Default data location (one level up from this file → sinmodPrior/)
# ---------------------------------------------------------------------------
_BASE_DIR        = Path(__file__).resolve().parent
_DEFAULT_FILEPATH = _BASE_DIR / "sinmodPrior" / "20250601_archive.npz"


class SinmodPrior:
    """ """

    def __init__(
        self,
        filepath: Path | str | None = None,
        x_min: float = 500_000.0,
        x_max: float = 510_000.0,
        y_min: float = 7_060_000.0,
        y_max: float = 7_070_000.0,
        spacing: float = 50.0,
        utm_zone: int = 32,
    ) -> None:
        self._filepath = Path(filepath) if filepath is not None else _DEFAULT_FILEPATH
        self._spacing  = float(spacing)
        self._utm_zone = int(utm_zone)

        # ── 1. Load archive ──────────────────────────────────────────────────
        data   = np.load(self._filepath, allow_pickle=True)
        latlon = data["latlon"]                 # (M, 2): [lat, lon]
        lat, lon = latlon[:, 0], latlon[:, 1]

        temperature = data["temperature"]
        biomass     = data["biomass"]




    # ── public API ────────────────────────────────────────────────────────

    @property
    def geometry(self) -> HexagonalGeometry:
        """
        ``HexagonalGeometry`` built from the SINMOD domain.
        Assign directly to ``CELL_GEOMETRY`` in ``test_config.py``.
        """
        return self._geometry

    @property
    def temperature(self) -> np.ndarray:
        """(n_rows, n_cols) interpolated temperature at hex centres (NaN outside hull)."""
        return self._temperature

    @property
    def biomass(self) -> np.ndarray:
        """(n_rows, n_cols) interpolated biomass at hex centres (NaN outside hull)."""
        return self._biomass

    def plot(self, field: str = "both") -> None:
        """
        Visualise interpolated field(s) on the hex grid over a real map.

        Parameters
        ----------
        field : {"temperature", "biomass", "both"}
        """


    def __repr__(self) -> str:
        H, W = self._geometry.shape
        return (
            f"SinmodPrior("
            f"shape=({H}, {W}), "
            f"spacing={self._spacing} m, "
            f"E=[{self._x_min:.0f}, {self._x_max:.0f}], "
            f"N=[{self._y_min:.0f}, {self._y_max:.0f}])"
        )


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot SINMOD prior hex fields.")
    parser.add_argument(
        "--field",
        choices=["temperature", "biomass", "both"],
        default="biomass",
        help="Which field to plot (default: both).",
    )
    args = parser.parse_args()

    prior = SinmodPrior()
    print(prior)
    print(f"  temperature range : {np.nanmin(prior.temperature):.3f} – "
          f"{np.nanmax(prior.temperature):.3f}")
    print(f"  biomass range     : {np.nanmin(prior.biomass):.3f} – "
          f"{np.nanmax(prior.biomass):.3f}")
    prior.plot(field=args.field)
