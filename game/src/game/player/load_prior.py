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
    """
    Load a SINMOD .npz archive, reproject to UTM, clip to a rectangular
    bounding box, and build an interpolated hexagonal grid of temperature
    and biomass fields.

    Parameters
    ----------
    filepath : Path or str, optional
        Path to the SINMOD .npz archive.  Defaults to the standard location
        ``<repo_root>/sinmodPrior/20250601_archive.npz``.
    x_min, x_max : float
        UTM easting bounds (metres).  Default: 500 000 – 510 000.
    y_min, y_max : float
        UTM northing bounds (metres).  Default: 7 060 000 – 7 070 000.
    spacing : float
        Hex centre-to-centre distance (same row) in metres.  Default: 50.
    utm_zone : int
        UTM zone number (EPSG = 32600 + utm_zone).  Default: 32 (Norway).

    Attributes
    ----------
    geometry : HexagonalGeometry
        Drop-in ``CELL_GEOMETRY`` value for ``test_config.py``.
    temperature : np.ndarray, shape (n_rows, n_cols)
        Interpolated temperature at each hex centre.  NaN where outside
        the SINMOD data convex hull.
    biomass : np.ndarray, shape (n_rows, n_cols)
        Interpolated biomass at each hex centre.
    """

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

        # ── 2. WGS84 → UTM ──────────────────────────────────────────────────
        transformer = Transformer.from_crs(
            "EPSG:4326",
            f"EPSG:{32600 + utm_zone}",
            always_xy=True,
        )
        x_m, y_m = transformer.transform(lon, lat)

        # ── 3. Clip to initial bounding box ─────────────────────────────────
        clip = (
            (x_m >= x_min) & (x_m <= x_max) &
            (y_m >= y_min) & (y_m <= y_max)
        )
        x_m, y_m = x_m[clip], y_m[clip]

        def _to_1d(arr: np.ndarray) -> np.ndarray:
            """Drop leading time dimension if present."""
            return arr[0] if arr.ndim == 2 else arr

        temp = _to_1d(temperature)[clip]
        bio  = _to_1d(biomass)[clip]

        # ── 4. True domain extent (tight, after clip) ────────────────────────
        self._x_min = float(x_m.min())
        self._x_max = float(x_m.max())
        self._y_min = float(y_m.min())
        self._y_max = float(y_m.max())

        # ── 5. Generate hex-grid centres ─────────────────────────────────────
        hex_pts = self._generate_hex_grid()
        hx = hex_pts[:, 0]   # easting  (E)
        hy = hex_pts[:, 1]   # northing (N)

        # ── 6. Interpolate fields onto hex centres ───────────────────────────
        temp_flat = griddata((x_m, y_m), temp, (hx, hy), method="linear")
        bio_flat  = griddata((x_m, y_m), bio,  (hx, hy), method="linear")

        # ── 7. Build HexagonalGeometry ───────────────────────────────────────
        self._geometry = self._build_geometry()

        # ── 8. Reshape flat arrays to (n_rows, n_cols) grids ─────────────────
        H, W = self._geometry.shape
        self._temperature = np.full((H, W), np.nan, dtype=np.float32)
        self._biomass     = np.full((H, W), np.nan, dtype=np.float32)

        for i, (ex, ny) in enumerate(zip(hx, hy)):
            # geometry uses (N, E) = (northing, easting)
            row, col = self._geometry.world_to_cell(ny, ex)
            self._temperature[row, col] = temp_flat[i]
            self._biomass[row, col]     = bio_flat[i]

    # ── private helpers ───────────────────────────────────────────────────

    def _generate_hex_grid(self) -> np.ndarray:
        """
        Return (N_pts, 2) array of (easting, northing) hex-centre positions
        over the clipped domain, using the same odd-row-offset layout as
        ``HexagonalGeometry``.
        """
        dx = self._spacing
        dy = np.sqrt(3) / 2.0 * dx
        points: list[tuple[float, float]] = []
        row = 0
        y = self._y_min
        while y <= self._y_max + 1e-9:
            x_offset = (dx / 2.0) if row % 2 else 0.0
            x = self._x_min + x_offset
            while x <= self._x_max + 1e-9:
                points.append((x, y))
                x += dx
            y += dy
            row += 1
        return np.array(points)

    def _build_geometry(self) -> HexagonalGeometry:
        """
        Derive ``HexagonalGeometry`` parameters from the clipped data extent.
        """
        dx = self._spacing
        dy = np.sqrt(3) / 2.0 * dx
        n_cols = max(1, int(round((self._x_max - self._x_min) / dx)) + 1)
        n_rows = max(1, int(round((self._y_max - self._y_min) / dy)) + 1)
        # origin = world coords of cell [0, 0]: (N0=northing_min, E0=easting_min)
        origin = (self._y_min, self._x_min)
        return HexagonalGeometry(
            shape=(n_rows, n_cols),
            spacing=dx,
            origin=origin,
        )

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
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from matplotlib.patches import RegularPolygon

        to_plot: list[tuple[np.ndarray, str, str]] = []
        if field in ("temperature", "both"):
            to_plot.append((self._temperature, "Temperature (°C)", "coolwarm"))
        if field in ("biomass", "both"):
            to_plot.append((self._biomass, "Biomass", "viridis"))

        H, W   = self._geometry.shape
        radius = self._spacing / np.sqrt(3)
        utm_crs = ccrs.UTM(zone=self._utm_zone)

        for values, title, cmap in to_plot:
            fig = plt.figure(figsize=(10, 7))
            ax  = plt.axes(projection=utm_crs)
            ax.add_feature(cfeature.LAND,      facecolor="brown")
            ax.add_feature(cfeature.OCEAN,     facecolor="white")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

            cmap_fn = plt.cm.get_cmap(cmap)
            vmin = float(np.nanmin(values))
            vmax = float(np.nanmax(values))

            for row in range(H):
                for col in range(W):
                    val = values[row, col]
                    if np.isnan(val):
                        continue
                    north, east = self._geometry.cell_to_world(row, col)
                    color = cmap_fn((val - vmin) / (vmax - vmin + 1e-12))
                    patch = RegularPolygon(
                        (east, north),          # (x=E, y=N) for UTM axes
                        numVertices=6,
                        radius=radius,
                        orientation=np.radians(30),
                        facecolor=color,
                        edgecolor="none",
                        transform=utm_crs,
                    )
                    ax.add_patch(patch)

            ax.set_title(title)
            ax.set_xlim(480_000, 520_000)
            ax.set_ylim(7_020_000, 7_090_000)
            plt.tight_layout()
            plt.show()

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
