"""Load a SINMOD archive into the game's hexagonal grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_GAME_SOURCE = _REPOSITORY_ROOT / "game" / "src"
for _import_root in (_REPOSITORY_ROOT, _GAME_SOURCE):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from sinmod_interaction import SINMODArchive

from game.internal_map.cell_geometry import HexagonalGeometry


_BASE_DIR = Path(__file__).resolve().parent
_DEFAULT_FILEPATH = _BASE_DIR / "20250601_archive.npz"


def _config_string(value: object) -> str:
    """Return an INI scalar without optional matching quotes."""
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def _config_float_values(value: str, *, name: str) -> tuple[float, ...]:
    """Parse comma-separated numeric values used by INI sections."""
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must contain comma-separated numbers") from exc


class LoadPrior:
    """
    Load a SINMOD variable and interpolate it onto a regular hexagonal grid.

    Values from an INI ``[prior]`` section can be passed directly:

        prior = LoadPrior(**parser["prior"])

    Comma-separated bounds and quoted scalar values are normalized before
    loading the archive.

    The original SINMOD field remains on its native Cartesian grid
    (typically 160 m x 160 m). The hexagonal grid is generated independently
    using a regular pointy-top hexagonal lattice.

    Hexagonal geometry for spacing S:

        column spacing = S
        row spacing    = sqrt(3)/2 * S
        row offset     = S/2
        hex radius     = S/sqrt(3)

    Invalid/shore SINMOD cells are preserved as invalid. Hex cells whose
    interpolation location cannot be evaluated from valid SINMOD data
    receive NaN.
    """

    def __init__(
        self,
        filepath: Path | str | None = None,
        *,
        path: Path | str | None = None,
        bounds: tuple[float, float, float, float] | str | None = None,
        time: int = 0,
        data: str = "biomass",
        spacing: float | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        **_: object,
    ) -> None:

        if filepath is not None and path is not None:
            raise ValueError("Specify only one of filepath and path")

        self._filepath = Path(_config_string(path or filepath or _DEFAULT_FILEPATH))

        self._data_name = _config_string(data)
        self._time = int(time)

          
        # Bounds compatibility
          

        if bounds is not None and any(v is not None for v in (x_min, x_max, y_min, y_max)):
            raise ValueError("Specify bounds or x_min/x_max/y_min/y_max, not both")

        if isinstance(bounds, str):
            bounds = tuple(_config_float_values(bounds, name="bounds"))

        if bounds is None and all(v is not None for v in (x_min, x_max, y_min, y_max)):
            bounds = (x_min, x_max, y_min, y_max,)

        if bounds is not None and len(bounds) != 4:
            raise ValueError("bounds must contain (x_min, x_max, y_min, y_max)")

          
        # Load SINMOD archive
          

        archive = SINMODArchive(self._filepath)
        xy = np.asarray(archive.ocean_model_xy, dtype=np.float64,)
        self._archive_shape = tuple(int(v) for v in archive.grid_shape)
        rows, cols = self._archive_shape

        if rows * cols != len(xy): raise ValueError("grid_shape does not match ocean_model_xy")

          
        # Extract regular SINMOD coordinate axes
          

        xy_2d = xy.reshape(rows, cols, 2)
        x_axis = xy_2d[0, :, 0]
        y_axis = xy_2d[:, 0, 1]

        print (f"original boundaries: x={x_axis.min()}..{x_axis.max()}, y={y_axis.min()}..{y_axis.max()}")

        if not np.allclose(xy_2d[:, :, 0], x_axis[None, :],):
            raise ValueError("SINMOD X coordinates are not regular")

        if not np.allclose(xy_2d[:, :, 1], y_axis[:, None],):
            raise ValueError("SINMOD Y coordinates are not regular")

        self._x_axis = x_axis
        self._y_axis = y_axis

          
        # Load original SINMOD field
          

        try:
            masked_values = archive.get_masked(self._data_name, timestep=self._time,)
            values_2d = np.asarray(archive.to_2d(masked_values), dtype=np.float64,)

        except (IndexError, KeyError) as exc:
            raise type(exc)(f"Could not load {self._data_name!r} at timestep {self._time} from {self._filepath}: "\
                              "{exc}") from exc

        if values_2d.shape != self._archive_shape:
            raise ValueError(f"Expected field shape {self._archive_shape}, got {values_2d.shape}")

        self._source_field = values_2d

        # The archive's get_masked() result already represents the
        # valid SINMOD field.
        self._source_mask = np.isfinite(values_2d)

          
        # Hex spacing
          

        archive_resolution = float(archive.resolution_m)
        self._spacing = float(spacing if spacing is not None else archive_resolution)
        if self._spacing <= 0: raise ValueError("spacing must be positive")

          
        # World bounds
          

        if bounds is None:
            bounds = (float(x_axis.min()), float(x_axis.max()), float(y_axis.min()), float(y_axis.max()),)

        self._bounds = tuple(float(v) for v in bounds)

        bx0, bx1, by0, by1 = self._bounds
        if not (bx0 < bx1 and by0 < by1): raise ValueError("bounds must be increasing")

        # Start the local grid at the requested lower-left point.  SINMOD
        # coordinates are discrete, so use the nearest source centre when a
        # bound falls between two archive cells.
        self._archive_origin = (
            self._nearest_axis_value(y_axis, by0),
            self._nearest_axis_value(x_axis, bx0),
        )

        # Build exact hexagonal geometry
        self._geometry = self._build_geometry()

        # Generate hex centers
        centres = self._centres()
          
        # Interpolate SINMOD field -> hex centers
        field = self._interpolate_field(values_2d, centres,)

        # Store hex field
        self._field = (field.reshape(self._geometry.shape).astype(np.float32))
        self._mask = np.isfinite(self._field)

    @staticmethod
    def _nearest_axis_value(axis: np.ndarray, value: float) -> float:
        """Return the archive coordinate nearest to a requested bound."""
        return float(axis[np.argmin(np.abs(axis - value))])

    # HEX GEOMETRY
    def _build_geometry(self) -> HexagonalGeometry:
        """
        Build a mathematically regular pointy-top hexagonal grid.
        """
        x_min, x_max, y_min, y_max = self._bounds
        col_spacing = self._spacing
        row_spacing = (np.sqrt(3.0) / 2.0 * self._spacing)
        # The local origin is cell [0, 0].  Use the requested extent divided
        # by the corresponding centre spacing to determine the shape.
        rows = max(1, int(np.ceil((y_max - self._archive_origin[0]) / row_spacing)))
        cols = max(1, int(np.ceil((x_max - self._archive_origin[1]) / col_spacing)))

        return HexagonalGeometry(
            shape=(rows, cols),
            spacing=self._spacing,
            origin=self._archive_origin,
            row_spacing=row_spacing,
        )

    
    # HEX CENTERS
    def _centres(self) -> np.ndarray:
        """
        Return hex centers as (x, y).
        """
        rows, cols = self._geometry.shape
        centres = []

        for row in range(rows):
            for col in range(cols):

                north, east = (self._geometry.cell_to_world(row, col,))
                x = east
                y = north
                # Only keep centers inside requested bounds.
                if (self._bounds[0] <= x <= self._bounds[1] and self._bounds[2] <= y <= self._bounds[3]):
                    centres.append((x, y))
                else:
                    centres.append((x, y))

        return np.asarray(centres, dtype=np.float64,)

    
    # INTERPOLATION
    def _interpolate_field(self, values_2d: np.ndarray, centres: np.ndarray,) -> np.ndarray:
        """
        Bilinearly interpolate the SINMOD field at hex centers.

        A center is only accepted if the interpolation stencil lies
        inside valid SINMOD data.
        """

          
        # First interpolate a validity field.
        #
        # 1 = valid SINMOD cell
        # 0 = invalid SINMOD cell
        #
        # This allows us to reject interpolation points that depend
        # on invalid/shore cells.
          

        valid_values = self._source_mask.astype(np.float64)

        valid_interpolator = RegularGridInterpolator((self._y_axis, self._x_axis,),
                                                     valid_values, method="linear", bounds_error=False, fill_value=0.0,)

        # Interpolate field.
        #
        # Replace invalid source values temporarily with 0.
        # The validity interpolation below determines whether the
        # result should actually be retained.
          

        clean_values = np.where(self._source_mask, values_2d, 0.0,)
        field_interpolator = RegularGridInterpolator((self._y_axis, self._x_axis,), clean_values,
                                                     method="linear", bounds_error=False, fill_value=np.nan,)
        query_points = centres[:, [1, 0]]
        interpolated = field_interpolator(query_points)
        validity = valid_interpolator(query_points)

          
        # Require the interpolation stencil to be completely valid.
        #
        # For bilinear interpolation, validity ~= 1 means all
        # surrounding SINMOD cells are valid.
          

        valid = (np.isfinite(interpolated) & (validity >= 0.999999))
        result = np.full(len(centres), np.nan, dtype=np.float64,)
        result[valid] = interpolated[valid]
        return result

    
    # PROPERTIES
    

    @property
    def geometry(self) -> HexagonalGeometry:
        return self._geometry

    @property
    def field(self) -> np.ndarray:
        """Interpolated field on the hexagonal grid."""
        return self._field.copy()

    @property
    def mask(self) -> np.ndarray:
        """Valid interpolated hex cells."""
        return self._mask.copy()

    @property
    def prior(self) -> np.ndarray:
        return self.field

    @property
    def data(self) -> np.ndarray:
        return self.field

    @property
    def source_field(self) -> np.ndarray:
        """Original SINMOD Cartesian field."""
        return self._source_field.copy()

    @property
    def source_mask(self) -> np.ndarray:
        """Original SINMOD validity mask."""
        return self._source_mask.copy()

    
    # LOAD
    def load_prior(self,) -> tuple[np.ndarray, HexagonalGeometry]:
        return (self.field, self.geometry,)

    
    # PLOT
    def plot(self, *, output: Path | str | None = None, show: bool = True,):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.patches import RegularPolygon

        field = self.field
        mask = self.mask
        finite = field[mask]
        source = self.source_field
        source_finite = source[np.isfinite(source)]

        if finite.size == 0 or source_finite.size == 0:
            raise ValueError("The prior contains no finite cells to plot")

        figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
        original_axis, cropped_axis = axes
        values = np.concatenate((finite, source_finite))
        norm = plt.Normalize(float(values.min()), float(values.max()),)
        cmap = plt.get_cmap("viridis")

        original_axis.imshow(
            source,
            origin="lower",
            extent=(
                float(self._x_axis.min()),
                float(self._x_axis.max()),
                float(self._y_axis.min()),
                float(self._y_axis.max()),
            ),
            cmap=cmap,
            norm=norm,
            aspect="equal",
        )
        x_min, x_max, y_min, y_max = self._bounds
        original_axis.add_patch(Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor="red",
            linewidth=1.5,
            linestyle="--",
        ))
        original_axis.set_title(f"Original {self._data_name}")
        original_axis.set_xlabel("x (m)")
        original_axis.set_ylabel("y (m)")

        for row, col in zip(*np.nonzero(mask)):

            north, east = (self.geometry.cell_to_world(row, col,))

            cropped_axis.add_patch(RegularPolygon(
                (east, north),
                6,
                radius=self.geometry.cell_radius * 1.001,
                orientation=np.pi / 3,
                facecolor=cmap(norm(field[row, col])),
                edgecolor="none",
                linewidth=0,
                antialiased=False,
            ))

        extent_x_min, extent_x_max, extent_y_min, extent_y_max = self.geometry.world_extent()
        cropped_axis.set_aspect("equal", adjustable="box",)
        cropped_axis.set_xlim(extent_x_min, extent_x_max,)
        cropped_axis.set_ylim(extent_y_min, extent_y_max,)
        cropped_axis.set_xlabel("x (m)")
        cropped_axis.set_ylabel("y (m)")
        cropped_axis.set_title(f"Cropped {self._data_name} at time {self._time}")
        figure.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axes,
            label=self._data_name,
        )

        if output is not None:
            figure.savefig(output, dpi=200, bbox_inches="tight",)

        if show:
            plt.show()

        return figure, axes

    
    # REPRESENTATION
    def __repr__(self) -> str:
        return (
            f"LoadPrior("
            f"data={self._data_name!r}, "
            f"time={self._time}, "
            f"shape={self.geometry.shape}, "
            f"dtype={self.field.dtype})"
        )

# Existing callers use this name; keep it as a backwards-compatible alias.
SinmodPrior = LoadPrior


def main() -> None:
    from matplotlib.patches import RegularPolygon
    """Visualize the loaded float32 hexagonal field."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=_DEFAULT_FILEPATH)
    parser.add_argument("--data", default="biomass")
    parser.add_argument("--time", type=int, default=0)
    parser.add_argument("--bounds", nargs=4, type=float, metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"))
    parser.add_argument("--spacing", type=float, default=160.0)
    parser.add_argument("--output", type=Path, help="Save the plot instead of only displaying it")
    args = parser.parse_args()


    prior = LoadPrior(path=args.path, data=args.data, time=args.time, bounds=args.bounds, spacing=args.spacing,)
    print(f"Loaded {prior} from {args.path}")
    print(f"Bounds: {prior._bounds}")
    print(f"Origin: {prior.geometry.origin}; shape: {prior.geometry.shape}")
    print("spacing:", prior.geometry._spacing)
    print("row_sp:", prior.geometry._row_sp)
    print("col_sp:", prior.geometry._col_sp)
    print("radius:", prior.geometry.cell_radius)

    R = prior.geometry.cell_radius

    for row, col in [(0, 0), (0, 1), (1, 0)]:
        N, E = prior.geometry.cell_to_world(row, col)

        hexagon = RegularPolygon((E, N), 6, radius=R, orientation=np.pi / 3,)

        vertices = hexagon.get_transform().transform(hexagon.get_path().vertices)

        print(f"\ncell ({row},{col}), center=({E},{N})")
        print(vertices)
    prior.plot(output=args.output, show=True)


if __name__ == "__main__":
    main()
