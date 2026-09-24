"""
cell_geometry.py
----------------
Strategy pattern for cell shapes.

To add a new cell shape:
    1. Subclass CellGeometry.
    2. Implement all abstract methods.
    3. Set CELL_GEOMETRY in test_config.py.

Nothing else needs to change.
"""
from abc import ABC, abstractmethod
from collections import deque
from typing import List, Tuple, Union

import numpy as np
import shapely.geometry as sg
from shapely.ops import unary_union
from shapely.prepared import prep

# Type aliases
CellIndex = Tuple[int, int]   # (row, col)  into the backing numpy array
WorldPt   = Tuple[float, float]  # (N, E)  in metres

ShapelyPolygon = Union[sg.Polygon, sg.MultiPolygon]

# ═══════════════════════════════════════════════════════════════════════════
#  Abstract base
# ═══════════════════════════════════════════════════════════════════════════

class CellGeometry(ABC):
    """
    Geometry abstraction for cell-based environments.

    World coordinates are represented as (N, E), while Shapely uses
    (x, y) = (E, N).
    """

    # ── grid dimensions ───────────────────────────────────────────────────
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """(n_rows, n_cols) of the backing numpy arrays."""

    # ── coordinate conversion ─────────────────────────────────────────────
    @abstractmethod
    def world_to_cell(self, N: float, E: float) -> CellIndex:
        """Nearest cell (row, col) to world point (N, E). Clamped to bounds."""

    @abstractmethod
    def cell_to_world(self, row: int, col: int) -> WorldPt:
        """World (N, E) of the centre of cell (row, col)."""


   # ── Cell geometry ─────────────────────────────────────────────

    @abstractmethod
    def cell_vertices(self, row: int, col: int) -> List[WorldPt]:
        """
        Return ordered cell vertices as (N, E).

        The polygon is not required to be explicitly closed.
        """

    def cell_polygon(self, row: int, col: int) -> sg.Polygon:
        """
        Return the cell as a Shapely Polygon.

        Shapely convention:
            x = E
            y = N
        """
        vertices = self.cell_vertices(row, col)

        return sg.Polygon([(E, N)for N, E in vertices])

    # ── batch coordinate conversion ─────────────────────────────────────────────────────────

    def cells_to_world(self, mask: np.ndarray,) -> Tuple[np.ndarray, np.ndarray]:
        rows, cols = np.where(mask)

        if rows.size == 0:
            return np.array([]), np.array([])

        centres = np.array([
            self.cell_to_world(int(r), int(c))
            for r, c in zip(rows, cols)
        ])

        return centres[:, 0], centres[:, 1]


    # ── adjacency ─────────────────────────────────────────────────────────
    @abstractmethod
    def neighbors(self, row: int, col: int) -> List[CellIndex]:
        """
        All valid in-bounds neighbours of (row, col).
        Rectangular: up to 8 (diagonal included).
        Hexagonal:   exactly 6.
        """

    # ── distance scale ────────────────────────────────────────────────────
    @property
    @abstractmethod
    def cell_spacing_m(self) -> float:
        """
        Characteristic spacing in metres.

        Used as the multiplier when converting distance_transform_edt output
        (in cell units) to metres::

            dist_m = distance_transform_edt(mask) * geom.cell_spacing_m

        """

    # ── world extent ──────────────────────────────────────────────────────
    @abstractmethod
    def world_extent(self) -> Tuple[float, float, float, float]:
        """(E_min, E_max, N_min, N_max) bounding box for matplotlib axes."""

    # ── mask ↔ Shapely geometry ────────────────────────────────────────────────
    def mask_to_polygon(self, mask: np.ndarray, threshold: float = 0.5,cleanup_tolerance_m: float = 0.05,
    min_area_m2: float = 1.0,) -> ShapelyPolygon:

        mask = np.asarray(mask)

        if mask.shape != self.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match geometry shape {self.shape}")

        active = mask >= threshold

        rows, cols = np.where(active)

        if rows.size == 0:
            return sg.Polygon()

        # 1. Convert active cells to polygons
        polygons = [self.cell_polygon(int(row), int(col)) for row, col in zip(rows, cols)]

        # 2. Merge cells and repair topology
        merged = unary_union(polygons)
        if merged.is_empty:
            return sg.Polygon()
        if not merged.is_valid:
            merged = merged.buffer(0)


        # 3. Remove very small disconnected components
        if merged.geom_type == "MultiPolygon":
            components = [p for p in merged.geoms if p.area >= min_area_m2]
            if not components:
                return sg.Polygon()
            merged = unary_union(components)

        # 4. Remove narrow slivers / tiny gaps
        if cleanup_tolerance_m > 0:
            merged = (merged.buffer(cleanup_tolerance_m).buffer(-cleanup_tolerance_m))

        # Repair again after buffering
        if not merged.is_valid:
            merged = merged.buffer(0)
        if merged.is_empty:
            return sg.Polygon()

        # 5. If multiple polygons remain, keep the largest one

        if merged.geom_type == "MultiPolygon":
            merged = max(merged.geoms, key=lambda p: p.area)

        # 6. Remove ALL interior holes
        if merged.geom_type == "Polygon":
            merged = sg.Polygon(merged.exterior)

        # 7. Final topology cleanup
        merged = merged.buffer(0)
        return merged


    # ── Compatibility: Shapely geometry -> outer boundary ───────────────────

    def mask_to_boundary_polygon(self, mask: np.ndarray,
                                 threshold: float = 0.5,) -> List[WorldPt]:
        """
        Compatibility method.

        Returns the exterior boundary of the largest connected component.

        For applications where disconnected regions matter, use
        mask_to_polygon() directly, because it can return a MultiPolygon.
        """

        geometry = self.mask_to_polygon(mask, threshold)

        if geometry.is_empty:
            return []

        if isinstance(geometry, sg.MultiPolygon):
            geometry = max(geometry.geoms, key=lambda polygon: polygon.area,)

        return [(float(y), float(x)) for x, y in geometry.exterior.coords]


    # ── Shapely polygon -> cell mask ─────────────────────────────────────────────────────
    def polygon_to_mask(self, polygon: sg.base.BaseGeometry,
                        shape: Tuple[int, int] = None,
                        include_boundary: bool = True,) -> np.ndarray:
        """
        Rasterise a Shapely polygon onto the cell grid.

        A cell is active when its centre lies inside the polygon.

        Parameters
        ----------
        polygon:
            Shapely Polygon or MultiPolygon.

        shape:
            Optional output shape. Defaults to self.shape.

        include_boundary:
            If True, cell centres lying exactly on the boundary are included.
        """

        if shape is None:
            shape = self.shape

        H, W = shape
        if len(shape) != 2:
            raise ValueError("shape must be (rows, cols)")

        if polygon.is_empty:
            return np.zeros((H, W), dtype=bool)

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        prepared = prep(polygon)
        mask = np.zeros((H, W), dtype=bool)

        # Restrict work to the polygon bounding box.
        minx, miny, maxx, maxy = polygon.bounds

        for row in range(H):
            for col in range(W):
                N, E = self.cell_to_world(row, col)
                if not (miny <= N <= maxy and minx <= E <= maxx):
                    continue

                point = sg.Point(E, N)
                if include_boundary:
                    if prepared.covers(point):
                        mask[row, col] = True
                else:
                    if prepared.contains(point):
                        mask[row, col] = True

        return mask


    # ── safety bubble ─────────────────────────────────────────────────────
    @abstractmethod
    def bubble_cells(
        self, center_row: int, center_col: int, radius_steps: int
    ) -> List[CellIndex]:
        """
        All valid cell indices within ``radius_steps`` topology steps from
        the centre, *excluding* the centre itself.

        Rectangular: Chebyshev (L∞) square.
        Hexagonal:   BFS over hex neighbours.
        """

    # ── neighbourhood mask (for seed selection) ───────────────────────────
    @abstractmethod
    def neighborhood_mask(
        self, row: int, col: int, size: int
    ) -> np.ndarray:
        """
        Boolean (H, W) mask: True for cells within ``size`` topology steps
        of (row, col), False at (row, col) itself.
        """


# ═══════════════════════════════════════════════════════════════════════════
#  Rectangular geometry  (current behaviour, unchanged)
# ═══════════════════════════════════════════════════════════════════════════

class RectangularGeometry(CellGeometry):
    """
    Square-cell rectangular grid — replicates the original
    Environment(shape, resolution, origin) behaviour exactly.

    Parameters
    ----------
    shape : (n_rows, n_cols)
    resolution : float
        Cell side length in metres.
    origin : (N0, E0)
        World coordinates of the *centre* of cell [0, 0].
        (Matches the convention used by the original world_to_grid.)
    connectivity : int
        4 or 8.  Default 8 (diagonals included), matching the original
        NEIGHBORHOOD = 8 simulation default.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        resolution: float,
        origin: Tuple[float, float] = (0.0, 0.0),
        connectivity: int = 8,
    ):
        if len(shape) != 2:
            raise ValueError("shape must be (rows, cols)")

        if resolution <= 0:
            raise ValueError("resolution must be > 0")

        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")

        self._shape       = tuple(shape)
        self._res         = float(resolution)
        self._origin      = tuple(origin)
        self._connectivity = connectivity

    # ── properties used by Environment shims ──────────────────────────────
    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def cell_spacing_m(self) -> float:
        return self._res

    # kept for Environment.origin shim
    @property
    def origin(self) -> Tuple[float, float]:
        return self._origin

    # ── coordinate conversion ─────────────────────────────────────────────
    def cell_to_world(self, row, col):

        H, W = self._shape
        row = int(np.clip(row, 0, H - 1))
        col = int(np.clip(col, 0, W - 1))
        N = self._origin[0] + row * self._res #perhaps add 0.5
        E = self._origin[1] + col * self._res

        return float(N), float(E)

    def world_to_cell(self, N, E):

        H, W = self._shape
        row = int(round((N - self._origin[0]) / self._res))
        col = int(round((E - self._origin[1]) / self._res))
        return (int(np.clip(row, 0, H - 1)), int(np.clip(col, 0, W - 1)),)

    # ── neighbors ───────────────────────────────────────────────────────
    def neighbors(self, row: int, col: int) -> List[CellIndex]:
        H, W = self._shape
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self._connectivity == 8:
            offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        return [(row + di, col + dj) for di, dj in offsets if 0 <= row + di < H and 0 <= col + dj < W]

    # ── cell polygon ──────────────────────────────────────────────────────
    def cell_vertices(self, row, col):

        N, E = self.cell_to_world(row, col)
        h = self._res / 2.0
        return [(N - h, E - h), (N - h, E + h), (N + h, E + h), (N + h, E - h),]

    # ── world extent ──────────────────────────────────────────────────────
    def world_extent(self):

        H, W = self._shape
        N0, E0 = self._origin
        half = self._res / 2.0
        return (E0 - half, E0 + (W - 1) * self._res + half, N0 - half, N0 + (H - 1) * self._res + half,)

    # ── safety bubble ─────────────────────────────────────────────────────
    def bubble_cells(self, center_row, center_col, radius_steps,):

        H, W = self._shape
        result = []
        r = int(radius_steps)

        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if dr == 0 and dc == 0:
                    continue
                row = center_row + dr
                col = center_col + dc

                if (0 <= row < H and 0 <= col < W):
                    result.append((row, col))

        return result

    # ── neighbourhood mask ────────────────────────────────────────────────
    def neighborhood_mask(self, row: int, col: int, size: int) -> np.ndarray:
        H, W = self._shape
        mask = np.zeros((H, W), dtype=bool)
        i_min = max(0, row - size)
        i_max = min(H, row + size + 1)
        j_min = max(0, col - size)
        j_max = min(W, col + size + 1)
        mask[i_min:i_max, j_min:j_max] = True
        mask[row, col] = False
        return mask


# ═══════════════════════════════════════════════════════════════════════════
#  Hexagonal geometry  (pointy-top, odd-row offset — matches load_prior.py)
# ═══════════════════════════════════════════════════════════════════════════

class HexagonalGeometry(CellGeometry):
    """
    Pointy-top hexagonal grid using odd-row offset coordinates.

    The centre of cell [0, 0] is ``origin``.

    Parameters
    ----------
    shape:
        (n_rows, n_cols)

    spacing:
        Centre-to-centre distance between cells in the same row.

    origin:
        (N0, E0), centre of cell [0, 0].
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        spacing: float,
        origin: Tuple[float, float] = (0.0, 0.0),
        row_spacing: float | None = None,
    ):

        if len(shape) != 2:
            raise ValueError("shape must be (rows, cols)")

        if spacing <= 0:
            raise ValueError("spacing must be > 0")

        self._shape    = tuple(shape)
        self._origin   = tuple(origin)
        self._spacing  = spacing
        self._col_sp   = spacing                          # dx
        self._row_sp   = (
            np.sqrt(3) / 2.0 * spacing
            if row_spacing is None else float(row_spacing)
        )
        if self._row_sp <= 0:
            raise ValueError("row_spacing must be > 0")
        self._radius    = spacing / np.sqrt(3)              # circumradius


    # ── properties ───────────────────────────────────────────────────────
    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def cell_spacing_m(self) -> float:
        return self._col_sp

    @property
    def origin(self) -> Tuple[float, float]:
        return self._origin

    @property
    def cell_radius(self):
        return self._radius


    # ── coordinate conversion ─────────────────────────────────────────────
    def cell_to_world(self, row, col):

        H, W = self._shape

        row = int(np.clip(row, 0, H - 1))
        col = int(np.clip(col, 0, W - 1))
        row_offset = (self._spacing / 2.0 if row % 2 else 0.0)
        #row_offset =0
        N = (self._origin[0] + row * self._row_sp)
        E = (self._origin[1] + col * self._col_sp + row_offset)

        return float(N), float(E)

    def world_to_cell(self, N, E):

        H, W = self._shape

        # Approximate row
        row0 = int(round((N - self._origin[0]) / self._row_sp))
        row0 = int(np.clip(row0, 0, H - 1))

        candidates = []

        for row in range(max(0, row0 - 1), min(H, row0 + 2),):
            row_offset = (self._spacing / 2.0 if row % 2 else 0.0)
            #row_offset = 0
            col0 = int(round((E - self._origin[1] - row_offset) / self._col_sp))

            for col in (col0 - 1, col0, col0 + 1,):

                if 0 <= col < W:
                    cN, cE = self.cell_to_world(row, col,)
                    d2 = ((N - cN) ** 2 + (E - cE) ** 2)
                    candidates.append((d2, row, col))

        if not candidates:
            return 0, 0

        _, row, col = min(candidates)
        return row, col

    # ── adjacency ─────────────────────────────────────────────────────────
    def neighbors(self, row: int, col: int) :
        """6-neighbour odd-r hex adjacency."""
        H, W = self._shape
        if row % 2 == 0:   # even row
            offsets = [(-1, -1), (-1, 0), ( 0, -1), ( 0, 1), ( 1, -1), ( 1, 0),]
        else:              # odd row
            offsets = [(-1,  0), (-1, 1), ( 0, -1), ( 0, 1), ( 1,  0), ( 1, 1),]
        return [(row + dr, col + dc) for dr, dc in offsets if 0 <= row + dr < H and 0 <= col + dc < W]

    # ── cell polygon ──────────────────────────────────────────────────────
    def cell_vertices(self, row, col):

        N, E = self.cell_to_world(row, col)
        R = self._radius
        vertices = []

        for k in range(6):
            angle = (np.pi / 6.0 + k * np.pi / 3.0)
            vertices.append((N + R * np.sin(angle), E + R * np.cos(angle),))

        return vertices

    # ── world extent ──────────────────────────────────────────────────────
    def world_extent(self):

        H, W = self._shape

        centres = [self.cell_to_world(r, c) for r, c in
                   [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1),]]
        Ns = [N for N, E in centres]
        Es = [E for N, E in centres]
        R = self._radius

        return (min(Es) - R, max(Es) + R, min(Ns) - R, max(Ns) + R,)

    # ── mask → boundary polygon ───────────────────────────────────────────


    # ── safety bubble ─────────────────────────────────────────────────────
    def bubble_cells(self, center_row, center_col, radius_steps,):

        visited = {(center_row, center_col)}
        frontier = deque([(center_row, center_col)])
        result = []

        for _ in range(int(radius_steps)):
            next_frontier = deque()
            while frontier:
                row, col = frontier.popleft()
                for neighbor in self.neighbors(row, col,):

                    if neighbor in visited:
                        continue

                    visited.add(neighbor)
                    result.append(neighbor)
                    next_frontier.append(neighbor)

            frontier = next_frontier

        return result



    # ── neighbourhood mask ────────────────────────────────────────────────
    def neighborhood_mask(self, row, col, size,):

        H, W = self._shape
        mask = np.zeros((H, W), dtype=bool,)
        for r, c in self.bubble_cells(row, col, size,):
            mask[r, c] = True
        return mask

# ═══════════════════════════════════════════════════════════════════════════
#  Utility
# ═══════════════════════════════════════════════════════════════════════════

def _point_in_polygon(N: float, E: float, poly_NE: np.ndarray) -> bool:
    """
    Ray-casting point-in-polygon test.
    poly_NE : (K, 2) array of (N, E) polygon vertices (not closed).
    """
    inside = False
    n = len(poly_NE)
    j = n - 1
    for i in range(n):
        xi, yi = poly_NE[i, 1], poly_NE[i, 0]   # E, N
        xj, yj = poly_NE[j, 1], poly_NE[j, 0]
        if ((yi > N) != (yj > N)) and (
            E < (xj - xi) * (N - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside