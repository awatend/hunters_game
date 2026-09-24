
from __future__ import annotations

import numpy as np
from pathlib import Path
from pyproj import Transformer
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from sinmod_interaction import SINMODArchive, build_hex_grid

_BASE_DIR        = Path(__file__).resolve().parent
_DEFAULT_FILEPATH = _BASE_DIR  / "20250601_archive.npz"

archive = SINMODArchive(_DEFAULT_FILEPATH)
grid = build_hex_grid(archive, coarse_resolution_m=1000.0,
                      aggregate_variables=["biomass"])

grid.save(f"{_BASE_DIR }/20250601_grid_hex3_1000m.npz")

# Side-by-side visualization: original vs coarsened
xy = grid.xy
bio = grid.get_field("biomass")
if bio.ndim > 1:  # if time-dependent, pick first timestep
    bio = bio[0]

bio_2d = archive.to_2d(archive.get_masked("biomass", timestep=0))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

im1 = ax1.imshow(bio_2d, origin="lower", cmap="viridis")
ax1.set_title("Original biomass (t=0)")
fig.colorbar(im1, ax=ax1, label="Biomass")

sc2 = ax2.scatter(xy[:, 0], xy[:, 1], c=bio, s=20, cmap="viridis")
ax2.set_aspect("equal", adjustable="box")
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.set_title("Coarsened hex grid biomass")
fig.colorbar(sc2, ax=ax2, label="Biomass")

plt.tight_layout()
plt.savefig(_BASE_DIR / "original_vs_coarse_hex_biomass.png", dpi=200, bbox_inches="tight")
plt.show()