import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pyproj import Transformer
from scipy.interpolate import griddata
from matplotlib.patches import RegularPolygon

BASE_DIR = Path(__file__).resolve().parent.parent

filepath=BASE_DIR/"sinmodPrior"/"20250601_archive.npz"

x_min  = 500000
x_max  = 520000
y_min  = 7070000
y_max  = 7080000


data = np.load(filepath, allow_pickle=True)

latlon = data["latlon"]   # (M, 2)
lat = latlon[:, 0]
lon = latlon[:, 1]


temperature = data["temperature"]
biomass = data["biomass"]

temperature = data["temperature"]
biomass = data["biomass"]

# -----------------------------
# WGS84 → UTM (meters)
# -----------------------------
transformer = Transformer.from_crs(
    "EPSG:4326",   # lat/lon WGS84
    "EPSG:32632",   # UTM zone 32N (Norway)
    always_xy=True
)

x_m, y_m = transformer.transform(lon, lat)
# -----------------------------
# Handle possible time dimension
# -----------------------------
def get_1d(arr):
    return arr[0] if arr.ndim == 2 else arr

temp = get_1d(temperature)
bio = get_1d(biomass)

#Cop Field

mask = (
    (x_m >= x_min) &
    (x_m <= x_max) &
    (y_m >= y_min) &
    (y_m <= y_max)
)

x_m = x_m[mask]
y_m = y_m[mask]
temp = temp[mask]
bio = bio[mask]

# =========================================================
# 2. DOMAIN BOUNDS
# =========================================================
x_min, x_max = x_m.min(), x_m.max()
y_min, y_max = y_m.min(), y_m.max()

# =========================================================
# 3. HEX GRID GENERATOR
# =========================================================
def generate_hex_grid(xmin, xmax, ymin, ymax, spacing):
    dx = spacing
    radius = dx / 2
    dy = np.sqrt(3) * radius

    pts = []
    row = 0

    y = ymin
    while y <= ymax:
        x_offset = radius if row % 2 else 0

        x = xmin + x_offset
        while x <= xmax:
            pts.append((x, y))
            x += dx

        y += dy
        row += 1

    return np.array(pts), radius

# =========================================================
# 4. BUILD HEX GRID
# =========================================================
spacing = 1000  # meters (from SINMOD metadata)  # meters (from your dataset analysis)

hex_pts = generate_hex_grid(x_min, x_max, y_min, y_max, spacing)

hx = hex_pts[:, 0]
hy = hex_pts[:, 1]

# =========================================================
# 5. INTERPOLATE DATA ONTO HEX CENTERS
# =========================================================
temp_hex = griddata((x_m, y_m),temp,(hx, hy),method="linear")

bio_hex = griddata((x_m, y_m),bio,(hx, hy),method="linear")

# =========================================================
# 6. TRUE HEXAGON PLOTTING FUNCTION
# =========================================================
def plot_hex_field(x, y, values, title, cmap):
    fig, ax = plt.subplots(figsize=(10, 7))

    radius = spacing / np.sqrt(3)

    for xi, yi, val in zip(x, y, values):
        if np.isnan(val):
            continue

        hex_cell = RegularPolygon(
            (xi, yi),
            numVertices=6,
            radius=radius,
            orientation=np.radians(30),
            facecolor=plt.cm.get_cmap(cmap)(val),
            edgecolor="none"
        )
        ax.add_patch(hex_cell)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    plt.tight_layout()
    plt.show()

# =========================================================
# 7. PLOT RESULTS
# =========================================================
plot_hex_field(hx, hy, temp_hex, "Temperature (Hex Grid)", "coolwarm")
plot_hex_field(hx, hy, bio_hex, "Biomass (Hex Grid)", "viridis")