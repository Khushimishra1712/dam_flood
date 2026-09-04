import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

output_dir = os.path.join("..", "data", "dem")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "raw_dem.tif")

width, height = 150, 150
x = np.linspace(-1, 1, width)
y = np.linspace(0, 2, height)
X, Y = np.meshgrid(x, y)

# V-valley elevation gradient
Z = (500 + 80 * (Y * 2) + 120 * (X**2)).astype(np.float32)

transform = from_origin(79.50, 30.50, 0.00027, 0.00027)

with rasterio.open(
    output_path,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype=Z.dtype,
    crs="EPSG:4326",
    transform=transform,
) as dst:
    dst.write(Z, 1)

print(f"Test DEM created at: {os.path.abspath(output_path)}")