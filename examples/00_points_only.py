"""SIMPLEST EXAMPLE — get embeddings at a few points.

This is the light way: it downloads only the values AT your points
(a few kilobytes), not the whole area.

Run:
    pip install -e ".[tessera]"
    python examples/00_points_only.py
"""
import geopandas as gpd
from shapely.geometry import Point

import eoembed

# 1. Your points (replace with: gpd.read_file("my_points.geojson"))
points = gpd.GeoDataFrame(
    {
        "label": ["urban", "water", "park"],
        "geometry": [
            Point(-0.1276, 51.5072),   # Trafalgar Square
            Point(-0.0322, 51.5036),   # river Thames
            Point(-0.1657, 51.5101),   # Hyde Park
        ],
    },
    crs="EPSG:4326",
)

# 2. One call: embeddings at those points, saved as a small file
table = eoembed.sample_points("tessera", points, year=2024, out="my_points.parquet")

# 3. Look at the result: one row per point, columns e000..e127 are the embedding
print(table.head())
print(f"\nResult: {len(table)} rows x {table.shape[1]} columns -> tiny file, ready for sklearn.")
