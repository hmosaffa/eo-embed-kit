"""End-to-end example: TESSERA embeddings -> linear probe -> 10 m map.

Run:  pip install -e ".[tessera]"  then  python examples/01_quickstart.py
"""
import geopandas as gpd
from shapely.geometry import Point

import eoembed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BBOX = (-0.15, 51.49, -0.10, 51.53)  # central London (small!)
YEAR = 2024

# --- 1. lazy cube + one-time local cache -----------------------------------
# NOTE: cache_aoi downloads EVERY pixel in BBOX. Check the size first!
print(f"Area download size: ~{eoembed.estimate_size(BBOX, 'tessera'):.0f} MB")
cube = eoembed.load_cube("tessera", bbox=BBOX, year=YEAR)
store = eoembed.cache_aoi(cube, "data/london_2024.zarr")
print("cached cube:", dict(store.sizes))

# --- 2. toy labels (replace with your own GeoJSON/Shapefile) ----------------
labels = gpd.GeoDataFrame(
    {
        "label": ["urban", "urban", "water", "water", "green", "green"],
        "geometry": [
            Point(-0.1276, 51.5072), Point(-0.0877, 51.5136),   # central London
            Point(-0.0322, 51.5036), Point(0.0098, 51.4880),    # Thames
            Point(-0.1657, 51.5101), Point(-0.1500, 51.5330),   # parks
        ],
    },
    crs="EPSG:4326",
)

# --- 3. extract features -> GeoParquet --------------------------------------
table = eoembed.sample_points("tessera", labels, year=YEAR, out="data/train.parquet")
print(table.filter(regex="^e0(0|1)").head())

# --- 4. train anything you like on the table (deliberately not part of core) -
cols = [c for c in table.columns if c.startswith("e")]
probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
probe.fit(table[cols], table["label"])
# see examples/02_linear_probe_example.py for CV reports + wall-to-wall maps

# --- 5. similarity search (core feature) -------------------------------------
hits = eoembed.similarity_search(store, lon=-0.1276, lat=51.5072, top_k=200)
print(hits.head())
