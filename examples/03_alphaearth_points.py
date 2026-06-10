"""ALPHAEARTH EXAMPLE — embeddings at points via Google Earth Engine.

ONE-TIME SETUP (5 minutes, free for noncommercial use):
  1. Go to https://earthengine.google.com -> "Get Started"
     -> sign in with a Google account
     -> register a Cloud project for NONCOMMERCIAL use (free).
     Note your project id, it looks like:  ee-yourname
  2. pip install -e ".[alphaearth]"
  3. The first run opens a browser window asking you to authorize Earth Engine.
     Approve it once — it is remembered on this machine.

WHY ALPHAEARTH VIA EARTH ENGINE?
  The extraction runs INSIDE Google's servers ("server-side sampling"):
  your computer receives only the final table. 10,000 points = ~2.5 MB
  downloaded, and nothing else. There is no faster way to get training data.

Run:
    python examples/03_alphaearth_points.py
"""
import geopandas as gpd
from shapely.geometry import Point

import eoembed

# >>> CHANGE THIS to your own Earth Engine project id <<<
EE_PROJECT = "ee-yourname"

# 1. Your labelled points (replace with gpd.read_file("my_points.geojson"))
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

# 2. Sample AlphaEarth (64 values per point) — runs inside Google's servers
table = eoembed.sample_points(
    "alphaearth", points, year=2024,
    backend="ee", project=EE_PROJECT,
    out="alphaearth_points.parquet",
)
print(table.head())

# 3. Reminder: AlphaEarth data is CC-BY 4.0 — when you publish results, include:
print()
print("Attribution required in your outputs:")
print(' "' + eoembed.describe("alphaearth")["attribution"] + '"')

# BONUS — compare both models at the same points (needs [all] extras installed):
# pairs = eoembed.paired_sample(points, year=2024,
#                               source_a="alphaearth", source_b="tessera",
#                               source_a_kwargs={"backend": "ee", "project": EE_PROJECT})
# print(pairs.filter(regex="^(a_|b_)e00").head())
