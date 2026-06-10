"""RECTANGLE EXAMPLE — download a full area (watch the download size!).

Downloading a rectangle gets EVERY 10 m pixel inside it. That gets big fast:
a 20 x 20 km box with TESSERA (128 values per pixel) is about 2 GB.

Rule of thumb:
  - training data at points  -> use sample_points (see 00_points_only.py)
  - a wall-to-wall map       -> cache a rectangle, but CHECK THE SIZE FIRST.

Run:
    python examples/00_small_rectangle.py
"""
import eoembed

# A small box (~2 x 2 km) in central London: lon_min, lat_min, lon_max, lat_max
bbox = (-0.13, 51.50, -0.11, 51.52)

# 1. ALWAYS check the size before downloading
mb = eoembed.estimate_size(bbox, "tessera")
print(f"This rectangle is about {mb:.0f} MB. ", end="")
if mb > 500:
    raise SystemExit("That's a lot — shrink the bbox or use sample_points instead.")
print("OK, proceeding.\n")

# 2. Open lazily (nothing downloads yet) ...
cube = eoembed.load_cube("tessera", bbox=bbox, year=2024)
print("Cube shape:", dict(cube.sizes))

# 3. ... and cache it ONCE to disk. Re-running this script later is instant
#    and works offline, because it reads the local copy.
store = eoembed.cache_aoi(cube, "data/my_area.zarr")
print("Cached to data/my_area.zarr")

# 4. Now you can do area-wide things, e.g. similarity search:
hits = eoembed.similarity_search(store, lon=-0.12, lat=51.51, top_k=50)
print(hits.head())
