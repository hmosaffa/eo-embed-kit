"""Reproducible benchmark: eoembed vs. raw provider access.

Run this yourself and paste the *measured* table into the README — do not
publish numbers you haven't reproduced. Network speed dominates, so report
your bandwidth alongside results.

    python benchmarks/bench_point_extraction.py --n-points 5000 --year 2024

Measures, for TESSERA over a fixed AOI:
  A) eoembed.sample_points  (Zarr chunk-local streaming)
  B) naive baseline: download every intersecting tile as GeoTIFF, then sample
and reports wall time, bytes on disk, and peak RSS.
"""
from __future__ import annotations

import argparse
import resource
import tempfile
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

BBOX = (-0.2, 51.4, 0.1, 51.6)  # London


def random_points(n: int, bbox=BBOX) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(42)
    lons = rng.uniform(bbox[0], bbox[2], n)
    lats = rng.uniform(bbox[1], bbox[3], n)
    return gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(lons, lats)], crs=4326)


def du(path: Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def bench_eoembed(pts, year):
    import eoembed

    t0 = time.perf_counter()
    table = eoembed.sample_points("tessera", pts, year=year)
    dt = time.perf_counter() - t0
    return dt, table.memory_usage(deep=True).sum(), len(table)


def bench_naive(pts, year):
    """Download-everything baseline using geotessera directly."""
    from geotessera import GeoTessera
    from eoembed.sample import sample_cube_at_points
    import rioxarray
    from rioxarray.merge import merge_arrays

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        gt = GeoTessera()
        tiles = gt.registry.load_blocks_for_region(bounds=BBOX, year=year)
        files = gt.export_embedding_geotiffs(tiles, output_dir=tmp)
        size = du(Path(tmp))
        arrays = [rioxarray.open_rasterio(f) for f in files]
        cube = arrays[0] if len(arrays) == 1 else merge_arrays(arrays)
        table = sample_cube_at_points(cube, pts)
    return time.perf_counter() - t0, size, len(table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=5000)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--skip-naive", action="store_true")
    args = ap.parse_args()

    pts = random_points(args.n_points)
    rows = []

    dt, footprint, n = bench_eoembed(pts, args.year)
    rows.append(("eoembed (Zarr stream)", dt, footprint, n))

    if not args.skip_naive:
        dt, footprint, n = bench_naive(pts, args.year)
        rows.append(("naive (download tiles)", dt, footprint, n))

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"\nAOI={BBOX}  points={args.n_points}  year={args.year}  peakRSS={peak_mb:.0f} MB\n")
    print(f"{'method':<26}{'wall time':>12}{'data footprint':>18}{'rows':>8}")
    for name, dt, size, n in rows:
        print(f"{name:<26}{dt:>10.1f} s{size / 1e6:>14.1f} MB{n:>8}")


if __name__ == "__main__":
    main()
