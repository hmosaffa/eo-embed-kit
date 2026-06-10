"""Cache an AOI cube to a local Zarr store — download once, reuse offline.

This is the key HPC pattern: run `cache_aoi` on a login/datamover node (which
has internet), then point all compute jobs at the resulting Zarr on scratch.
Zarr chunks align with Dask chunks, so parallel reads from many workers are
contention-free.
"""
from __future__ import annotations

from pathlib import Path

import xarray as xr


def cache_aoi(
    cube: xr.DataArray,
    path: str | Path,
    chunks: dict | None = None,
    overwrite: bool = False,
) -> xr.DataArray:
    """Write a lazy cube to local Zarr and return the (lazy) cached version.

    If the store already exists it is opened directly — making this safe to
    call unconditionally at the top of a pipeline.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        return open_cache(path)

    chunks = chunks or {"band": -1, "y": 1024, "x": 1024}
    ds = cube.chunk(chunks).to_dataset(name="embedding")
    # carry CRS/transform attrs through the store
    ds.attrs.update(cube.attrs)
    ds.to_zarr(path, mode="w", consolidated=True)
    return open_cache(path)


def open_cache(path: str | Path) -> xr.DataArray:
    ds = xr.open_zarr(path, consolidated=True)
    da = ds["embedding"]
    da.attrs.update(ds.attrs)
    return da


def estimate_size(bbox, source: str = "tessera") -> float:
    """Estimate (in MB) how much data caching this rectangle would download.

    Rough but honest: pixels * dims * 4 bytes at 10 m resolution.
    Call this BEFORE cache_aoi() so a 2 GB download never surprises you.

    >>> eoembed.estimate_size((-0.2, 51.4, 0.1, 51.6), "tessera")
    """
    import math

    from .metadata import METADATA

    dims = METADATA[source]["dims"]
    lon_min, lat_min, lon_max, lat_max = bbox
    mid_lat = math.radians((lat_min + lat_max) / 2)
    width_m = abs(lon_max - lon_min) * 111_320 * math.cos(mid_lat)
    height_m = abs(lat_max - lat_min) * 110_540
    n_pixels = (width_m / 10) * (height_m / 10)
    return n_pixels * dims * 4 / 1e6  # float32, MB
