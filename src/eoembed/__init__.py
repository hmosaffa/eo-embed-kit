"""eo-embed-kit — unified, lazy, HPC-friendly access to EO foundation-model embeddings."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
import xarray as xr

from .cache import cache_aoi, estimate_size, open_cache
from .metadata import describe, list_sources
from .sample import sample_cube_at_points, to_geoparquet
from .search import paired_sample, similarity_search
from .sources.base import BBox, get_source

__version__ = "0.1.0"
__all__ = [
    # core verbs
    "sample_points",
    "cache_aoi",
    "open_cache",
    "estimate_size",
    "similarity_search",
    "paired_sample",
    # metadata
    "list_sources",
    "describe",
    # supporting
    "load_cube",
    "get_source",
    "to_geoparquet",
    "sample_cube_at_points",
]


def load_cube(source: str, bbox: BBox, year: int, **kwargs) -> xr.DataArray:
    """Lazy (band, y, x) embedding cube for an AOI from any registered source.

    >>> cube = load_cube("tessera", bbox=(-0.2, 51.4, 0.1, 51.6), year=2024)
    """
    init = {k: kwargs.pop(k) for k in ("backend", "project", "dataset_version", "dataset_variant") if k in kwargs}
    return get_source(source, **init).load_cube(bbox, year, **kwargs)


def sample_points(
    source: str,
    points: gpd.GeoDataFrame,
    year: int,
    out: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Extract embedding vectors at points; optionally persist as GeoParquet."""
    init = {k: kwargs.pop(k) for k in ("backend", "project", "dataset_version", "dataset_variant") if k in kwargs}
    table = get_source(source, **init).sample_points(points, year, **kwargs)
    if out:
        to_geoparquet(table, out)
    return table
