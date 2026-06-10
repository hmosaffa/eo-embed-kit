"""Generic, chunk-local point extraction and GeoParquet persistence."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr


def sample_cube_at_points(cube: xr.DataArray, points: gpd.GeoDataFrame) -> pd.DataFrame:
    """Nearest-pixel lookup on a (band, y, x) cube using vectorised xarray
    indexing. Dask-backed cubes only materialise the chunks the points touch.
    """
    crs = cube.attrs.get("crs") or getattr(getattr(cube, "rio", None), "crs", None)
    pts = points.to_crs(crs) if crs else points
    pts = pts.reset_index(drop=True)

    xs = xr.DataArray(pts.geometry.x.values, dims="points")
    ys = xr.DataArray(pts.geometry.y.values, dims="points")
    sampled = cube.sel(x=xs, y=ys, method="nearest").transpose("points", "band")
    values = np.asarray(sampled.compute())

    emb = pd.DataFrame(values, columns=[f"e{i:03d}" for i in range(values.shape[1])])
    out = pts.drop(columns="geometry").join(emb)
    out["lon"] = points.to_crs(4326).geometry.x.values
    out["lat"] = points.to_crs(4326).geometry.y.values
    return out


def to_geoparquet(df: pd.DataFrame, path: str) -> str:
    """Persist a feature table as GeoParquet (point geometry from lon/lat)."""
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    )
    gdf.to_parquet(path)
    return path
