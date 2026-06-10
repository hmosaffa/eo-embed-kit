"""TESSERA (Univ. of Cambridge) adapter, built on the `geotessera` package.

Preferred path: the **cloud-native Zarr store** (`GeoTesseraZarr`) — embeddings
stream chunk-by-chunk over HTTP, nothing is bulk-downloaded, and point sampling
is built in. Fallback path: per-tile NPY download via the classic `GeoTessera`
client (useful when you *want* a persistent local copy).

Dataset: 128 bands, 10 m, annual; prefer dataset version v1.1 where available
(the v1.0 line is frozen).
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from .base import BBox, EmbeddingSource, register

N_DIMS = 128


@register
class Tessera(EmbeddingSource):
    name = "tessera"
    dims = N_DIMS
    resolution_m = 10.0

    def __init__(self, dataset_version: str = "v1.1", dataset_variant: str | None = "cambridge"):
        self.dataset_version = dataset_version
        self.dataset_variant = dataset_variant
        self._zarr = None
        self._client = None

    # ------------------------------------------------------------ lazy clients
    @property
    def zarr(self):
        if self._zarr is None:
            from geotessera import GeoTesseraZarr

            self._zarr = GeoTesseraZarr()
        return self._zarr

    @property
    def client(self):
        if self._client is None:
            from geotessera import GeoTessera

            kwargs = {"dataset_version": self.dataset_version}
            if self.dataset_variant:
                kwargs["dataset_variant"] = self.dataset_variant
            try:
                self._client = GeoTessera(**kwargs)
            except TypeError:  # older geotessera without variant kwarg
                self._client = GeoTessera(dataset_version=self.dataset_version)
        return self._client

    # ------------------------------------------------------------------ cubes
    def load_cube(self, bbox: BBox, year: int, **kwargs) -> xr.DataArray:
        """Lazy (band, y, x) cube streamed from the Zarr store.

        `read_region` returns the mosaic for the bbox with CRS/transform
        metadata; we wrap it as a dask-chunked DataArray.
        """
        try:
            region = self.zarr.read_region(bbox=bbox, year=year)
            # geotessera returns (data, crs, transform)-style metadata; adapt
            data = np.asarray(region["data"] if isinstance(region, dict) else region)
        except Exception:
            # Fallback: download tiles for the region and mosaic locally.
            data, crs, transform = self._tiles_to_array(bbox, year)
            da = xr.DataArray(
                data, dims=("band", "y", "x"),
                coords={"band": [f"t{i:03d}" for i in range(data.shape[0])]},
            ).chunk({"band": -1, "y": 1024, "x": 1024})
            da.attrs.update(crs=str(crs), transform=tuple(transform), source=self.name)
            return da

        if data.ndim == 3 and data.shape[-1] == N_DIMS:   # (y, x, band) → (band, y, x)
            data = np.moveaxis(data, -1, 0)
        da = xr.DataArray(
            data, dims=("band", "y", "x"),
            coords={"band": [f"t{i:03d}" for i in range(data.shape[0])]},
        ).chunk({"band": -1, "y": 1024, "x": 1024})
        da.attrs["source"] = self.name
        return da

    def _tiles_to_array(self, bbox: BBox, year: int):
        import rioxarray
        from rioxarray.merge import merge_arrays
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tiles = self.client.registry.load_blocks_for_region(bounds=bbox, year=year)
            files = self.client.export_embedding_geotiffs(tiles, output_dir=tmp)
            arrays = [rioxarray.open_rasterio(f, chunks={"x": 1024, "y": 1024}) for f in files]
            merged = arrays[0] if len(arrays) == 1 else merge_arrays(arrays)
            merged = merged.rio.clip_box(*bbox, crs="EPSG:4326")
            return merged.values, merged.rio.crs, merged.rio.transform()

    # ----------------------------------------------------------------- points
    def sample_points(self, points: gpd.GeoDataFrame, year: int, **kwargs) -> pd.DataFrame:
        """Point sampling via the Zarr store's native sampler (chunk-local reads)."""
        pts = points.to_crs(4326).reset_index(drop=True)
        coords = np.column_stack([pts.geometry.x.values, pts.geometry.y.values])
        try:
            vectors = self.zarr.sample_points(coords, year=year)  # (n, 128)
        except Exception:
            from ..sample import sample_cube_at_points

            cube = self.load_cube(tuple(pts.total_bounds), year)
            return sample_cube_at_points(cube, pts)

        emb = pd.DataFrame(
            np.asarray(vectors), columns=[f"e{i:03d}" for i in range(N_DIMS)]
        )
        return pts.drop(columns="geometry").join(emb).assign(
            lon=pts.geometry.x, lat=pts.geometry.y, year=year, source=self.name
        )
