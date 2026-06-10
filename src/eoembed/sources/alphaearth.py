"""AlphaEarth Foundations Satellite Embedding V1 adapter.

Two backends:

* ``backend="ee"``  — Google Earth Engine. Point sampling runs **server-side**
  (terabytes never leave Google's datacenter; you receive a small table).
  Region loading uses ``xee`` so the cube is a lazy xarray.
  Requires an Earth Engine account: https://earthengine.google.com

* ``backend="gcs"`` — Cloud-Optimized GeoTIFFs outside Earth Engine.
  The official bucket ``gs://alphaearth_foundations`` is **requester-pays**
  (egress is billed to *your* GCP project), so this backend defaults to the
  free community mirror on source.coop / AWS open data (``tge-labs/aef``).
  rasterio/rioxarray perform HTTP range-reads, so only your AOI's bytes move.
  Caution: the source.coop COGs are "bottom-up" oriented; QGIS handles this
  but raw GDAL reads may need flipping — verify orientation for your AOI.

Licence: data is CC-BY 4.0 — attribute "The AlphaEarth Foundations Satellite
Embedding dataset is produced by Google and Google DeepMind." in derived work.

Dataset: 64 bands (A00..A63), 10 m, annual 2017+, unit-norm vectors.
Catalog id: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
import xarray as xr

from .base import BBox, EmbeddingSource, register

EE_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
GCS_BUCKET = "gs://alphaearth_foundations"          # official (requester-pays)
AWS_MIRROR = "s3://us-west-2.opendata.source.coop/tge-labs/aef"  # free mirror
N_DIMS = 64
BAND_NAMES = [f"A{i:02d}" for i in range(N_DIMS)]


def _init_ee(project: str | None = None):
    import ee

    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    return ee


@register
class AlphaEarth(EmbeddingSource):
    name = "alphaearth"
    dims = N_DIMS
    resolution_m = 10.0

    def __init__(self, backend: str = "ee", project: str | None = None):
        if backend not in ("ee", "gcs"):
            raise ValueError("backend must be 'ee' or 'gcs'")
        self.backend = backend
        self.project = project

    # ------------------------------------------------------------------ cubes
    def load_cube(self, bbox: BBox, year: int, scale: int = 10, **kwargs) -> xr.DataArray:
        if self.backend == "ee":
            return self._load_cube_ee(bbox, year, scale)
        return self._load_cube_gcs(bbox, year)

    def _load_cube_ee(self, bbox: BBox, year: int, scale: int) -> xr.DataArray:
        ee = _init_ee(self.project)
        import xee  # noqa: F401  (registers the 'ee' xarray engine)

        geom = ee.Geometry.Rectangle(list(bbox))
        ic = (
            ee.ImageCollection(EE_COLLECTION)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .filterBounds(geom)
        )
        ds = xr.open_dataset(
            ic,
            engine="ee",
            geometry=geom,
            scale=scale,                # metres; >10 averages pixels (valid: AEF
            crs="EPSG:4326",            # vectors are unit-norm & linearly separable)
        )
        da = ds.to_array("band").squeeze(drop=True)
        da.attrs["crs"] = "EPSG:4326"
        da.attrs["source"] = self.name
        return da.chunk({"band": -1})

    def _load_cube_gcs(self, bbox: BBox, year: int) -> xr.DataArray:
        """Stream the public COGs — no Earth Engine account needed.

        The bucket is organised by year with COG tiles; we open the tiles
        intersecting the bbox lazily and mosaic them. Tile listing uses gcsfs
        with anonymous access.
        """
        import rioxarray  # noqa: F401
        from rioxarray.merge import merge_arrays
        import fsspec

        # Default: free AWS/source.coop mirror (anonymous). To use the official
        # requester-pays GCS bucket instead, pass project=<billing-project> and
        # set use_official_gcs=True on the instance.
        fs = fsspec.filesystem("s3", anon=True)
        root = AWS_MIRROR.removeprefix("s3://")
        # Layout mirrors GCS: .../satellite_embedding/v1/annual/<year>/<utm_zone>/*.tif
        candidates = fs.glob(f"{root}/satellite_embedding/v1/annual/{year}/*/*.tif")
        if not candidates:
            raise FileNotFoundError(
                f"No COGs found for {year} under {AWS_MIRROR}. "
                "Inspect the bucket layout (it may have changed) and update the glob."
            )
        # TODO: pre-filter tiles by UTM zone intersecting bbox instead of
        # opening every candidate (cheap win: parse zone from the path).
        tiles = []
        for path in candidates:
            url = f"s3://{path}"
            da = rioxarray.open_rasterio(url, chunks={"x": 1024, "y": 1024}, masked=True)
            sub = da.rio.clip_box(*bbox, crs="EPSG:4326", allow_one_dimensional_raster=True)
            if sub.size:
                tiles.append(sub)
        if not tiles:
            raise FileNotFoundError("No COG tiles intersect the requested bbox.")
        cube = tiles[0] if len(tiles) == 1 else merge_arrays(tiles)
        cube = cube.assign_coords(band=("band", BAND_NAMES[: cube.sizes["band"]]))
        cube.attrs["source"] = self.name
        return cube

    # ----------------------------------------------------------------- points
    def sample_points(
        self,
        points: gpd.GeoDataFrame,
        year: int,
        scale: int = 10,
        batch: int = 4000,
        **kwargs,
    ) -> pd.DataFrame:
        """Server-side sampling on Earth Engine (recommended for AlphaEarth).

        For n points this transfers ~n*64 floats and nothing else.
        """
        if self.backend != "ee":
            # fall back to generic chunk-local sampling on the GCS cube
            from ..sample import sample_cube_at_points

            cube = self.load_cube(tuple(points.total_bounds), year)
            return sample_cube_at_points(cube, points)

        ee = _init_ee(self.project)
        pts = points.to_crs(4326).reset_index(drop=True)
        img = (
            ee.ImageCollection(EE_COLLECTION)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .mosaic()
        )

        frames: list[pd.DataFrame] = []
        for start in range(0, len(pts), batch):
            chunk = pts.iloc[start : start + batch]
            features = [
                ee.Feature(ee.Geometry.Point([geom.x, geom.y]), {"idx": int(i)})
                for i, geom in zip(chunk.index, chunk.geometry)
            ]
            fc = ee.FeatureCollection(features)
            sampled = img.sampleRegions(collection=fc, scale=scale, geometries=False)
            rows = sampled.getInfo()["features"]  # TODO: switch to getDownloadURL / high-volume endpoint for >100k pts
            frames.append(pd.DataFrame([f["properties"] for f in rows]))
        emb = pd.concat(frames, ignore_index=True).set_index("idx").sort_index()
        emb = emb.rename(columns={b: f"e{i:03d}" for i, b in enumerate(BAND_NAMES)})
        return pts.drop(columns="geometry").join(emb).assign(
            lon=pts.geometry.x, lat=pts.geometry.y, year=year, source=self.name
        )
