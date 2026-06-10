"""Core search & cross-source primitives.

`similarity_search` — kNN over an AOI cube ("find everything that looks like
this place"). FAISS-accelerated when available.

`paired_sample` — co-located embedding vectors from *two* sources at the same
points. This is the dataset you need for cross-model alignment research
(e.g. learning an AlphaEarth→TESSERA mapping via CCA/Procrustes). eoembed does
NOT ship an `align()` because the latent spaces are independently learned and
naive projection silently corrupts results — but it gives you the paired data
to study it properly.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr


def similarity_search(
    cube: xr.DataArray,
    lon: float,
    lat: float,
    top_k: int = 500,
) -> pd.DataFrame:
    """Top-k most similar pixels to the pixel at (lon, lat), cosine metric.

    AlphaEarth vectors are unit-norm, so dot product == cosine similarity.
    """
    query = cube.sel(x=lon, y=lat, method="nearest").compute().values.astype("float32")
    qn = query / (np.linalg.norm(query) + 1e-12)

    data = cube.compute().values  # (band, y, x) — keep AOIs modest or tile this
    b, h, w = data.shape
    flat = data.reshape(b, -1).T.astype("float32")
    flat_n = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12)

    try:
        import faiss

        index = faiss.IndexFlatIP(b)
        index.add(flat_n)
        scores, idx = index.search(qn[None, :], top_k)
        scores, idx = scores[0], idx[0]
    except ImportError:
        sims = flat_n @ qn
        idx = np.argpartition(-sims, top_k)[:top_k]
        idx = idx[np.argsort(-sims[idx])]
        scores = sims[idx]

    ys, xs = np.unravel_index(idx, (h, w))
    return pd.DataFrame(
        {"x": cube.x.values[xs], "y": cube.y.values[ys], "similarity": scores}
    )


def paired_sample(
    points: gpd.GeoDataFrame,
    year: int,
    source_a: str = "alphaearth",
    source_b: str = "tessera",
    out: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Co-located embeddings from two sources at the same points.

    Returns one row per point with columns ``a_e000.. / b_e000..``. Useful for
    cross-model alignment studies, redundancy analysis, and ensembling.
    """
    from .sources.base import get_source
    from .sample import to_geoparquet

    pts = points.to_crs(4326).reset_index(drop=True)
    ta = get_source(source_a, **kwargs.pop("source_a_kwargs", {})).sample_points(pts, year)
    tb = get_source(source_b, **kwargs.pop("source_b_kwargs", {})).sample_points(pts, year)

    ea = ta.filter(regex=r"^e\d+$").add_prefix("a_")
    eb = tb.filter(regex=r"^e\d+$").add_prefix("b_")
    meta = pts.drop(columns="geometry").assign(
        lon=pts.geometry.x, lat=pts.geometry.y, year=year,
        source_a=source_a, source_b=source_b,
    )
    table = pd.concat([meta, ea, eb], axis=1)
    if out:
        to_geoparquet(table, out)
    return table
