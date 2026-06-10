"""Abstract interface every embedding source implements.

The contract is intentionally tiny so adding a new provider (Clay, Earth Genome,
Copernicus embeddings, ...) means implementing two methods.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import geopandas as gpd
import pandas as pd
import xarray as xr

BBox = tuple[float, float, float, float]  # lon_min, lat_min, lon_max, lat_max (EPSG:4326)


class EmbeddingSource(ABC):
    """A provider of precomputed EO embeddings."""

    #: short key used in the registry, e.g. "tessera"
    name: str
    #: embedding dimensionality (None if not applicable)
    dims: Optional[int]
    #: native ground sampling distance in metres (None for chip-level products)
    resolution_m: Optional[float]

    @abstractmethod
    def load_cube(self, bbox: BBox, year: int, **kwargs) -> xr.DataArray:
        """Return a *lazy*, dask-backed DataArray with dims ("band", "y", "x").

        Must carry CRS information via the ``rio`` accessor (rioxarray) or a
        ``crs`` attribute. No data should be transferred until ``.compute()``
        or a write triggers it.
        """

    @abstractmethod
    def sample_points(self, points: gpd.GeoDataFrame, year: int, **kwargs) -> pd.DataFrame:
        """Extract embedding vectors at point locations.

        Returns a DataFrame with one row per input point: original columns are
        preserved and embedding values are added as columns ``e000..eNNN``.
        Implementations should be as server-side / chunk-local as possible.
        """

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return f"<EmbeddingSource {self.name} dims={self.dims} res={self.resolution_m}m>"


_REGISTRY: dict[str, type[EmbeddingSource]] = {}


def register(cls: type[EmbeddingSource]) -> type[EmbeddingSource]:
    _REGISTRY[cls.name] = cls
    return cls


def get_source(name: str, **init_kwargs) -> EmbeddingSource:
    """Instantiate a registered source by key ('alphaearth', 'tessera')."""
    # import adapters lazily so optional deps are only needed when used
    if name not in _REGISTRY:
        if name == "alphaearth":
            from . import alphaearth  # noqa: F401
        elif name == "tessera":
            from . import tessera  # noqa: F401
    try:
        return _REGISTRY[name](**init_kwargs)
    except KeyError:
        raise ValueError(
            f"Unknown source '{name}'. Available: {sorted(_REGISTRY)} "
            "(did you install the matching extra, e.g. pip install 'eo-embed-kit[tessera]'?)"
        ) from None
