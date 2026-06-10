"""EXAMPLE (not part of the core package): downstream ML on embedding tables.

The core of eo-embed-kit is data access (sample / cache / search). Training is
deliberately left to you — this file shows how little code it takes.

EO foundation-model embeddings are engineered so that *simple* models work:
a logistic regression / small MLP on the 64–128 dims typically matches a
fine-tuned deep model for classification, at 1/1000th the compute.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def _embedding_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("e") and c[1:].isdigit()]


class LinearProbe:
    """Logistic-regression probe with standardisation, CV report, and
    low-memory chunked raster prediction."""

    def __init__(self, **logreg_kwargs):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        logreg_kwargs.setdefault("max_iter", 2000)
        self.model = make_pipeline(StandardScaler(), LogisticRegression(**logreg_kwargs))
        self._report: str | None = None

    def fit(self, table: pd.DataFrame, label_col: str = "label") -> "LinearProbe":
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import classification_report

        cols = _embedding_cols(table)
        X = table[cols].to_numpy(dtype="float32")
        y = table[label_col].to_numpy()
        y_pred = cross_val_predict(self.model, X, y, cv=5)
        self._report = classification_report(y, y_pred)
        self.model.fit(X, y)
        self._classes = self.model.classes_
        return self

    def report(self) -> str:
        return self._report or "fit() has not been called yet"

    def predict_cube(self, cube: xr.DataArray, out: str | None = None) -> xr.DataArray:
        """Wall-to-wall prediction, one Dask chunk at a time (constant memory)."""
        classes = {c: i for i, c in enumerate(self._classes)}

        def _block(block: np.ndarray) -> np.ndarray:
            b, h, w = block.shape
            flat = block.reshape(b, -1).T            # (pixels, bands)
            ok = np.isfinite(flat).all(axis=1)
            pred = np.full(flat.shape[0], -1, dtype="int16")
            if ok.any():
                labels = self.model.predict(flat[ok])
                pred[ok] = np.vectorize(classes.get)(labels)
            return pred.reshape(1, h, w)

        result = xr.apply_ufunc(
            _block,
            cube,
            input_core_dims=[["band", "y", "x"]],
            output_core_dims=[["cls", "y", "x"]],
            dask="parallelized",
            dask_gufunc_kwargs={"output_sizes": {"cls": 1}},
            output_dtypes=["int16"],
        ).squeeze("cls", drop=True)
        result.attrs.update(cube.attrs, classes=list(map(str, self._classes)))

        if out:
            import rioxarray  # noqa: F401

            r = result.rio.write_crs(cube.attrs.get("crs", "EPSG:4326"))
            r.rio.to_raster(out, compress="deflate", tiled=True)
        return result


def similarity_search(
    cube: xr.DataArray,
    lon: float,
    lat: float,
    top_k: int = 500,
    metric: str = "cosine",
) -> pd.DataFrame:
    """Find the top-k pixels most similar to the pixel at (lon, lat).

    Uses FAISS when available; falls back to a NumPy dot-product scan.
    AlphaEarth vectors are unit-norm, so dot product == cosine similarity.
    """
    query = cube.sel(x=lon, y=lat, method="nearest").compute().values.astype("float32")
    qn = query / (np.linalg.norm(query) + 1e-12)

    data = cube.compute().values  # (band, y, x) — keep AOIs modest or chunk this
    b, h, w = data.shape
    flat = data.reshape(b, -1).T.astype("float32")
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12
    flat_n = flat / norms

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
        {
            "x": cube.x.values[xs],
            "y": cube.y.values[ys],
            "similarity": scores,
        }
    )
