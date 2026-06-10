"""Static metadata registry for embedding sources.

>>> import eoembed
>>> eoembed.list_sources()          # summary DataFrame
>>> eoembed.describe("alphaearth")  # full metadata dict
"""
from __future__ import annotations

import pandas as pd

METADATA: dict[str, dict] = {
    "alphaearth": {
        "product": "AlphaEarth Foundations Satellite Embedding V1 (annual)",
        "producer": "Google / Google DeepMind",
        "dims": 64,
        "resolution": "10 m",
        "years": "2017-2025 (annual, ongoing)",
        "granularity": "per-pixel",
        "license": "CC-BY 4.0",
        "attribution": (
            "The AlphaEarth Foundations Satellite Embedding dataset is "
            "produced by Google and Google DeepMind."
        ),
        "citation": (
            "Brown, Kazmierski, Pasquarella et al. (2025). AlphaEarth "
            "Foundations: An embedding field model for accurate and "
            "efficient global mapping from sparse label data."
        ),
        "backends": ["ee (Earth Engine, server-side sampling)",
                     "gcs (COG mirror on AWS/source.coop; official GCS bucket is requester-pays)"],
        "notes": "Vectors are unit-norm; coarser scales remain valid by averaging. "
                 "EE access subject to Earth Engine Terms of Service.",
        "homepage": "https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL",
    },
    "tessera": {
        "product": "TESSERA global embeddings (v1 / v1.1)",
        "producer": "University of Cambridge (EEG)",
        "dims": 128,
        "resolution": "10 m",
        "years": "2017-present (coverage growing; v1.0 line frozen)",
        "granularity": "per-pixel",
        "license": "CC0 (embeddings & weights); MIT (software). Attribution requested.",
        "attribution": "TESSERA, University of Cambridge (attribution requested, not required).",
        "citation": (
            "Feng et al. (2025). TESSERA: Temporal Embeddings of Surface "
            "Spectra for Earth Representation and Analysis."
        ),
        "backends": ["zarr (cloud-native streaming)", "tiles (NPY/GeoTIFF download via geotessera)"],
        "notes": "Never mix dataset versions/variants in one downstream task: "
                 "feature spaces are independently learned and not interchangeable.",
        "homepage": "https://github.com/ucam-eo/geotessera",
    },
}


def list_sources() -> pd.DataFrame:
    """Summary table of all supported embedding sources."""
    rows = [
        {
            "source": key,
            "dims": m["dims"],
            "resolution": m["resolution"],
            "years": m["years"],
            "license": m["license"].split(" (")[0].split(";")[0],
            "backends": ", ".join(b.split(" ")[0] for b in m["backends"]),
        }
        for key, m in METADATA.items()
    ]
    return pd.DataFrame(rows).set_index("source")


def describe(source: str) -> dict:
    """Full metadata for one source (dims, license, citation, backends, ...)."""
    try:
        return dict(METADATA[source])
    except KeyError:
        raise ValueError(f"Unknown source '{source}'. Available: {sorted(METADATA)}") from None
