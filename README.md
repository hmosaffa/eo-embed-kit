# eo-embed-kit 🛰️

**Get Earth-observation AI embeddings for your study area without downloading terabytes.**

## What is this?

AI models like **AlphaEarth** (Google DeepMind) and **TESSERA** (Univ. of Cambridge)
have already "looked at" every 10 m patch of Earth and written a numeric summary
of it: 64 or 128 numbers per pixel, called an *embedding*. These summaries are
amazing for mapping (land cover, crops, forests, change detection) you can train
a good classifier with a tiny laptop model instead of a GPU.

The problem: the datasets are **hundreds of terabytes**, and each provider has a
different, confusing access method (Earth Engine, NPY tiles, Zarr stores, cloud
buckets). This package is one simple Python interface for all of them that
transfers **only the data you actually need**.

## Install

```bash
pip install -e ".[tessera]"      # TESSERA only (no account needed start here)
pip install -e ".[alphaearth]"   # AlphaEarth (needs an Earth Engine account)
pip install -e ".[all]"          # everything
```

Many packages get installed that is normal: the geospatial Python stack
(rasterio/GDAL, geopandas, xarray, dask) is heavy, and every serious geo project
uses the same bricks. Note: TESSERA support needs **Python 3.12+**.

## The two ways to get data , READ THIS FIRST

(Both datasets are **free of charge**. The difference between the two ways is only
**how much data gets downloaded** to your computer, your time, disk and bandwidth.)

### Way 1: a few (or many) points → `sample_points` ✅ small download (KB–MB)

You have labelled points (field plots, GPS samples, training labels). This grabs
the embedding **only at those points**. 100,000 points ≈ **50 MB**.

```python
import geopandas as gpd
import eoembed

points = gpd.read_file("my_points.geojson")        # any points with a 'label' column
table = eoembed.sample_points("tessera", points, year=2024, out="train.parquet")
# -> one row per point, columns e000..e127 = the embedding. Ready for sklearn.
```

### Way 2: a whole rectangle → `cache_aoi` ⚠️ large download (can be GBs)

You want **every pixel** in an area (e.g. to produce a wall-to-wall map).
This downloads it all, once, to a local file , later runs are instant and offline.
**It gets big fast**: a 20 × 20 km box ≈ 2 GB. Always check the size first:

```python
bbox = (-0.13, 51.50, -0.11, 51.52)                 # lon_min, lat_min, lon_max, lat_max

print(eoembed.estimate_size(bbox, "tessera"))       # MB , check BEFORE downloading!

cube  = eoembed.load_cube("tessera", bbox=bbox, year=2024)   # lazy, downloads nothing
store = eoembed.cache_aoi(cube, "data/my_area.zarr")         # the actual download
```

**Rule of thumb:** points → Way 1, always. Rectangle → only when you need a map of
every pixel , and even then, train your model on points first (Way 1), and cache
the rectangle only at the end for the final prediction.

A typical project (50 × 50 km, 100k labelled points): downloading the rectangle
is ~12 GB, of which your points touch 0.4%. `sample_points` gets you ~50 MB , 250× less.

## All functions, in plain words

| function | what it does | download size |
|---|---|---|
| `list_sources()` | table of available datasets (dims, resolution, years, license) | nothing |
| `describe("tessera")` | full details of one dataset incl. required citation | nothing |
| `estimate_size(bbox, source)` | how many MB a rectangle would be | nothing |
| `sample_points(source, points, year)` | embeddings **at your points** → small table | a few KB–MB ✅ |
| `load_cube(source, bbox, year)` | open a rectangle *lazily* (no download yet) | nothing until used |
| `cache_aoi(cube, "x.zarr")` | download a rectangle **once** to disk for offline work | can be GBs ⚠️ |
| `similarity_search(store, lon, lat)` | "find places in this area that look like this spot" | uses the cache |
| `paired_sample(points, year)` | AlphaEarth **and** TESSERA values at the same points | a few KB–MB ✅ |
| `hpc.slurm_cluster()` | one line to run on a SLURM supercomputer |  

Model training is deliberately **not** part of the package , embeddings are
designed so 5 lines of sklearn work. See `examples/02_linear_probe_example.py`.

## Examples (in order of difficulty)

```
examples/00_points_only.py        <- START HERE: 3 points, kilobytes downloaded
examples/00_small_rectangle.py    <- small area with size check before download
examples/01_quickstart.py         <- full workflow: points + cache + search + train
examples/02_linear_probe_example.py  <- training & wall-to-wall map prediction
examples/03_alphaearth_points.py  <- AlphaEarth via Earth Engine (one-time setup below)
```

## AlphaEarth one-time setup (Earth Engine)

TESSERA needs nothing. AlphaEarth's recommended path goes through Google Earth
Engine, which needs a free one-time registration:

1. Go to https://earthengine.google.com → **Get Started** → sign in with a Google
   account → register a **Cloud project** for *noncommercial* use (free). Note the
   project id (looks like `ee-yourname`).
2. `pip install -e ".[alphaearth]"`
3. First run opens a browser window asking you to authorize  once per machine.

Then sampling works exactly like TESSERA, with two extra arguments:

```python
table = eoembed.sample_points("alphaearth", points, year=2024,
                              backend="ee", project="ee-yourname")
```

Why bother? With the `ee` backend the extraction runs **inside Google's servers**:
for 10,000 points only ~2.5 MB reaches your computer , nothing else is downloaded
at all. (Commercial use of Earth Engine requires a paid plan; the embeddings data
itself is CC-BY 4.0 either way.)

## Using it on HPC

Compute nodes often have no internet. The pattern:

1. on the **login node**: run `cache_aoi(...)` → local Zarr on scratch;
2. submit jobs that read only the Zarr, fast, offline, quota-friendly;
3. optional: `eoembed.hpc.slurm_cluster(cores=16, memory="64GB", jobs=8)` to scale out.

## Data licences & attribution

This package contains **code only** (MIT) and never redistributes data, it streams
from the providers at run time. Obligations attach to the **data and your derived maps**:

| dataset | licence | what you must do |
|---|---|---|
| AlphaEarth Satellite Embedding V1 | **CC-BY 4.0** | Include: *"The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."* in papers/maps/apps. Commercial use OK. Official GCS bucket is requester-pays; the source.coop/AWS mirror is free. Earth Engine access is also subject to EE Terms of Service. |
| TESSERA embeddings & weights | **CC0** (software MIT) | No legal requirement; the Cambridge team requests citation of the TESSERA paper. Commercial use OK. |

Tip: keep a `DATA_LICENSES.md` in any project that ships derived layers.

## Supported sources

| key | product | dims | res | access |
|---|---|---|---|---|
| `tessera` | Cambridge TESSERA v1/v1.1 | 128 | 10 m | cloud Zarr stream / tile download (no account) |
| `alphaearth` | Google DeepMind Satellite Embedding V1, annual 2017+ | 64 | 10 m | Earth Engine (server-side sampling) or free COG mirror on AWS/source.coop |

## Repo layout

```
src/eoembed/
  __init__.py     # public API
  sample.py       # point extraction -> GeoParquet
  cache.py        # rectangle -> local Zarr + estimate_size
  search.py       # similarity_search, paired_sample
  metadata.py     # list_sources(), describe()
  hpc.py          # Dask SLURM helper
  sources/        # alphaearth.py, tessera.py, base.py
examples/         # 00 -> 02, simplest first
benchmarks/       # measure it yourself; no unmeasured numbers published here
```

## Benchmarks

`python benchmarks/bench_point_extraction.py --n-points 5000 --year 2024`
measures point extraction vs. a naive download-all-tiles baseline. Numbers depend
on your bandwidth, run it and report **your own measurements**.

## Roadmap

- [ ] CI-verified bucket globs against live providers
- [ ] CLI (`eoembed sample ...`)
- [ ] Clay / Copernicus-embeddings adapters
- [ ] *experimental* cross-model alignment learned from `paired_sample()` data
      (not in core on purpose: the latent spaces are independently learned, and a
      naive projection silently corrupts results, needs evaluation first)

## Citations

- Brown, Kazmierski, Pasquarella et al., *AlphaEarth Foundations* (2025)
- Feng et al., *TESSERA* (2025) — https://github.com/ucam-eo/geotessera

License: MIT
