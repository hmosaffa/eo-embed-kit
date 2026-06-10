"""One-line Dask clusters for HPC schedulers.

Typical pattern on a SLURM machine:

    from eoembed.hpc import slurm_cluster
    client = slurm_cluster(cores=16, memory="64GB", jobs=8)

Every dask-backed eoembed operation (cube reads, predict_cube, cache writes)
then runs on the allocation transparently. Remember: compute nodes frequently
have no internet — stage data first with `eoembed.cache_aoi` on a login node.
"""
from __future__ import annotations


def slurm_cluster(
    cores: int = 8,
    memory: str = "32GB",
    walltime: str = "02:00:00",
    jobs: int = 4,
    queue: str | None = None,
    account: str | None = None,
    **kwargs,
):
    from dask_jobqueue import SLURMCluster
    from distributed import Client

    cluster = SLURMCluster(
        cores=cores,
        memory=memory,
        walltime=walltime,
        queue=queue,
        account=account,
        **kwargs,
    )
    cluster.scale(jobs=jobs)
    return Client(cluster)


def local_cluster(n_workers: int | None = None, memory_limit: str = "auto"):
    """Same API on a laptop/workstation."""
    from distributed import Client, LocalCluster

    return Client(LocalCluster(n_workers=n_workers, memory_limit=memory_limit))
