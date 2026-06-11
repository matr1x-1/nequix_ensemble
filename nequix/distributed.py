"""Utilities for multi-device (SPMD) inference.

Multi-device inference shards the *edges* of a graph across devices while
replicating the node arrays and the model weights on every device (the model
is small; the per-edge tensor products dominate the cost). Each device
computes messages for its edge shard and the partial per-node aggregates are
summed across devices with one all-reduce per layer (see
``NequixConvolution.__call__``).
"""

import functools
import inspect
from typing import Optional, Union

import jax
import numpy as np

# name of the device axis in the mesh used for inference
AXIS_NAME = "gpu"


def get_mesh(n_devices: Optional[Union[int, str]] = None) -> jax.sharding.Mesh:
    """Create a 1D device mesh over the first ``n_devices`` local devices.

    Args:
        n_devices: number of devices to use, or None/"all" for all available.
    """
    devices = jax.devices()
    if n_devices is None or n_devices == "all":
        n = len(devices)
    else:
        n = int(n_devices)
    if not 1 <= n <= len(devices):
        raise ValueError(f"requested {n} devices but {len(devices)} are available")
    return jax.sharding.Mesh(np.array(devices[:n]), (AXIS_NAME,))


def shard_map_no_check(f, mesh, in_specs, out_specs):
    """``jax.shard_map`` across jax versions, with replication checking disabled.

    Replication checking ("check_rep"/"check_vma") does not support custom_vjp
    functions (such as all_reduce_sum below, or the openequivariance kernel) in
    all jax versions, so we disable it; correctness of the collectives'
    gradients is guaranteed by the explicit custom_vjp rules instead.
    """
    try:
        from jax import shard_map as _shard_map
    except ImportError:
        from jax.experimental.shard_map import shard_map as _shard_map

    kwargs = {}
    params = inspect.signature(_shard_map).parameters
    if "check_vma" in params:
        kwargs["check_vma"] = False
    elif "check_rep" in params:
        kwargs["check_rep"] = False
    return _shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **kwargs)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def all_reduce_sum(x: jax.Array, axis_name: str) -> jax.Array:
    """Sum ``x`` across all devices of ``axis_name`` (replicated result).

    Same as ``jax.lax.psum`` but with an explicit gradient rule: the cotangents
    are summed across devices as well, which is the correct transpose for
    device-varying cotangents (each device sees the gradient of its *own*
    partial objective; the sum collects the contributions of all of them).
    This does not rely on shard_map's replication tracking, so it stays correct
    under check_rep=False.
    """
    return jax.lax.psum(x, axis_name)


def _all_reduce_sum_fwd(x, axis_name):
    return jax.lax.psum(x, axis_name), None


def _all_reduce_sum_bwd(axis_name, _residual, ct):
    return (jax.lax.psum(ct, axis_name),)


all_reduce_sum.defvjp(_all_reduce_sum_fwd, _all_reduce_sum_bwd)
