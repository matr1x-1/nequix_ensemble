"""GPU-only tests for the OpenEquivariance kernel path, single- and multi-device.

These are skipped without a CUDA device and OpenEquivariance; run them on a
GPU node (e.g. via scripts/bench_kernel.sbatch).
"""

import os

import ase
import jax
import numpy as np
import pytest

from nequix.calculator import NequixCalculator
from nequix.model import Nequix, save_model


def _oeq_available():
    # mirror the import logic of NequixConvolution: openequivariance imports
    # torch unless OEQ_NOTORCH is set
    try:
        import torch  # noqa: F401
    except ImportError:
        os.environ["OEQ_NOTORCH"] = "1"
    try:
        import openequivariance  # noqa: F401
        import openequivariance_extjax  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    jax.default_backend() != "gpu" or not _oeq_available(),
    reason="needs a CUDA device and OpenEquivariance",
)

CUTOFF = 4.0
ATOMIC_NUMBERS = [1, 6, 8]


def periodic_atoms(n_atoms=40, seed=0):
    rng = np.random.default_rng(seed)
    return ase.Atoms(
        numbers=rng.choice(ATOMIC_NUMBERS, size=n_atoms),
        positions=rng.uniform(0.0, 8.0, size=(n_atoms, 3)),
        cell=np.eye(3) * 8.0,
        pbc=True,
    )


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    config = {
        "cutoff": CUTOFF,
        "atomic_numbers": ATOMIC_NUMBERS,
        "hidden_irreps": "8x0e+4x1o+2x2e",
        "lmax": 2,
        "n_layers": 3,
        "radial_basis_size": 4,
        "radial_mlp_size": 8,
        "radial_mlp_layers": 2,
        "radial_polynomial_p": 2.0,
        "mlp_init_scale": 4.0,
        "index_weights": True,
        "layer_norm": True,
        "shift": 0.05,
        "scale": 1.3,
        "avg_n_neighbors": 10.0,
        "atom_energies": dict(zip(ATOMIC_NUMBERS, [0.1, 0.2, 0.3])),
    }
    model = Nequix(
        jax.random.key(0),
        n_species=len(ATOMIC_NUMBERS),
        lmax=config["lmax"],
        cutoff=CUTOFF,
        hidden_irreps=config["hidden_irreps"],
        n_layers=config["n_layers"],
        radial_basis_size=config["radial_basis_size"],
        radial_mlp_size=config["radial_mlp_size"],
        radial_mlp_layers=config["radial_mlp_layers"],
        avg_n_neighbors=config["avg_n_neighbors"],
        atom_energies=[0.1, 0.2, 0.3],
        shift=config["shift"],
        scale=config["scale"],
        layer_norm=config["layer_norm"],
    )
    path = tmp_path_factory.mktemp("model") / "small.nqx"
    save_model(path, model, config)
    return str(path)


def _results(model_path, atoms, **kwargs):
    a = atoms.copy()
    a.calc = NequixCalculator(model_path=model_path, **kwargs)
    return a.get_potential_energy(), a.get_forces(), a.get_stress()

def assert_close(test, reference):
    # OpenEquivariance uses atomic adds (deterministic=False), so tolerances
    # are looser than for the pure e3nn path
    np.testing.assert_allclose(test[0], reference[0], rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(test[1], reference[1], rtol=1e-3, atol=5e-4)
    np.testing.assert_allclose(test[2], reference[2], rtol=1e-3, atol=1e-5)


def test_kernel_single_device(model_path):
    atoms = periodic_atoms()
    reference = _results(model_path, atoms, use_kernel=False)
    kernel = _results(model_path, atoms, use_kernel=True)
    assert_close(kernel, reference)


@pytest.mark.skipif(len(jax.devices()) < 2, reason="needs more than one device")
def test_kernel_sharded(model_path):
    n_devices = min(4, len(jax.devices()))
    atoms = periodic_atoms()
    reference = _results(model_path, atoms, use_kernel=False)
    sharded = _results(model_path, atoms, use_kernel=True, n_devices=n_devices)
    assert_close(sharded, reference)


@pytest.mark.skipif(len(jax.devices()) < 2, reason="needs more than one device")
def test_kernel_sharded_md(model_path):
    from ase import units
    from ase.md.verlet import VelocityVerlet

    n_devices = min(4, len(jax.devices()))
    trajectories = []
    for kwargs in [dict(use_kernel=False), dict(use_kernel=True, n_devices=n_devices)]:
        atoms = periodic_atoms(n_atoms=20, seed=1)
        atoms.calc = NequixCalculator(model_path=model_path, **kwargs)
        dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)
        dyn.run(5)
        trajectories.append(atoms.get_positions())

    np.testing.assert_allclose(trajectories[1], trajectories[0], rtol=1e-3, atol=1e-4)
