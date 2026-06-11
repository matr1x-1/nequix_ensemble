import os

# fake multiple devices on CPU for the combined skin + sharding test; must be
# set before jax initializes its backends
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import ase
import jax
import numpy as np
import pytest
from ase import units
from ase.md.verlet import VelocityVerlet

from nequix.calculator import NequixCalculator
from nequix.model import Nequix, save_model

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


def make_calc(model_path, **kwargs):
    return NequixCalculator(model_path=model_path, use_kernel=False, **kwargs)


def test_skin_equivalence(model_path):
    # edges between cutoff and cutoff + skin must contribute exactly zero
    atoms = periodic_atoms()
    results = {}
    for skin in [0.0, 1.0]:
        a = atoms.copy()
        a.calc = make_calc(model_path, skin=skin)
        results[skin] = (a.get_potential_energy(), a.get_forces(), a.get_stress())

    np.testing.assert_allclose(results[1.0][0], results[0.0][0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(results[1.0][1], results[0.0][1], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(results[1.0][2], results[0.0][2], rtol=1e-5, atol=1e-7)


def test_skin_md_equivalence_and_caching(model_path):
    trajectories = {}
    builds = {}
    for skin in [0.0, 0.5]:
        atoms = periodic_atoms(n_atoms=20, seed=1)
        atoms.calc = make_calc(model_path, skin=skin)
        dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)
        dyn.run(10)
        trajectories[skin] = atoms.get_positions()
        builds[skin] = atoms.calc._nl_builds

    np.testing.assert_allclose(trajectories[0.5], trajectories[0.0], rtol=1e-4, atol=1e-5)
    # skin=0 rebuilds every step, skin>0 must reuse the cached neighbor list
    assert builds[0.0] >= 10
    assert builds[0.5] < builds[0.0] / 2


def test_rebuild_triggers(model_path):
    atoms = periodic_atoms()
    calc = make_calc(model_path, skin=0.5)
    atoms.calc = calc

    atoms.get_potential_energy()
    assert calc._nl_builds == 1

    # small move: cached list reused
    atoms.positions[0] += 0.05
    atoms.get_potential_energy()
    assert calc._nl_builds == 1

    # move beyond skin/2: rebuild
    atoms.positions[0] += 0.5
    atoms.get_potential_energy()
    assert calc._nl_builds == 2

    # cell change: rebuild
    atoms.set_cell(atoms.cell * 1.01, scale_atoms=True)
    atoms.get_potential_energy()
    assert calc._nl_builds == 3


def test_cached_results_match_fresh_calculator(model_path):
    # results computed via the cached (positions-only) path must match a
    # freshly built neighbor list at the same positions
    atoms = periodic_atoms(n_atoms=20, seed=2)
    atoms.calc = make_calc(model_path, skin=0.5)
    dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)
    dyn.run(5)
    assert atoms.calc._nl_builds < 5

    fresh = atoms.copy()
    fresh.calc = make_calc(model_path, skin=0.0)
    np.testing.assert_allclose(
        atoms.get_potential_energy(), fresh.get_potential_energy(), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(atoms.get_forces(), fresh.get_forces(), rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(len(jax.devices()) < 4, reason="needs 4 (forced host) devices")
def test_skin_with_sharding(model_path):
    trajectories = []
    for kwargs in [dict(skin=0.0), dict(skin=0.5, n_devices=4)]:
        atoms = periodic_atoms(n_atoms=20, seed=3)
        atoms.calc = make_calc(model_path, **kwargs)
        dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)
        dyn.run(5)
        trajectories.append(atoms.get_positions())

    np.testing.assert_allclose(trajectories[1], trajectories[0], rtol=1e-4, atol=1e-5)
