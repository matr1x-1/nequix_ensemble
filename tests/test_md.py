import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import ase
import jax
import numpy as np
import pytest
from ase import units
from ase.constraints import FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

pytest.importorskip("jax_md")

from nequix.calculator import NequixCalculator
from nequix.md import run_md
from nequix.model import Nequix, save_model

CUTOFF = 4.0
ATOMIC_NUMBERS = [1, 6, 8]


def periodic_atoms(seed=0, temperature_K=300):
    # lattice + rattle: uniform-random positions can overlap, and the huge
    # forces make any two engines diverge through fp32 chaos within steps
    rng = np.random.default_rng(seed)
    pts = (
        np.array([[i, j, k] for i in range(3) for j in range(3) for k in range(3)], dtype=float)
        * 2.2 + 0.6
    )
    pts += rng.normal(0.0, 0.05, pts.shape)
    atoms = ase.Atoms(
        numbers=rng.choice(ATOMIC_NUMBERS, size=len(pts)),
        positions=pts,
        cell=np.eye(3) * 6.6,
        pbc=True,
    )
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=np.random.default_rng(7))
    return atoms


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    config = {
        "cutoff": CUTOFF,
        "atomic_numbers": ATOMIC_NUMBERS,
        "hidden_irreps": "8x0e+4x1o+2x2e",
        "lmax": 2,
        "n_layers": 2,
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


def ase_nve(atoms, calc, n_steps, dt):
    atoms.calc = calc
    VelocityVerlet(atoms, timestep=dt).run(n_steps)
    return atoms


def test_nve_matches_ase_step_for_step(model_path):
    dt = 0.5 * units.fs
    ref = periodic_atoms(seed=1)
    ase_nve(ref, make_calc(model_path, skin=0.5), 16, dt)

    atoms = periodic_atoms(seed=1)
    calc = make_calc(model_path, skin=0.5)
    run_md(atoms, calc, 16, timestep=dt, chunk_size=8)

    np.testing.assert_allclose(atoms.get_positions(), ref.get_positions(), atol=1e-4)
    np.testing.assert_allclose(atoms.get_momenta(), ref.get_momenta(), atol=1e-3)


def test_nve_remainder_steps(model_path):
    # n_steps not a multiple of chunk_size exercises the jitted tail path
    dt = 0.5 * units.fs
    ref = periodic_atoms(seed=2)
    ase_nve(ref, make_calc(model_path, skin=0.5), 11, dt)

    atoms = periodic_atoms(seed=2)
    run_md(atoms, make_calc(model_path, skin=0.5), 11, timestep=dt, chunk_size=4)
    np.testing.assert_allclose(atoms.get_positions(), ref.get_positions(), atol=1e-4)


def test_nve_with_rebuilds_matches_ase(model_path):
    # small skin + hot atoms forces mid-run neighbor-list rebuilds
    dt = 1.0 * units.fs
    ref = periodic_atoms(seed=3, temperature_K=600)
    ase_nve(ref, make_calc(model_path, skin=0.4), 24, dt)

    atoms = periodic_atoms(seed=3, temperature_K=600)
    calc = make_calc(model_path, skin=0.4)
    run_md(atoms, calc, 24, timestep=dt, chunk_size=8)
    np.testing.assert_allclose(atoms.get_positions(), ref.get_positions(), atol=1e-4)


def test_frozen_atoms_match_ase(model_path):
    dt = 0.5 * units.fs
    frozen = [0, 1, 2, 3]

    ref = periodic_atoms(seed=4)
    ref.set_constraint(FixAtoms(indices=frozen))
    ase_nve(ref, make_calc(model_path, skin=0.5), 16, dt)

    atoms = periodic_atoms(seed=4)
    atoms.set_constraint(FixAtoms(indices=frozen))
    start = atoms.get_positions().copy()
    run_md(atoms, make_calc(model_path, skin=0.5), 16, timestep=dt, chunk_size=8)

    np.testing.assert_allclose(atoms.get_positions()[frozen], start[frozen], atol=1e-7)
    np.testing.assert_allclose(atoms.get_positions(), ref.get_positions(), atol=1e-4)
    assert np.abs(atoms.get_momenta()[frozen]).max() == 0.0


def test_nvt_langevin_runs_and_thermalizes(model_path):
    atoms = periodic_atoms(seed=5)
    calc = make_calc(model_path, skin=0.5)
    sps, temp, info = run_md(
        atoms, calc, 48, timestep=0.5 * units.fs,
        temperature_K=300, friction=0.05 / units.fs, chunk_size=8, seed=3,
    )
    assert np.isfinite(atoms.get_positions()).all()
    assert np.isfinite(temp) and 10.0 < temp < 3000.0


def test_calculator_consistent_after_run(model_path):
    # the calculator must produce correct results on the final state (its
    # neighbor-list cache is invalidated by run_md)
    atoms = periodic_atoms(seed=6)
    calc = make_calc(model_path, skin=0.5)
    run_md(atoms, calc, 8, timestep=0.5 * units.fs, chunk_size=8)
    atoms.calc = calc
    e = atoms.get_potential_energy()

    fresh = atoms.copy()
    fresh.calc = make_calc(model_path, skin=0.0)
    np.testing.assert_allclose(e, fresh.get_potential_energy(), rtol=1e-6, atol=1e-6)
