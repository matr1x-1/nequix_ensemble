import os

# fake multiple devices on CPU so the sharded code path can be tested without
# GPUs; must be set before jax initializes its backends
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import ase
import ase.build
import jax
import jraph
import numpy as np
import pytest

from nequix.calculator import NequixCalculator
from nequix.data import dict_to_graphstuple, preprocess_graph
from nequix.distributed import get_mesh
from nequix.model import Nequix, save_model

if len(jax.devices()) < 8:
    pytest.skip(
        "needs 8 (forced host) devices; run this file in its own pytest process",
        allow_module_level=True,
    )

CUTOFF = 4.0
ATOMIC_NUMBERS = [1, 6, 8]
ATOM_INDICES = {n: i for i, n in enumerate(ATOMIC_NUMBERS)}


def small_model(key=None, n_layers=3):
    # 3 layers so gradients flow through multiple cross-device reductions
    return Nequix(
        key if key is not None else jax.random.key(0),
        n_species=len(ATOMIC_NUMBERS),
        lmax=2,
        cutoff=CUTOFF,
        hidden_irreps="8x0e+4x1o+2x2e",
        n_layers=n_layers,
        radial_basis_size=4,
        radial_mlp_size=8,
        radial_mlp_layers=2,
        avg_n_neighbors=10.0,
        atom_energies=[0.1, 0.2, 0.3],
        shift=0.05,
        scale=1.3,
        layer_norm=True,
    )


def periodic_atoms(n_atoms=40, seed=0):
    rng = np.random.default_rng(seed)
    return ase.Atoms(
        numbers=rng.choice(ATOMIC_NUMBERS, size=n_atoms),
        positions=rng.uniform(0.0, 8.0, size=(n_atoms, 3)),
        cell=np.eye(3) * 8.0,
        pbc=True,
    )


def atoms_to_padded_graph(atoms, n_devices, n_graph=2):
    graph = dict_to_graphstuple(preprocess_graph(atoms, ATOM_INDICES, CUTOFF, False))
    total_nodes = int(np.sum(graph.n_node))
    total_edges = int(np.sum(graph.n_edge))
    n_node = total_nodes + 8 - total_nodes % 8
    multiple = 8 * n_devices
    n_edge = ((total_edges + multiple) // multiple) * multiple
    return jraph.pad_with_graphs(graph, n_node=n_node, n_edge=n_edge, n_graph=n_graph)


def assert_outputs_match(model, graph, n_devices):
    energy_1, forces_1, stress_1 = model(graph)
    mesh = get_mesh(n_devices)
    energy_n, forces_n, stress_n = model(graph, mesh=mesh)

    np.testing.assert_allclose(energy_n, energy_1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(forces_n, forces_1, rtol=1e-4, atol=1e-5)
    if stress_1 is None:
        assert stress_n is None
    else:
        np.testing.assert_allclose(stress_n, stress_1, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("n_devices", [2, 8])
def test_sharded_periodic(n_devices):
    model = small_model()
    graph = atoms_to_padded_graph(periodic_atoms(), n_devices)
    assert_outputs_match(model, graph, n_devices)


def test_sharded_molecule():
    # no cell: exercises the cell=None branch
    model = small_model()
    atoms = ase.build.molecule("CH3CH2OH")
    graph = atoms_to_padded_graph(atoms, 4)
    assert graph.globals["cell"] is None
    assert_outputs_match(model, graph, 4)


def test_sharded_batched():
    # multiple graphs in a batch: exercises per-edge graph indexing for the strain
    model = small_model()
    graphs = [
        dict_to_graphstuple(preprocess_graph(periodic_atoms(seed=i), ATOM_INDICES, CUTOFF, False))
        for i in range(3)
    ]
    graph = jraph.batch_np(graphs)
    total_nodes = int(np.sum(graph.n_node))
    total_edges = int(np.sum(graph.n_edge))
    graph = jraph.pad_with_graphs(
        graph,
        n_node=total_nodes + 8 - total_nodes % 8,
        n_edge=((total_edges + 32) // 32) * 32,
        n_graph=4,
    )
    assert_outputs_match(model, graph, 4)


def test_sharded_uneven_nodes():
    # node count not divisible by the number of devices
    model = small_model()
    graph = atoms_to_padded_graph(periodic_atoms(n_atoms=37), 3)
    assert graph.nodes["positions"].shape[0] % 3 != 0
    assert_outputs_match(model, graph, 3)


def test_sharded_edge_divisibility_error():
    model = small_model()
    graph = dict_to_graphstuple(preprocess_graph(periodic_atoms(), ATOM_INDICES, CUTOFF, False))
    total_nodes = int(np.sum(graph.n_node))
    total_edges = int(np.sum(graph.n_edge))
    n_edge = total_edges + 1 if (total_edges + 1) % 3 else total_edges + 2
    graph = jraph.pad_with_graphs(graph, n_node=total_nodes + 1, n_edge=n_edge, n_graph=2)
    assert graph.senders.shape[0] % 3 != 0
    with pytest.raises(ValueError, match="divisible"):
        model(graph, mesh=get_mesh(3))


def _save_small_model(tmp_path):
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
    path = tmp_path / "small.nqx"
    save_model(path, small_model(), config)
    return path


def test_calculator_sharded(tmp_path):
    path = _save_small_model(tmp_path)
    calc_1 = NequixCalculator(model_path=str(path), use_kernel=False)
    calc_n = NequixCalculator(model_path=str(path), use_kernel=False, n_devices=4)

    atoms = periodic_atoms()
    atoms_1 = atoms.copy()
    atoms_1.calc = calc_1
    atoms_n = atoms.copy()
    atoms_n.calc = calc_n

    np.testing.assert_allclose(
        atoms_n.get_potential_energy(), atoms_1.get_potential_energy(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(atoms_n.get_forces(), atoms_1.get_forces(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(atoms_n.get_stress(), atoms_1.get_stress(), rtol=1e-4, atol=1e-6)


def test_calculator_sharded_md(tmp_path):
    from ase import units
    from ase.md.verlet import VelocityVerlet

    path = _save_small_model(tmp_path)
    trajectories = []
    for n_devices in [None, 4]:
        atoms = periodic_atoms(n_atoms=20, seed=1)
        atoms.calc = NequixCalculator(model_path=str(path), use_kernel=False, n_devices=n_devices)
        dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)
        dyn.run(5)
        trajectories.append(atoms.get_positions())

    np.testing.assert_allclose(trajectories[1], trajectories[0], rtol=1e-4, atol=1e-5)
