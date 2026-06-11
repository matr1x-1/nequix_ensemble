"""Benchmark (and validate) multi-device inference against single-device.

Example:
    python scripts/bench_distributed.py --model-path models/nequix-mp-1.nqx \
        --size 8 --n-devices 4 --steps 20 --check
"""

import argparse
import time

import ase.build
import numpy as np
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from nequix.calculator import NequixCalculator


def make_atoms(size: int):
    atoms = ase.build.bulk("Cu", cubic=True).repeat((size, size, size))
    atoms.rattle(0.02, seed=42)
    return atoms


def check_equivalence(model_path: str, n_devices: int, size: int = 4):
    atoms = make_atoms(size)
    results = {}
    for label, nd in [("single", None), ("sharded", n_devices)]:
        a = atoms.copy()
        a.calc = NequixCalculator(model_path=model_path, use_kernel=False, n_devices=nd)
        results[label] = (a.get_potential_energy(), a.get_forces(), a.get_stress())

    e1, f1, s1 = results["single"]
    en, fn, sn = results["sharded"]
    print(f"[check] n_atoms={len(atoms)}")
    print(f"[check] energy: single={e1:.6f} sharded={en:.6f} diff={abs(en - e1):.3e}")
    print(f"[check] max |force diff|:  {np.abs(fn - f1).max():.3e}")
    print(f"[check] max |stress diff|: {np.abs(sn - s1).max():.3e}")
    np.testing.assert_allclose(en, e1, rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(fn, f1, rtol=1e-3, atol=2e-4)
    np.testing.assert_allclose(sn, s1, rtol=1e-3, atol=1e-5)
    print("[check] OK")


def benchmark(model_path: str, n_devices, size: int, steps: int, skin: float):
    atoms = make_atoms(size)
    nd = None if n_devices in (None, 1) else n_devices
    atoms.calc = NequixCalculator(
        model_path=model_path, use_kernel=False, n_devices=nd, skin=skin
    )

    MaxwellBoltzmannDistribution(atoms, temperature_K=600)
    dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
    # warmup: includes jit compilation and the initial edge-capacity setup
    dyn.run(3)

    start = time.perf_counter()
    dyn.run(steps)
    elapsed = time.perf_counter() - start

    ms_per_step = 1000 * elapsed / steps
    print(
        f"[bench] n_atoms={len(atoms):>8d} n_devices={n_devices or 1} skin={skin} "
        f"steps={steps} nl_builds={atoms.calc._nl_builds} ms/step={ms_per_step:9.2f}"
    )
    return ms_per_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/nequix-mp-1.nqx")
    parser.add_argument("--size", type=int, default=8, help="supercell repeats (4*size^3 atoms)")
    parser.add_argument("--n-devices", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--skin", type=float, default=0.3, help="neighbor list skin (A)")
    parser.add_argument("--check", action="store_true", help="validate against single device")
    args = parser.parse_args()

    import jax

    print(f"[info] jax devices: {jax.devices()}")

    if args.check and args.n_devices > 1:
        check_equivalence(args.model_path, args.n_devices)

    benchmark(args.model_path, args.n_devices, args.size, args.steps, args.skin)
