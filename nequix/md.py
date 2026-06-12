"""Chunked on-device MD driver for NequixCalculator.

Runs K MD steps per jitted call (lax.scan) using jax-md's integrators over the
calculator's padded COO graph, eliminating the per-step Python/ASE/dispatch
overhead of the ASE loop. The neighbor list machinery is the calculator's own:
the graph is built on the host (vesin) at cutoff + skin and stays exact while
no atom has moved more than skin/2 since the build.

Design notes:
  - positions are CARTESIAN and never wrapped; periodicity lives entirely in
    the fixed integer shifts of the graph (exact while displacement < skin/2,
    the same invariant as the ASE path). This avoids the fractional-coordinate
    stale-shift hazard of neighbor-list-managed jax-md setups.
  - before each chunk, a conservative bound (max speed * chunk steps) decides
    whether the list could expire mid-chunk; if so it is rebuilt first. After
    the chunk the actual displacement is verified; if the bound was ever
    violated the chunk is REDONE from the saved state with a fresh list, so
    results are exact regardless (the integrator PRNG lives in the state, so
    the redo is deterministic).
  - frozen atoms (ASE FixAtoms) are handled by substituting their reference
    positions inside the energy (exact zero forces on frozen, exact mobile
    forces) plus a post-step freeze. The padded tail is pinned the same way so
    NaN forces on padded nodes (r = 0 edges) cannot drift padded positions.
  - the jitted chunk functions are cached on the calculator keyed by the
    integrator parameters; repeated run_md calls (e.g. once per cycle in a
    deposition workflow) re-trace only when array shapes change.

ASE units throughout (eV, A, amu, ASE time; pass timestep as 1.0 * units.fs).
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from ase import units
from ase.constraints import FixAtoms

try:
    from jax_md import dataclasses, simulate
except ImportError as e:
    raise ImportError(
        "jax_md is required for nequix.md: pip install git+https://github.com/jax-md/jax-md.git"
    ) from e


def _frozen_indices(atoms):
    idx = []
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            idx.extend(np.asarray(c.index).tolist())
    return np.unique(idx).astype(int)


def _get_md_fns(calc, timestep, temperature_K, friction, chunk_size, log_energy):
    """Build (or fetch cached) jitted init/chunk/step functions for this
    calculator and integrator parameters. All arrays are arguments, so the jit
    cache survives across run_md calls and re-traces only on shape changes."""
    key = (float(timestep), temperature_K, float(friction), int(chunk_size), bool(log_energy))
    cache = getattr(calc, "_md_fns", None)
    if cache is None:
        cache = calc._md_fns = {}
    if key in cache:
        return cache[key]

    model = calc.model

    def energy_fn(positions, graph, R_frozen, frozen_col, node_mask, **kwargs):
        # frozen atoms (and the padded tail) enter at their reference
        # positions no matter what the integrator did to them; padded edges
        # connect padded nodes only, so their r = 0 NaNs never reach real
        # forces (masked exactly like model.__call__)
        positions = jnp.where(frozen_col, R_frozen, positions)
        cell = graph.globals["cell"][0]
        offsets = graph.edges["shifts"] @ cell
        r = positions[graph.senders] - positions[graph.receivers] + offsets
        node_energies = model.node_energies(
            r, graph.nodes["species"], graph.senders, graph.receivers
        ).reshape(-1)  # (n, 1) -> (n,); a (n,) mask against (n, 1) would broadcast
        return jnp.sum(jnp.where(node_mask[:, 0], node_energies, 0.0))

    def shift_fn(R, dR, **kwargs):
        return R + dR  # cartesian, non-wrapping

    if friction > 0.0:
        kT = units.kB * temperature_K
        init_fn, apply_fn = simulate.nvt_langevin(
            energy_fn, shift_fn, timestep, kT, gamma=friction
        )
    else:
        init_fn, apply_fn = simulate.nve(energy_fn, shift_fn, timestep)

    def freeze(state, R_frozen, frozen_col):
        return dataclasses.replace(
            state,
            position=jnp.where(frozen_col, R_frozen, state.position),
            momentum=jnp.where(frozen_col, 0.0, state.momentum),
        )

    @jax.jit
    def chunk_fn(state, graph, R_build, R_frozen, frozen_col, node_mask, mobile_col,
                 masses, n_mobile):
        kwargs = dict(graph=graph, R_frozen=R_frozen, frozen_col=frozen_col,
                      node_mask=node_mask)

        def body(state, _):
            state = freeze(apply_fn(state, **kwargs), R_frozen, frozen_col)
            return state, None

        state, _ = jax.lax.scan(body, state, None, length=chunk_size)
        ekin = 0.5 * jnp.sum(jnp.where(mobile_col, state.momentum**2 / masses, 0.0))
        temp = 2.0 * ekin / (3.0 * n_mobile * units.kB)
        disp_sq = jnp.max(
            jnp.where(mobile_col[:, 0], jnp.sum((state.position - R_build) ** 2, axis=1), 0.0)
        )
        v_max = jnp.max(jnp.where(mobile_col, jnp.abs(state.momentum) / masses, 0.0))
        etot = (energy_fn(state.position, graph, R_frozen, frozen_col, node_mask) + ekin
                if log_energy else 0.0)
        return state, (temp, disp_sq, v_max, etot)

    @jax.jit
    def step_fn(state, graph, R_frozen, frozen_col, node_mask):
        kwargs = dict(graph=graph, R_frozen=R_frozen, frozen_col=frozen_col,
                      node_mask=node_mask)
        return freeze(apply_fn(state, **kwargs), R_frozen, frozen_col)

    cache[key] = (init_fn, chunk_fn, step_fn)
    return cache[key]


def run_md(
    atoms,
    calc,
    n_steps,
    *,
    timestep,
    temperature_K=None,
    friction=0.0,  # 1/ASE-time; 0 -> NVE
    seed=0,
    chunk_size=8,
    safety=1.5,  # margin factor on the max-displacement-per-chunk bound
    log_energy=False,  # record total energy once per chunk (one extra forward)
    verbose=False,
):
    """Run n_steps of Langevin NVT (or NVE if friction == 0) on device,
    updating atoms in place. Returns (steps_per_second, mean_mobile_T_K, info)
    where info has rebuilds/redos/temps (+ energies if log_energy)."""
    if calc.backend != "jax" or calc.mesh is not None:
        raise ValueError("run_md requires the single-device jax backend")
    if calc.skin <= 0.0:
        raise ValueError("run_md requires a positive neighbor-list skin")
    if friction > 0.0 and temperature_K is None:
        raise ValueError("temperature_K is required for Langevin (friction > 0)")

    n_real = len(atoms)
    half_skin = calc.skin / 2.0
    frozen = _frozen_indices(atoms)
    n_mobile = float(n_real - len(frozen))

    init_fn, chunk_fn, step_fn = _get_md_fns(
        calc, timestep, temperature_K, friction, chunk_size, log_energy
    )

    def fresh_graph():
        # synchronous build via the calculator's own machinery (vesin +
        # padding + device_put); adopt so the calculator cache stays consistent
        graph = calc._build_padded_graph_jax(atoms, False)
        calc._adopt_nl(graph, atoms.positions)
        return graph

    calc._nl_future = None  # a pending async build would race the writes below
    graph = fresh_graph()
    n_pad = graph.nodes["positions"].shape[0]

    node_mask = np.zeros((n_pad, 1), dtype=bool)
    node_mask[:n_real] = True
    frozen_col = np.zeros((n_pad, 1), dtype=bool)
    frozen_col[frozen] = True
    frozen_col[n_real:] = True  # pin the padded tail (see module docstring)
    mobile_col = jnp.asarray(node_mask & ~frozen_col)
    frozen_col = jnp.asarray(frozen_col)
    node_mask = jnp.asarray(node_mask)

    masses = np.ones((n_pad, 1), dtype=np.float32)  # padded mass 1 avoids div0
    masses[:n_real, 0] = atoms.get_masses()
    masses = jnp.asarray(masses)

    momenta = np.zeros((n_pad, 3), dtype=np.float32)
    momenta[:n_real] = atoms.get_momenta()
    momenta[frozen] = 0.0
    momenta = jnp.asarray(momenta)

    R0 = np.zeros((n_pad, 3), dtype=np.float32)
    R0[:n_real] = atoms.positions
    R0 = jnp.asarray(R0)
    R_frozen = R0  # frozen reference positions (constant for the whole run)
    R_build = R0  # positions at the current neighbor-list build

    key = jax.random.key(seed)
    init_kwargs = dict(graph=graph, R_frozen=R_frozen, frozen_col=frozen_col,
                       node_mask=node_mask)
    if friction > 0.0:
        state = init_fn(key, R0, mass=masses, **init_kwargs)
    else:
        state = init_fn(key, R0, kT=0.0, mass=masses, **init_kwargs)
    state = dataclasses.replace(state, momentum=momenta)

    v_max = float(jnp.max(jnp.where(mobile_col, jnp.abs(state.momentum) / masses, 0.0)))
    disp_sq = 0.0
    n_mobile_arr = jnp.asarray(n_mobile, dtype=jnp.float32)

    temps, energies = [], []
    done = rebuilds = redos = 0

    def single_steps(state, graph, R_build, disp_sq, v_max, k):
        # k steps with a per-step rebuild check; used for chunk redos and the
        # tail of n_steps that does not fill a chunk
        nonlocal rebuilds
        for _ in range(k):
            if disp_sq**0.5 + safety * v_max * timestep > half_skin:
                atoms.set_positions(np.asarray(state.position[:n_real], dtype=np.float64))
                graph = fresh_graph()
                R_build = state.position
                disp_sq = 0.0
                rebuilds += 1
            state = step_fn(state, graph, R_frozen, frozen_col, node_mask)
            disp_sq = float(jnp.max(jnp.where(
                mobile_col[:, 0], jnp.sum((state.position - R_build) ** 2, axis=1), 0.0
            )))
            v_max = float(jnp.max(jnp.where(mobile_col, jnp.abs(state.momentum) / masses, 0.0)))
        return state, graph, R_build, disp_sq, v_max

    # adaptive per-chunk drift estimate: the conservative max-speed bound
    # overestimates the actual net drift over a chunk by 3-5x (velocities
    # decorrelate), so the rebuild decision learns from observed chunk drifts
    # instead; the redo path below keeps every trajectory exact regardless
    drift_hist = []
    drift_est = None
    fallbacks = 0

    # background rebuilds: the next list is built in the calculator's worker
    # thread (vesin releases the GIL) while the main thread waits on the
    # chunk's scalar outputs, mirroring the calculator's async_nl machinery
    pending = None  # (future, R at snapshot [device array]) or None

    def request_build(positions_device):
        nonlocal pending
        if calc._nl_executor is None:
            from concurrent.futures import ThreadPoolExecutor

            calc._nl_executor = ThreadPoolExecutor(max_workers=1)
        snapshot = atoms.copy()
        snapshot.set_positions(np.asarray(positions_device[:n_real], dtype=np.float64))
        future = calc._nl_executor.submit(calc._build_padded_graph_jax, snapshot, False)
        pending = (future, positions_device)

    def disp_from(R_ref):
        return float(jnp.max(jnp.where(
            mobile_col[:, 0], jnp.sum((state.position - R_ref) ** 2, axis=1), 0.0
        )))

    t0 = time.perf_counter()
    while done + chunk_size <= n_steps:
        if drift_est is not None and drift_est > half_skin:
            # the dynamics outrun the skin within a single chunk: chunking
            # cannot work at these parameters, single-step instead (and decay
            # the estimate so chunking is retried once the system calms down)
            pending = None
            state, graph, R_build, disp_sq, v_max = single_steps(
                state, graph, R_build, disp_sq, v_max, chunk_size
            )
            drift_est *= 0.9
            fallbacks += 1
            done += chunk_size
            continue

        # adopt a finished background build
        if pending is not None and pending[0].done():
            future, R_new = pending
            pending = None
            graph, R_build = future.result(), R_new
            disp_sq = disp_from(R_build)
            rebuilds += 1

        est = drift_est if drift_est is not None else half_skin
        if disp_sq**0.5 + est > half_skin:
            # the list could expire inside this chunk: use the pending build
            # if there is one (blocking), else rebuild synchronously
            if pending is not None:
                future, R_new = pending
                pending = None
                graph, R_build = future.result(), R_new
                disp_sq = disp_from(R_build)
                rebuilds += 1
            if disp_sq**0.5 + est > half_skin:
                atoms.set_positions(np.asarray(state.position[:n_real], dtype=np.float64))
                graph = fresh_graph()
                R_build = state.position
                disp_sq = 0.0
                rebuilds += 1
        elif pending is None and disp_sq**0.5 + 2.0 * est > half_skin:
            # expiry predicted within ~2 chunks: start the next build now so
            # it overlaps with this chunk's GPU execution
            request_build(state.position)

        prev_state = state
        state, (temp, new_disp_sq, new_v_max, etot) = chunk_fn(
            state, graph, R_build, R_frozen, frozen_col, node_mask, mobile_col,
            masses, n_mobile_arr,
        )
        drift_hist.append(max(float(new_disp_sq) ** 0.5 - disp_sq**0.5, 0.0))
        drift_est = 1.3 * max(drift_hist[-10:])
        if float(new_disp_sq) > half_skin**2:
            # the chunk overshot skin/2: redo it from the saved state ONE STEP
            # AT A TIME with per-step rebuild checks (retrying the whole chunk
            # would overshoot again deterministically and livelock). Exact for
            # the same reason the ASE path is: every force evaluation sees a
            # list that is valid at the evaluated positions.
            state = prev_state
            pending = None  # single_steps rebuilds fresh; drop the older build
            redos += 1
            state, graph, R_build, disp_sq, v_max = single_steps(
                state, graph, R_build, disp_sq, v_max, chunk_size
            )
            done += chunk_size
            continue
        disp_sq = float(new_disp_sq)
        v_max = float(new_v_max)
        temps.append(float(temp))
        if log_energy:
            energies.append(float(etot))
        done += chunk_size

    # finish remainder steps with the jitted single-step function
    if done < n_steps:
        state, graph, R_build, disp_sq, v_max = single_steps(
            state, graph, R_build, disp_sq, v_max, n_steps - done
        )
    jax.block_until_ready(state.position)
    elapsed = time.perf_counter() - t0

    atoms.set_positions(np.asarray(state.position[:n_real], dtype=np.float64))
    atoms.set_momenta(np.asarray(state.momentum[:n_real], dtype=np.float64))
    # the calculator's cached list refers to mid-run positions; invalidate so
    # the next ASE call rebuilds from the final state
    calc._nl_graph = None
    calc._nl_positions = None
    calc.results = {}

    sps = n_steps / elapsed
    if temps:
        mean_temp = float(np.mean(temps))
    else:
        # no successful chunk recorded a temperature; compute from final state
        ekin = 0.5 * float(jnp.sum(jnp.where(mobile_col, state.momentum**2 / masses, 0.0)))
        mean_temp = 2.0 * ekin / (3.0 * n_mobile * units.kB)
    if verbose:
        print(
            f"[nequix.md] {n_steps} steps in {elapsed:.2f} s ({sps:.1f} steps/s), "
            f"rebuilds={rebuilds} redos={redos} fallbacks={fallbacks}, "
            f"T(mobile) = {mean_temp:.0f} K",
            flush=True,
        )
    info = {
        "rebuilds": rebuilds,
        "redos": redos,
        "fallbacks": fallbacks,
        "temps": temps,
        "energies": energies,
    }
    return sps, mean_temp, info
