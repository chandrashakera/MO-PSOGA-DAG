"""
psoga_r.py
PSOGA_R baseline — Wang et al. [2024] doi:10.1007/s10586-024-04851-3

Implements the primary baseline algorithm exactly as described in
Wang et al. [2024] Section 4, extended with DAG_Repair to handle
precedence constraints (all baselines receive this extension).

Key design decisions (from paper):
  - Per-particle per-iteration: 3 GA crossovers + 1 mutation
  - PSO iterative framework (velocity + position update)
  - Cloud-biased initialisation (p_cloud=0.5)
  - Single-objective: maximise N + U (acceptance + utilisation)
  - Rescheduling operator post-decoding
  - Budget: N=100, I=100 (same as MO-PSOGA-DAG)

For Pareto metric comparison: the best-compromise solution is
extracted per run using minimum normalised distance to ideal point
(Section 5.5.2 of paper).
"""

import numpy as np
from environment.de3c_system import DE3CSystem
from environment.dag_generator import DAGInstance
from environment.task_scheduler import compute_finish_times, objectives_from_result
from algorithms.dag_repair import repair_and_reschedule


def _fitness(disc: np.ndarray, dag: DAGInstance,
             system: DE3CSystem) -> float:
    """
    Compute Wang et al. [2024] scalar fitness: N + U.

    N = number of accepted tasks (acceptance count)
    U = sum of utilisation across all nodes
       = sum_{i} (busy_time_i / total_time)

    Note: Wang et al. use N + U as the objective, not just N.
    We maximise this (equivalently, minimise -(N+U)).
    """
    result = compute_finish_times(disc, dag, system)
    N = result.N
    T = dag.T

    # Utilisation: fraction of each node's time that is busy
    # Approximated as: sum of processing times / finish time span
    total_util = 0.0
    for node_idx in range(system.total_nodes):
        tasks_on_node = [k for k in range(T) if disc[k] == node_idx]
        if not tasks_on_node:
            continue
        node = system.get_node(node_idx)
        busy = sum(dag.tasks[k].compute / node.capacity_ghz
                   for k in tasks_on_node)
        span = max(result.finish_times[k] for k in tasks_on_node)
        total_util += busy / max(span, 1e-9)

    return float(N + total_util)


def _init_population(N: int, T: int,
                     dag: DAGInstance,
                     system: DE3CSystem,
                     rng: np.random.Generator) -> np.ndarray:
    """Cloud-biased initialisation, same as MO-PSOGA-DAG (Wang 2024 §4.1)."""
    D, E, V = system.D, system.E, system.V
    pop = np.zeros((N, T), dtype=float)
    first_cloud = D + E

    for k in range(T):
        task = dag.tasks[k]
        feasible = system.feasible_nodes_for_task(task.device)
        cloud_nodes = [n for n in feasible if system.is_cloud(n)]
        es_nodes    = [n for n in feasible if system.is_edge(n)]
        device_node = task.device

        nodes = [device_node] + es_nodes + cloud_nodes
        p_dev = 0.20
        p_es  = 0.30 / max(len(es_nodes), 1)
        p_cl  = 0.50 / max(len(cloud_nodes), 1)
        probs = ([p_dev] + [p_es]*len(es_nodes) + [p_cl]*len(cloud_nodes))
        probs = np.array(probs); probs /= probs.sum()
        pop[:, k] = rng.choice(nodes, size=N, p=probs)

    return pop


def _discretize(pos: np.ndarray, dag: DAGInstance,
                system: DE3CSystem) -> np.ndarray:
    disc = np.clip(np.round(pos), 0, system.total_nodes - 1).astype(int)
    first_cloud = system.D + system.E
    for k in range(dag.T):
        feasible = system.feasible_nodes_for_task(dag.tasks[k].device)
        if disc[k] not in feasible:
            disc[k] = first_cloud
    return disc


def run(dag: DAGInstance,
        system: DE3CSystem,
        rng: np.random.Generator,
        N: int = 100,
        I: int = 100,
        omega_max: float = 1.2,
        omega_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        p_c: float = 0.8,
        p_m: float = 0.1) -> dict:
    """
    Run PSOGA_R (Wang et al. 2024) with DAG extension.

    Returns best solution found (max N+U) as both scalar and
    multi-objective form [F1, F2, -F3] for cross-algorithm comparison.
    """
    T = dag.T
    n_evals = 0

    # Initialise population
    pop_pos = _init_population(N, T, dag, system, rng)
    pop_vel = np.zeros((N, T))
    pop_fit = np.full(N, -np.inf)
    pop_disc = np.zeros((N, T), dtype=int)

    for i in range(N):
        disc = _discretize(pop_pos[i], dag, system)
        disc, result = repair_and_reschedule(disc, dag, system)
        pop_disc[i] = disc
        pop_pos[i]  = disc.astype(float)
        pop_fit[i]  = _fitness(disc, dag, system)
        n_evals += 1

    # Personal bests
    pb_pos  = pop_pos.copy()
    pb_disc = pop_disc.copy()
    pb_fit  = pop_fit.copy()

    # Global best
    gb_idx  = int(np.argmax(pb_fit))
    gb_pos  = pb_pos[gb_idx].copy()
    gb_disc = pb_disc[gb_idx].copy()
    gb_fit  = pb_fit[gb_idx]

    best_history = []

    for t in range(I):
        omega = omega_max - (omega_max - omega_min) / I * t

        for i in range(N):
            # PSO velocity update
            phi1 = rng.random(T)
            phi2 = rng.random(T)
            pop_vel[i] = (omega * pop_vel[i]
                          + c1 * phi1 * (pb_pos[i] - pop_pos[i])
                          + c2 * phi2 * (gb_pos - pop_pos[i]))
            pop_pos[i] += pop_vel[i]

            # GA: 3 crossovers per particle (Wang et al. 2024 §4.2)
            for _ in range(3):
                r = rng.integers(0, N)
                while r == i:
                    r = rng.integers(0, N)
                mask = rng.random(T) < p_c
                pop_pos[i] = np.where(mask, pop_pos[i], pop_pos[r])

            # GA: 1 mutation — random reset of selected dimensions
            mut_mask = rng.random(T) < p_m
            for k in np.where(mut_mask)[0]:
                feasible = system.feasible_nodes_for_task(dag.tasks[k].device)
                pop_pos[i, k] = float(rng.choice(feasible))

            # Decode + repair + reschedule
            disc = _discretize(pop_pos[i], dag, system)
            disc, result = repair_and_reschedule(disc, dag, system)
            pop_disc[i] = disc
            pop_pos[i]  = disc.astype(float)
            fit = _fitness(disc, dag, system)
            n_evals += 1

            # Update personal best
            if fit > pb_fit[i]:
                pb_pos[i]  = pop_pos[i].copy()
                pb_disc[i] = disc.copy()
                pb_fit[i]  = fit

        # Update global best
        best_i = int(np.argmax(pb_fit))
        if pb_fit[best_i] > gb_fit:
            gb_pos  = pb_pos[best_i].copy()
            gb_disc = pb_disc[best_i].copy()
            gb_fit  = pb_fit[best_i]

        best_history.append(gb_fit)

    # Compute multi-objective values of best solution for cross-algorithm comparison
    final_result = compute_finish_times(gb_disc, dag, system)
    best_obj = objectives_from_result(final_result)

    return {
        'best_disc':    gb_disc,
        'best_obj':     best_obj,
        'best_f4':      final_result.cloud_cost,
        'best_fitness': gb_fit,
        'history':      best_history,
        'n_evals':      n_evals,
        # For Pareto metric consistency: wrap in (1, 3) archive format
        'archive_pos':  gb_disc.reshape(1, -1),
        'archive_obj':  best_obj.reshape(1, -1),
        'archive_f4':   np.array([final_result.cloud_cost]),
    }
