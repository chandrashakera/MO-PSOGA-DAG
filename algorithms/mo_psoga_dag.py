"""
mo_psoga_dag.py
MO-PSOGA-DAG: Multi-Objective Hybrid PSO-GA with DAG-Aware Rescheduling.

Implements Algorithm 3 (paper Section 4.8) with:
  - Cloud-biased initialisation (Eq. 15, p_cloud=0.5)
  - PSO velocity update (Eq. 16)
  - GA uniform crossover (Eq. 18)
  - NSGA-II non-dominated sorting + crowding distance (Eqs. 19-20)
  - gbest from Pareto archive (Eq. 21)
  - pbest update by Pareto dominance (Eq. 22)
  - DAG_Repair + DAG_Reschedule applied after every position update

Parameters: Table 2 of paper (N=100, I=100, ω_max=1.2, ω_min=0.4,
             c1=c2=2.0, p_c=0.8, archive_size=100)

Reference: Paper Section 4, Wang et al. [2024] doi:10.1007/s10586-024-04851-3,
           Deb et al. [2002] doi:10.1109/4235.996017,
           Kennedy & Eberhart [1995] doi:10.1109/ICNN.1995.488968
"""

import numpy as np
from environment.de3c_system import DE3CSystem
from environment.dag_generator import DAGInstance
from environment.task_scheduler import (compute_finish_times,
                                        objectives_from_result)
from algorithms.dag_repair import repair_and_reschedule
from algorithms.pareto_utils import (fast_non_dominated_sort,
                                     compute_all_crowding_distances,
                                     nsga2_select,
                                     select_gbest,
                                     update_archive,
                                     dominates,
                                     crowded_better)


def _init_population(N: int, T: int,
                     dag: DAGInstance,
                     system: DE3CSystem,
                     rng: np.random.Generator,
                     p_cloud: float = 0.50,
                     p_edge: float = 0.30,
                     p_device: float = 0.20) -> np.ndarray:
    """
    Cloud-biased initialisation — paper Eq. (15).

    For each task k and each particle i:
      P(cloud)   = 0.50  → uniform over V cloud servers
      P(edge_j)  = 0.30 / E (feasible ES only)
      P(device)  = 0.20  → launching device

    Returns pop : shape (N, T) integer array of node assignments.
    """
    D, E, V = system.D, system.E, system.V
    pop = np.zeros((N, T), dtype=int)

    for k in range(T):
        task = dag.tasks[k]
        feasible = system.feasible_nodes_for_task(task.device)
        es_nodes = [n for n in feasible if system.is_edge(n)]
        cloud_nodes = [n for n in feasible if system.is_cloud(n)]

        # Probabilities per feasible node (Eq. 15)
        probs = []
        nodes = []
        # device
        nodes.append(task.device)
        probs.append(p_device)
        # edge servers
        for n in es_nodes:
            nodes.append(n)
            probs.append(p_edge / len(es_nodes) if es_nodes else 0.0)
        # cloud servers
        for n in cloud_nodes:
            nodes.append(n)
            probs.append(p_cloud / len(cloud_nodes) if cloud_nodes else 0.0)

        probs = np.array(probs)
        probs /= probs.sum()   # normalise (handles edge case)

        pop[:, k] = rng.choice(nodes, size=N, p=probs)

    return pop.astype(float)   # real-valued for velocity update


def _discretize(positions: np.ndarray, system: DE3CSystem,
                dag: DAGInstance) -> np.ndarray:
    """
    Map real-valued position to integer node index (Eq. 14).
    clip(round(x), 1, D+E+V) — then re-map to nearest feasible node.
    """
    T = dag.T
    total = system.total_nodes
    disc = np.clip(np.round(positions), 0, total - 1).astype(int)
    # Feasibility correction: if assigned node not feasible, use cloud[0]
    first_cloud = system.D + system.E
    for k in range(T):
        task = dag.tasks[k]
        feasible = system.feasible_nodes_for_task(task.device)
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
        archive_size: int = 100) -> dict:
    """
    Run MO-PSOGA-DAG and return Pareto archive + objective history.

    Returns
    -------
    dict with keys:
      'archive_pos'  : (M, T) best Pareto solutions (node assignments)
      'archive_obj'  : (M, 3) objective values [F1, F2, -F3]
      'archive_f4'   : (M,)   cloud cost (secondary metric)
      'hv_history'   : list of HV at each iteration (for convergence plot)
      'n_evals'      : total function evaluations (should equal N*I)
    """
    T = dag.T
    n_evals = 0

    # ── Initialisation ────────────────────────────────────────────────────────
    # Real-valued population (Eq. 13)
    pop_pos = _init_population(N, T, dag, system, rng)  # (N, T) float
    pop_vel = np.zeros((N, T))                           # velocities

    # Apply repair + reschedule to all particles
    pop_disc = np.zeros((N, T), dtype=int)
    pop_obj  = np.zeros((N, 3))
    pop_f4   = np.zeros(N)

    for i in range(N):
        disc = _discretize(pop_pos[i], system, dag)
        disc, result = repair_and_reschedule(disc, dag, system)
        pop_disc[i] = disc
        pop_pos[i] = disc.astype(float)   # sync real position to repaired
        pop_obj[i] = objectives_from_result(result)
        pop_f4[i] = result.cloud_cost
        n_evals += 1

    # Personal bests (Eq. 22 initialisation)
    pb_pos = pop_pos.copy()
    pb_obj = pop_obj.copy()

    # Pareto archive (non-dominated solutions)
    archive_pos = np.zeros((0, T), dtype=float)
    archive_obj = np.zeros((0, 3))
    archive_f4  = np.zeros(0)
    archive_pos, archive_obj = update_archive(
        archive_pos, archive_obj, pop_pos, pop_obj, archive_size)
    archive_f4 = np.zeros(len(archive_obj))   # placeholder

    # Global best (Eq. 21)
    gb_idx = select_gbest(archive_obj)
    gb_pos = archive_pos[gb_idx].copy()

    hv_history = []

    # ── Main loop ─────────────────────────────────────────────────────────────
    for t in range(I):
        # Inertia weight decay (Eq. 17)
        omega = omega_max - (omega_max - omega_min) / I * t

        new_pop_pos = np.zeros_like(pop_pos)
        new_pop_obj = np.zeros_like(pop_obj)
        new_pop_f4  = np.zeros(N)

        for i in range(N):
            # ── PSO velocity update (Eq. 16) ──────────────────────────────
            phi1 = rng.random(T)    # φ₁ ~ U(0,1) per dimension
            phi2 = rng.random(T)    # φ₂ ~ U(0,1) per dimension
            pop_vel[i] = (omega * pop_vel[i]
                          + c1 * phi1 * (pb_pos[i] - pop_pos[i])
                          + c2 * phi2 * (gb_pos - pop_pos[i]))

            x_pso = pop_pos[i] + pop_vel[i]

            # ── GA uniform crossover (Eq. 18) ─────────────────────────────
            r_idx = rng.integers(0, N)
            while r_idx == i:
                r_idx = rng.integers(0, N)
            u = rng.random(T)
            x_cross = np.where(u < p_c, x_pso, pop_pos[r_idx])

            # ── DAG repair + reschedule; 1 eval per particle ─────────────
            # Budget fairness: 1 evaluation per particle per iteration,
            # matching Wang et al. [2024] N*I total budget for all algorithms.
            # Crossover is the final transformation; only the crossover
            # offspring is evaluated (Eq. 19).
            disc_cross = _discretize(x_cross, system, dag)
            disc_cross, result_cross = repair_and_reschedule(
                disc_cross, dag, system)
            obj_cross = objectives_from_result(result_cross)
            n_evals += 1   # 1 eval per particle per iteration

            new_pop_pos[i] = disc_cross.astype(float)
            new_pop_obj[i] = obj_cross
            new_pop_f4[i]  = result_cross.cloud_cost

            # ── Update personal best (Eq. 22) ─────────────────────────────
            if dominates(new_pop_obj[i], pb_obj[i]):
                pb_pos[i] = new_pop_pos[i].copy()
                pb_obj[i] = new_pop_obj[i].copy()
            elif dominates(pb_obj[i], new_pop_obj[i]):
                pass  # keep pb
            else:
                # Non-dominating: random choice
                if rng.random() < 0.5:
                    pb_pos[i] = new_pop_pos[i].copy()
                    pb_obj[i] = new_pop_obj[i].copy()

        pop_pos = new_pop_pos
        pop_obj = new_pop_obj
        pop_f4  = new_pop_f4

        # ── NSGA-II selection on combined pool (pop ∪ pbests) ─────────────
        combined_pos = np.vstack([pop_pos, pb_pos])
        combined_obj = np.vstack([pop_obj, pb_obj])
        sel_idx = nsga2_select(combined_obj, N)
        pop_pos = combined_pos[sel_idx]
        pop_obj = combined_obj[sel_idx]

        # ── Update Pareto archive ─────────────────────────────────────────
        archive_pos, archive_obj = update_archive(
            archive_pos, archive_obj, pop_pos, pop_obj, archive_size)

        # ── Update global best ────────────────────────────────────────────
        gb_idx = select_gbest(archive_obj)
        gb_pos = archive_pos[gb_idx].copy()

        hv_history.append(len(archive_obj))   # placeholder; HV computed in metrics

    # Compute F4 for archive solutions
    archive_f4 = np.zeros(len(archive_obj))
    for idx in range(len(archive_pos)):
        disc = archive_pos[idx].astype(int)
        result = compute_finish_times(disc, dag, system)
        archive_f4[idx] = result.cloud_cost

    return {
        'archive_pos': archive_pos.astype(int),
        'archive_obj': archive_obj,
        'archive_f4':  archive_f4,
        'hv_history':  hv_history,
        'n_evals':     n_evals,
    }
