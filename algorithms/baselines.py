"""
baselines.py
All four remaining baseline algorithms for MO-PSOGA-DAG paper.

  1. NSGA-II   — Deb et al. [2002] doi:10.1109/4235.996017
  2. MOPSO     — Coello et al. [2004] doi:10.1007/978-3-540-24694-7_71
  3. MOEA/D    — Zhang & Li [2007] doi:10.1109/TEVC.2007.892759
  4. FF        — First Fit heuristic (Wang et al. 2024)
  5. EDF       — Earliest Deadline First (Baker 2005)

All multi-objective algorithms:
  - Same budget N=100 × I=100 = 10,000 evaluations
  - Same cloud-biased initialisation
  - DAG_Repair + DAG_Reschedule applied after every update

Heuristics:
  - Deterministic; no budget constraint (run once)
  - DAG_Repair applied for feasibility
"""

import numpy as np
from environment.de3c_system import DE3CSystem
from environment.dag_generator import DAGInstance
from environment.task_scheduler import compute_finish_times, objectives_from_result
from algorithms.dag_repair import repair_and_reschedule
from algorithms.pareto_utils import (fast_non_dominated_sort,
                                     compute_all_crowding_distances,
                                     nsga2_select,
                                     select_gbest,
                                     update_archive,
                                     dominates,
                                     crowding_distance)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _cloud_biased_init(N: int, T: int, dag: DAGInstance,
                       system: DE3CSystem,
                       rng: np.random.Generator) -> np.ndarray:
    """Cloud-biased initialisation (Wang et al. 2024; p_cloud=0.5)."""
    pop = np.zeros((N, T), dtype=int)
    for k in range(T):
        task = dag.tasks[k]
        feasible = system.feasible_nodes_for_task(task.device)
        cloud_n = [n for n in feasible if system.is_cloud(n)]
        edge_n  = [n for n in feasible if system.is_edge(n)]
        dev_n   = [task.device]
        nodes   = dev_n + edge_n + cloud_n
        p_d = 0.20; p_e = 0.30 / max(len(edge_n), 1)
        p_c = 0.50 / max(len(cloud_n), 1)
        probs = np.array([p_d]*1 + [p_e]*len(edge_n) + [p_c]*len(cloud_n))
        probs /= probs.sum()
        pop[:, k] = rng.choice(nodes, size=N, p=probs)
    return pop


def _eval_population(pop: np.ndarray, dag: DAGInstance,
                     system: DE3CSystem) -> tuple:
    """Evaluate objectives for all particles; return (obj_array, f4_array)."""
    N = len(pop)
    obj = np.zeros((N, 3))
    f4  = np.zeros(N)
    for i in range(N):
        result = compute_finish_times(pop[i], dag, system)
        obj[i] = objectives_from_result(result)
        f4[i]  = result.cloud_cost
    return obj, f4


def _uniform_crossover(a: np.ndarray, b: np.ndarray,
                       p_c: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(len(a)) < p_c
    return np.where(mask, a, b).astype(int)


def _mutate(x: np.ndarray, dag: DAGInstance, system: DE3CSystem,
            p_m: float, rng: np.random.Generator) -> np.ndarray:
    x = x.copy()
    for k in range(len(x)):
        if rng.random() < p_m:
            feasible = system.feasible_nodes_for_task(dag.tasks[k].device)
            x[k] = int(rng.choice(feasible))
    return x


def _std_result(archive_pos, archive_obj, archive_f4, n_evals):
    return {
        'archive_pos': np.atleast_2d(archive_pos).astype(int),
        'archive_obj': np.atleast_2d(archive_obj),
        'archive_f4':  np.atleast_1d(archive_f4),
        'n_evals':     n_evals,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. NSGA-II (Deb et al. 2002)
# ═══════════════════════════════════════════════════════════════════════════════

def run_nsga2(dag: DAGInstance, system: DE3CSystem,
              rng: np.random.Generator,
              N: int = 100, I: int = 100,
              p_c: float = 0.8, p_m: float = 0.1) -> dict:
    """
    NSGA-II with uniform crossover and random mutation.
    No PSO component — isolates GA contribution vs MO-PSOGA-DAG.

    Tournament selection (binary) for parent choice.
    Budget: N * I evaluations.
    """
    T = dag.T
    n_evals = 0

    # Initialise
    pop = _cloud_biased_init(N, T, dag, system, rng)
    for i in range(N):
        pop[i], _ = repair_and_reschedule(pop[i], dag, system)
    obj, f4 = _eval_population(pop, dag, system)
    n_evals += N

    for t in range(I):
        # Offspring generation: N offspring
        offspring = np.zeros_like(pop)
        for i in range(N):
            # Binary tournament selection (2 parents)
            p1, p2 = rng.choice(N, size=2, replace=False)
            # Compare by rank then crowding
            fronts = fast_non_dominated_sort(obj)
            ranks = np.zeros(N, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = r
            cd_all = compute_all_crowding_distances(obj, fronts)
            parent = p1 if (ranks[p1] < ranks[p2] or
                            (ranks[p1] == ranks[p2] and cd_all[p1] >= cd_all[p2])) else p2
            # Second parent: random
            p2b = rng.integers(0, N)
            child = _uniform_crossover(pop[parent], pop[p2b], p_c, rng)
            child = _mutate(child, dag, system, p_m, rng)
            child, _ = repair_and_reschedule(child, dag, system)
            offspring[i] = child

        off_obj, off_f4 = _eval_population(offspring, dag, system)
        n_evals += N

        # Combined selection
        combined     = np.vstack([pop, offspring])
        combined_obj = np.vstack([obj, off_obj])
        sel = nsga2_select(combined_obj, N)
        pop = combined[sel]
        obj = combined_obj[sel]

    # Archive: non-dominated solutions from final population
    fronts = fast_non_dominated_sort(obj)
    nd_idx = fronts[0]
    archive_pos = pop[nd_idx]
    archive_obj = obj[nd_idx]
    archive_f4  = np.array([
        compute_finish_times(archive_pos[i], dag, system).cloud_cost
        for i in range(len(nd_idx))])

    return _std_result(archive_pos, archive_obj, archive_f4, n_evals)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MOPSO (Coello et al. 2004)
# ═══════════════════════════════════════════════════════════════════════════════

def run_mopso(dag: DAGInstance, system: DE3CSystem,
              rng: np.random.Generator,
              N: int = 100, I: int = 100,
              omega_max: float = 1.2, omega_min: float = 0.4,
              c1: float = 2.0, c2: float = 2.0,
              archive_size: int = 100) -> dict:
    """
    MOPSO with external archive (Coello et al. 2004).
    No GA crossover — isolates PSO contribution vs MO-PSOGA-DAG.

    Leader selected from archive by crowding distance.
    Budget: N * I evaluations.
    """
    T = dag.T
    n_evals = 0

    pop = _cloud_biased_init(N, T, dag, system, rng).astype(float)
    vel = np.zeros((N, T))
    for i in range(N):
        disc, _ = repair_and_reschedule(pop[i].astype(int), dag, system)
        pop[i] = disc.astype(float)
    obj, f4 = _eval_population(pop.astype(int), dag, system)
    n_evals += N

    pb_pos = pop.copy()
    pb_obj = obj.copy()

    archive_pos = np.zeros((0, T), dtype=float)
    archive_obj = np.zeros((0, 3))
    archive_pos, archive_obj = update_archive(
        archive_pos, archive_obj, pop, obj, archive_size)

    for t in range(I):
        omega = omega_max - (omega_max - omega_min) / I * t

        # Leader from archive
        if len(archive_obj) > 0:
            leader_idx = select_gbest(archive_obj)
            leader = archive_pos[leader_idx]
        else:
            leader = pop[0]

        for i in range(N):
            phi1 = rng.random(T)
            phi2 = rng.random(T)
            vel[i] = (omega * vel[i]
                      + c1 * phi1 * (pb_pos[i] - pop[i])
                      + c2 * phi2 * (leader - pop[i]))
            pop[i] = pop[i] + vel[i]

            disc, result = repair_and_reschedule(
                pop[i].astype(int), dag, system)
            pop[i] = disc.astype(float)
            obj_i = objectives_from_result(result)
            n_evals += 1

            if dominates(obj_i, pb_obj[i]):
                pb_pos[i] = pop[i].copy()
                pb_obj[i] = obj_i
            elif not dominates(pb_obj[i], obj_i) and rng.random() < 0.5:
                pb_pos[i] = pop[i].copy()
                pb_obj[i] = obj_i

        obj, _ = _eval_population(pop.astype(int), dag, system)
        archive_pos, archive_obj = update_archive(
            archive_pos, archive_obj, pop, obj, archive_size)

    archive_f4 = np.array([
        compute_finish_times(archive_pos[i].astype(int), dag, system).cloud_cost
        for i in range(len(archive_pos))])

    return _std_result(archive_pos, archive_obj, archive_f4, n_evals)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MOEA/D (Zhang & Li 2007)
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_weight_vectors(N: int, m: int = 3) -> np.ndarray:
    """
    Uniform weight vectors for MOEA/D (Zhang & Li 2007).
    For m=3 objectives, use Das & Dennis systematic approach.
    Returns (N, 3) weight matrix with rows summing to 1.
    """
    # Simple uniform grid for 3 objectives
    H = int(round((-3 + (9 + 8 * (N - 1)) ** 0.5) / 2))
    H = max(H, 1)
    weights = []
    for i in range(H + 1):
        for j in range(H - i + 1):
            k = H - i - j
            weights.append([i / H, j / H, k / H])
    weights = np.array(weights)
    # If not exactly N, sample N rows
    if len(weights) != N:
        idx = np.round(np.linspace(0, len(weights) - 1, N)).astype(int)
        weights = weights[idx]
    return weights


def run_moead(dag: DAGInstance, system: DE3CSystem,
              rng: np.random.Generator,
              N: int = 100, I: int = 100,
              T_nb: int = 20,
              p_c: float = 0.8, p_m: float = 0.1) -> dict:
    """
    MOEA/D with Tchebycheff decomposition (Zhang & Li 2007).

    Decomposes 3-objective problem into N scalar subproblems.
    Neighbourhood-based update (T_nb nearest weight vectors).
    Budget: N * I evaluations.
    """
    T = dag.T
    n_evals = 0
    m = 3   # objectives

    weights = _generate_weight_vectors(N, m)
    N_actual = len(weights)

    # Neighbourhood: T_nb nearest weight vectors per subproblem
    dist_matrix = np.linalg.norm(
        weights[:, None, :] - weights[None, :, :], axis=2)
    neighbourhoods = np.argsort(dist_matrix, axis=1)[:, :T_nb]

    # Initialise population
    pop = _cloud_biased_init(N_actual, T, dag, system, rng)
    for i in range(N_actual):
        pop[i], _ = repair_and_reschedule(pop[i], dag, system)
    obj, f4 = _eval_population(pop, dag, system)
    n_evals += N_actual

    # Reference point z* (ideal; minimise all) (Zhang & Li 2007)
    z_star = obj.min(axis=0)

    # Scalar fitness: Tchebycheff aggregation
    def tcheb(obj_i: np.ndarray, w: np.ndarray, z: np.ndarray) -> float:
        return float(np.max(w * np.abs(obj_i - z)))

    for t in range(I):
        for i in range(N_actual):
            nb = neighbourhoods[i]

            # Select two neighbours as parents
            p1, p2 = rng.choice(nb, size=2, replace=False)
            child = _uniform_crossover(pop[p1], pop[p2], p_c, rng)
            child = _mutate(child, dag, system, p_m, rng)
            child, result = repair_and_reschedule(child, dag, system)
            child_obj = objectives_from_result(result)
            n_evals += 1

            # Update reference point
            z_star = np.minimum(z_star, child_obj)

            # Update neighbours
            for j in nb:
                w_j = weights[j]
                if (tcheb(child_obj, w_j, z_star) <=
                        tcheb(obj[j], w_j, z_star)):
                    pop[j] = child.copy()
                    obj[j] = child_obj.copy()

    # Archive: non-dominated solutions from final population
    fronts = fast_non_dominated_sort(obj)
    nd_idx = fronts[0]
    archive_pos = pop[nd_idx]
    archive_obj = obj[nd_idx]
    archive_f4 = np.array([
        compute_finish_times(archive_pos[i], dag, system).cloud_cost
        for i in range(len(nd_idx))])

    return _std_result(archive_pos, archive_obj, archive_f4, n_evals)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Heuristics: FF and EDF (Wang et al. 2024)
# ═══════════════════════════════════════════════════════════════════════════════

def _heuristic_result(assignment: np.ndarray, dag: DAGInstance,
                      system: DE3CSystem) -> dict:
    """Compute objectives for a heuristic assignment and return standard dict."""
    assignment, result = repair_and_reschedule(assignment, dag, system)
    obj = objectives_from_result(result)
    return _std_result(
        assignment.reshape(1, -1), obj.reshape(1, -1),
        np.array([result.cloud_cost]), n_evals=1)


def run_ff(dag: DAGInstance, system: DE3CSystem,
           rng: np.random.Generator = None, **kwargs) -> dict:
    """
    First Fit (FF) heuristic — Wang et al. [2024].

    Assigns each task to the first feasible node (priority: device → ES → cloud)
    that can complete it within deadline.
    If no node meets deadline, assign to cloud (best-effort).
    """
    T = dag.T
    assignment = np.zeros(T, dtype=int)
    first_cloud = system.D + system.E

    # Process in topological order to respect DAG constraints
    # (ready times must be computed sequentially)
    node_free_time = np.zeros(system.total_nodes)  # when each node next free

    for k in dag.topo_order:
        task = dag.tasks[k]
        feasible = system.feasible_nodes_for_task(task.device)
        chosen = first_cloud  # default: cloud[0]

        for node_idx in feasible:  # device → ES → cloud order
            node = system.get_node(node_idx)
            # Estimate finish time on this node
            proc = task.compute / node.capacity_ghz
            if system.is_device(node_idx):
                est_finish = max(node_free_time[node_idx], 0) + proc
            else:
                transfer = task.data_mb / node.transfer_mbps
                est_finish = node_free_time[node_idx] + transfer + proc
            if est_finish <= task.deadline:
                chosen = node_idx
                break

        assignment[k] = chosen
        # Update node free time estimate
        node = system.get_node(chosen)
        proc = task.compute / node.capacity_ghz
        if system.is_device(chosen):
            node_free_time[chosen] += proc
        else:
            node_free_time[chosen] += task.data_mb / node.transfer_mbps + proc

    return _heuristic_result(assignment, dag, system)


def run_edf(dag: DAGInstance, system: DE3CSystem,
            rng: np.random.Generator = None, **kwargs) -> dict:
    """
    Earliest Deadline First (EDF) heuristic.

    Processes tasks in EDF order (earliest deadline first) and assigns each
    to the feasible node with smallest estimated finish time.
    Baker [2005] proves EDF is optimal for acceptance ratio on a single node.
    Extended to multi-node DE3C with greedy node selection.
    """
    T = dag.T
    assignment = np.zeros(T, dtype=int)
    first_cloud = system.D + system.E
    node_free_time = np.zeros(system.total_nodes)

    # Sort tasks by deadline (EDF order) but respect topological order:
    # use topological sort with EDF tie-breaking (from DAGInstance)
    edf_order = sorted(dag.topo_order,
                       key=lambda k: dag.tasks[k].deadline)

    for k in edf_order:
        task = dag.tasks[k]
        feasible = system.feasible_nodes_for_task(task.device)
        best_node = first_cloud
        best_finish = np.inf

        for node_idx in feasible:
            node = system.get_node(node_idx)
            if system.is_device(node_idx):
                est = max(node_free_time[node_idx], 0) + task.compute / node.capacity_ghz
            else:
                est = (node_free_time[node_idx]
                       + task.data_mb / node.transfer_mbps
                       + task.compute / node.capacity_ghz)
            if est < best_finish:
                best_finish = est
                best_node = node_idx

        assignment[k] = best_node
        node = system.get_node(best_node)
        if system.is_device(best_node):
            node_free_time[best_node] += task.compute / node.capacity_ghz
        else:
            node_free_time[best_node] += (task.data_mb / node.transfer_mbps
                                          + task.compute / node.capacity_ghz)

    return _heuristic_result(assignment, dag, system)
