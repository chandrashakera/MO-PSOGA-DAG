"""
task_scheduler.py  — vectorised rewrite for performance
Implements Wang et al. [2024] Eqs 6–12, extended with DAG ready time.

Key optimisation over v1:
  - feasible_nodes_for_task() result cached at module level
  - compute_finish_times() uses pre-sorted topological order (no re-sort)
  - Node type tests replaced with precomputed boolean arrays
  - numpy used for array ops wherever possible
All results are numerically identical to v1 — only speed changes.
"""
import numpy as np
from environment.de3c_system import DE3CSystem
from environment.dag_generator import DAGInstance


class FinishTimeResult:
    __slots__ = ('finish_times', 'accepted', 'N', 'makespan',
                 'energy_device', 'energy_edge', 'acceptance_ratio',
                 'cloud_cost')
    def __init__(self, T: int):
        self.finish_times    = np.zeros(T)
        self.accepted        = np.zeros(T, dtype=bool)
        self.N               = 0
        self.makespan        = 0.0
        self.energy_device   = 0.0
        self.energy_edge     = 0.0
        self.acceptance_ratio = 0.0
        self.cloud_cost      = 0.0


def _precompute(dag: DAGInstance, system: DE3CSystem):
    """Cache per-task attributes as numpy arrays for fast indexing."""
    T   = dag.T
    N_n = system.total_nodes

    # Task arrays
    compute  = np.array([dag.tasks[k].compute  for k in range(T)])
    data_mb  = np.array([dag.tasks[k].data_mb  for k in range(T)])
    deadline = np.array([dag.tasks[k].deadline for k in range(T)])
    device   = np.array([dag.tasks[k].device   for k in range(T)], dtype=int)

    # Node arrays
    capacity = np.array([system.nodes[i].capacity_ghz  for i in range(N_n)])
    rate     = np.array([system.nodes[i].transfer_mbps for i in range(N_n)])
    rate     = np.where(rate < 1e-9, 1e-9, rate)   # avoid div-by-zero

    # Node type masks
    is_dev   = np.array([system.is_device(i) for i in range(N_n)])
    is_edge  = np.array([system.is_edge(i)   for i in range(N_n)])
    is_cloud = np.array([system.is_cloud(i)  for i in range(N_n)])

    # Per-task feasible node list (cached)
    feasible = [system.feasible_nodes_for_task(device[k]) for k in range(T)]
    first_cloud = system.D + system.E

    return (compute, data_mb, deadline, device,
            capacity, rate, is_dev, is_edge, is_cloud,
            feasible, first_cloud)


# Cache per (dag id, system id) to avoid recomputing across runs
_CACHE: dict = {}

def _get_cache(dag, system):
    key = (id(dag), id(system))
    if key not in _CACHE:
        _CACHE[key] = _precompute(dag, system)
    return _CACHE[key]


def compute_finish_times(assignment: np.ndarray,
                         dag: DAGInstance,
                         system: DE3CSystem) -> FinishTimeResult:
    """
    Compute finish times for all tasks.  Vectorised where possible;
    sequential task loop is unavoidable due to DAG dependencies.
    """
    (compute, data_mb, deadline, dev_of_task,
     capacity, rate, is_dev, is_edge, is_cloud,
     feasible_list, first_cloud) = _get_cache(dag, system)

    T   = dag.T
    N_n = system.total_nodes
    result = FinishTimeResult(T)
    ft = result.finish_times          # shape (T,)

    node_compute_done   = np.zeros(N_n)
    node_transfer_accum = np.zeros(N_n)   # cumulative transfer to each ES/cloud

    for k in dag.topo_order:          # topological order, EDF-within-level
        node  = int(assignment[k])
        proc  = compute[k] / capacity[node]
        r_k   = rate[node]

        # ── Ready time from DAG predecessors (Eq. 6) ─────────────────────
        rt = 0.0
        for j in dag.pred[k]:
            fj = ft[j]
            if int(assignment[j]) != node and r_k > 1e-9:
                fj += data_mb[k] / r_k
            if fj > rt:
                rt = fj

        # ── Finish time (Eqs 8–11) ────────────────────────────────────────
        if is_dev[node]:
            # Local device: no transfer, sequential compute
            start = max(rt, node_compute_done[node])
            ft[k] = start + proc
            node_compute_done[node] = ft[k]
        else:
            # Edge / cloud: pipelined transfer + compute
            node_transfer_accum[node] += data_mb[k] / r_k
            f_R = node_transfer_accum[node] + proc          # transfer bottleneck
            f_C = node_compute_done[node]   + proc          # compute bottleneck
            ft[k] = max(rt, max(f_R, f_C))
            node_compute_done[node] = ft[k]

    # ── Metrics ───────────────────────────────────────────────────────────
    accepted = ft <= deadline
    result.accepted = accepted
    result.N        = int(accepted.sum())
    result.acceptance_ratio = result.N / T

    accepted_ft = ft[accepted]
    result.makespan = float(accepted_ft.max()) if result.N > 0 else 0.0

    result.energy_device = _energy_device(assignment, compute, data_mb,
                                           dev_of_task, is_dev, system, T)
    result.energy_edge   = _energy_edge(assignment, compute, ft,
                                         is_edge, system, T, node_compute_done)
    result.cloud_cost    = _cloud_cost(assignment, ft, is_cloud, system, T)

    return result


def _energy_device(assignment, compute, data_mb, dev_of_task,
                   is_dev, system, T) -> float:
    energy = 0.0
    for k in range(T):
        d = dev_of_task[k]
        node = int(assignment[k])
        nd = system.nodes[d]
        if node == d:
            energy += nd.power_compute * compute[k]
        else:
            energy += nd.power_transmit * data_mb[k]
    return energy


def _energy_edge(assignment, compute, ft, is_edge, system, T,
                  node_compute_done) -> float:
    energy = 0.0
    for e_local in range(system.E):
        e_g  = system.D + e_local
        node = system.nodes[e_g]
        if node.power_max == 0:
            continue
        busy = sum(compute[k] / node.capacity_ghz
                   for k in range(T) if int(assignment[k]) == e_g)
        if busy == 0:
            continue
        span = node_compute_done[e_g]
        util = min(busy / max(span, 1e-9), 1.0)
        energy += node.power_idle * span + (node.power_max - node.power_idle) * util * span
    return energy


def _cloud_cost(assignment, ft, is_cloud, system, T) -> float:
    cost = 0.0
    for v_local in range(system.V):
        v_g  = system.D + system.E + v_local
        node = system.nodes[v_g]
        tasks = [k for k in range(T) if int(assignment[k]) == v_g]
        if tasks:
            cost += np.ceil(max(ft[k] for k in tasks)) * node.price_per_sec
    return cost


def objectives_from_result(result: FinishTimeResult) -> np.ndarray:
    return np.array([
        result.makespan,
        result.energy_device + result.energy_edge,
        -result.acceptance_ratio
    ])
