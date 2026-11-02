"""
dag_generator.py
Random DAG task generation for MO-PSOGA-DAG experiments.

Produces three DAG instances per task scale with edge densities
[1.0, 1.5, 2.0] × T edges.  Acyclicity is guaranteed by only
adding edges from lower to higher task index (standard DAG
construction method; Hosny et al. [2024] doi:10.1016/j.aej.2024.04.038).

Task parameters from Wang et al. [2024] Table 1:
  c_k  in [0.5, 1.2]  GHz·s
  m_k  in [1.5, 6.0]  MB
  d_k  in [1.0, 5.0]  s
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Set


@dataclass
class Task:
    """A single computation task in the DAG workflow."""
    idx: int              # task index 0 ... T-1
    compute: float        # c_k  GHz·s
    data_mb: float        # m_k  MB
    deadline: float       # d_k  seconds
    device: int           # launching device index (b_{i,k} = 1)


class DAGInstance:
    """
    A DAG workflow instance.

    Attributes
    ----------
    tasks      : list of Task objects (length T)
    pred       : pred[k] = set of immediate predecessor indices of task k
    succ       : succ[k] = set of immediate successor indices of task k
    topo_order : valid topological ordering (list of task indices)
    """

    def __init__(self, tasks: List[Task],
                 pred: List[Set[int]],
                 succ: List[Set[int]]):
        self.tasks = tasks
        self.pred = pred
        self.succ = succ
        self.topo_order = self._topological_sort()
        self.T = len(tasks)

    def _topological_sort(self) -> List[int]:
        """
        Kahn's algorithm for topological sort.
        Returns a valid processing order respecting all precedence edges.
        """
        T = len(self.tasks)
        in_degree = [len(self.pred[k]) for k in range(T)]
        queue = [k for k in range(T) if in_degree[k] == 0]
        order = []
        while queue:
            # Among ready tasks, process in deadline order (EDF-consistent)
            queue.sort(key=lambda k: self.tasks[k].deadline)
            k = queue.pop(0)
            order.append(k)
            for s in self.succ[k]:
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    queue.append(s)
        assert len(order) == T, "DAG has a cycle — generation error"
        return order

    def get_predecessors(self, k: int) -> Set[int]:
        return self.pred[k]

    def get_successors(self, k: int) -> Set[int]:
        return self.succ[k]

    def __repr__(self) -> str:
        n_edges = sum(len(s) for s in self.succ)
        return f"DAGInstance(T={self.T}, edges={n_edges})"


def generate_dag_instance(T: int, edge_density: float,
                          D: int, rng: np.random.Generator) -> DAGInstance:
    """
    Generate one random DAG instance with approximately edge_density × T edges.

    Acyclicity: edges only go from lower to higher index (i → j iff i < j).
    This is the standard Erdős–Rényi DAG model used in scheduling literature
    (Hosny et al. [2024]).

    Task parameters follow Wang et al. [2024] Table 1.
    Each task is assigned to one launching device (uniform random).
    """
    # ── Generate tasks ────────────────────────────────────────────────────────
    tasks = []
    for k in range(T):
        tasks.append(Task(
            idx=k,
            compute=rng.uniform(0.5, 1.2),    # c_k  GHz·s
            data_mb=rng.uniform(1.5, 6.0),    # m_k  MB
            deadline=rng.uniform(1.0, 5.0),   # d_k  seconds
            device=int(rng.integers(0, D)),   # launching device
        ))

    # ── Generate edges ────────────────────────────────────────────────────────
    # Target number of edges: edge_density × T
    target_edges = int(edge_density * T)

    # All possible forward edges (i < j): total = T*(T-1)/2
    # Sample without replacement to reach target
    all_possible = [(i, j) for i in range(T) for j in range(i + 1, T)]
    if target_edges > len(all_possible):
        target_edges = len(all_possible)

    chosen_indices = rng.choice(len(all_possible), size=target_edges,
                                replace=False)
    edges = [all_possible[idx] for idx in chosen_indices]

    # ── Build predecessor / successor sets ────────────────────────────────────
    pred: List[Set[int]] = [set() for _ in range(T)]
    succ: List[Set[int]] = [set() for _ in range(T)]
    for (i, j) in edges:
        pred[j].add(i)
        succ[i].add(j)

    # ── Transitive reduction (remove redundant edges for cleaner DAG) ─────────
    # An edge (i,j) is redundant if there is another path from i to j.
    # We skip full transitive reduction for performance; the DAG is valid as-is.

    return DAGInstance(tasks=tasks, pred=pred, succ=succ)


def generate_all_instances(T: int, D: int,
                           rng: np.random.Generator,
                           densities=(1.0, 1.5, 2.0)) -> List[DAGInstance]:
    """
    Generate all DAG instances for one task scale.

    Returns a list of len(densities) DAGInstance objects.
    The same rng is used sequentially so results are reproducible.
    """
    instances = []
    for density in densities:
        inst = generate_dag_instance(T=T, edge_density=density, D=D, rng=rng)
        instances.append(inst)
    return instances


# ── Quick validation ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    for T in [200, 500, 1000]:
        instances = generate_all_instances(T=T, D=10, rng=rng)
        for density, inst in zip([1.0, 1.5, 2.0], instances):
            n_edges = sum(len(s) for s in inst.succ)
            print(f"T={T:4d}  density={density}  edges={n_edges:5d}  "
                  f"topo_ok={len(inst.topo_order)==T}")
