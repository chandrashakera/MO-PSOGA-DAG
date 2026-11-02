"""
pareto_utils.py
NSGA-II Pareto utilities shared by MO-PSOGA-DAG, NSGA-II, MOPSO, MOEA/D.

Implements:
  - Fast non-dominated sorting  (Deb et al. 2002, Algorithm 1)
  - Crowding distance           (Deb et al. 2002, Algorithm 2)
  - Dominance check             (paper Eq. 17)
  - gbest selection             (paper Eq. 21)
  - pbest update                (paper Eq. 22)

Reference: Deb et al. [2002] doi:10.1109/4235.996017
"""

import numpy as np
from typing import List, Tuple


# ── Dominance ──────────────────────────────────────────────────────────────────

def dominates(obj_a: np.ndarray, obj_b: np.ndarray) -> bool:
    """
    Return True if solution a Pareto-dominates solution b.

    a ≺ b  iff  F_l(a) ≤ F_l(b) ∀l  and  ∃l: F_l(a) < F_l(b)
    (paper Eq. 17; all objectives are minimised)
    """
    return (np.all(obj_a <= obj_b) and np.any(obj_a < obj_b))


def crowded_better(rank_a: int, cd_a: float,
                   rank_b: int, cd_b: float) -> bool:
    """
    NSGA-II crowded-comparison operator (Deb et al. 2002).
    Returns True if solution a is preferred over b.
    """
    if rank_a < rank_b:
        return True
    if rank_a == rank_b and cd_a > cd_b:
        return True
    return False


# ── Non-dominated sorting ──────────────────────────────────────────────────────

def fast_non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """
    Fast non-dominated sorting (Deb et al. 2002, Algorithm 1).

    Parameters
    ----------
    objectives : np.ndarray, shape (N, m)
        Objective values for N solutions, m objectives (all minimised).

    Returns
    -------
    fronts : list of lists
        fronts[0] = rank-1 (non-dominated) solution indices, etc.
    Time complexity: O(m N^2)  (Deb et al. 2002)
    """
    N = len(objectives)
    dominated_by_count = np.zeros(N, dtype=int)   # n_p: # solutions dominating p
    dominates_set = [[] for _ in range(N)]         # S_p: solutions p dominates

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                dominates_set[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                dominated_by_count[p] += 1

    fronts = []
    current_front = [p for p in range(N) if dominated_by_count[p] == 0]
    while current_front:
        fronts.append(current_front)
        next_front = []
        for p in current_front:
            for q in dominates_set[p]:
                dominated_by_count[q] -= 1
                if dominated_by_count[q] == 0:
                    next_front.append(q)
        current_front = next_front

    return fronts


def get_ranks(objectives: np.ndarray) -> np.ndarray:
    """Return rank array (1-indexed) for each solution."""
    N = len(objectives)
    ranks = np.zeros(N, dtype=int)
    fronts = fast_non_dominated_sort(objectives)
    for rank, front in enumerate(fronts, start=1):
        for i in front:
            ranks[i] = rank
    return ranks


# ── Crowding distance ──────────────────────────────────────────────────────────

def crowding_distance(objectives: np.ndarray,
                      front_indices: List[int]) -> np.ndarray:
    """
    Crowding distance for solutions in one front (Deb et al. 2002, Eq. 20).

    I_d(i) = ∑_l [F_l^{(i+1)} - F_l^{(i-1)}] / [F_l^max - F_l^min]

    Boundary solutions receive inf.

    Parameters
    ----------
    objectives    : (N, m) objective array for full population
    front_indices : indices of solutions in this front

    Returns
    -------
    distances : crowding distances for each solution in front_indices
                (same order as front_indices)
    """
    n = len(front_indices)
    if n <= 2:
        return np.full(n, np.inf)

    m = objectives.shape[1]
    dist = np.zeros(n)
    obj_front = objectives[front_indices]  # shape (n, m)

    for l in range(m):
        sorted_idx = np.argsort(obj_front[:, l])
        dist[sorted_idx[0]] = np.inf
        dist[sorted_idx[-1]] = np.inf
        f_min = obj_front[sorted_idx[0], l]
        f_max = obj_front[sorted_idx[-1], l]
        denom = f_max - f_min if (f_max - f_min) > 1e-12 else 1e-12
        for i in range(1, n - 1):
            dist[sorted_idx[i]] += (
                (obj_front[sorted_idx[i + 1], l] -
                 obj_front[sorted_idx[i - 1], l]) / denom
            )
    return dist


def compute_all_crowding_distances(objectives: np.ndarray,
                                   fronts: List[List[int]]) -> np.ndarray:
    """
    Compute crowding distances for all solutions across all fronts.

    Returns cd array of shape (N,).
    """
    N = len(objectives)
    cd = np.zeros(N)
    for front in fronts:
        if not front:
            continue
        d = crowding_distance(objectives, front)
        for i, idx in enumerate(front):
            cd[idx] = d[i]
    return cd


# ── NSGA-II selection ──────────────────────────────────────────────────────────

def nsga2_select(objectives: np.ndarray, N: int) -> np.ndarray:
    """
    Select N solutions from a population using NSGA-II crowded comparison.

    Fills front by front; last front added partially by crowding distance.
    Returns array of selected indices (length N).
    """
    fronts = fast_non_dominated_sort(objectives)
    cd = compute_all_crowding_distances(objectives, fronts)

    selected = []
    for front in fronts:
        if len(selected) + len(front) <= N:
            selected.extend(front)
        else:
            # Sort remaining front by crowding distance (descending)
            remaining = N - len(selected)
            front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
            selected.extend(front_sorted[:remaining])
            break

    return np.array(selected, dtype=int)


# ── Global best selection (paper Eq. 21) ──────────────────────────────────────

def select_gbest(archive_objectives: np.ndarray) -> int:
    """
    Select global best from Pareto archive as the rank-1 solution
    with maximum crowding distance (paper Eq. 21).

    gb = argmax_{x ∈ A, r(x)=1} I_d(x)

    Returns index into archive_objectives.
    """
    N = len(archive_objectives)
    if N == 0:
        return 0

    fronts = fast_non_dominated_sort(archive_objectives)
    rank1 = fronts[0]   # non-dominated solutions

    if len(rank1) == 1:
        return rank1[0]

    cd = crowding_distance(archive_objectives, rank1)
    best_local = int(np.argmax(cd))
    return rank1[best_local]


# ── Archive management ────────────────────────────────────────────────────────

def update_archive(archive_pos: np.ndarray,
                   archive_obj: np.ndarray,
                   new_pos: np.ndarray,
                   new_obj: np.ndarray,
                   max_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add new solutions to archive, retain only non-dominated ones,
    then prune to max_size by removing solutions with smallest
    crowding distance.

    Returns updated (archive_pos, archive_obj).
    """
    if len(archive_pos) == 0:
        combined_pos = new_pos
        combined_obj = new_obj
    else:
        combined_pos = np.vstack([archive_pos, new_pos])
        combined_obj = np.vstack([archive_obj, new_obj])

    # Keep non-dominated only (rank 1)
    fronts = fast_non_dominated_sort(combined_obj)
    nd_idx = fronts[0]

    nd_pos = combined_pos[nd_idx]
    nd_obj = combined_obj[nd_idx]

    # Prune if over max_size: remove smallest crowding distance
    if len(nd_idx) > max_size:
        cd = crowding_distance(nd_obj, list(range(len(nd_obj))))
        keep_idx = np.argsort(cd)[::-1][:max_size]
        nd_pos = nd_pos[keep_idx]
        nd_obj = nd_obj[keep_idx]

    return nd_pos, nd_obj
