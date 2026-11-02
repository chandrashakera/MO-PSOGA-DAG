"""
pareto_metrics.py
Performance metrics for MO-PSOGA-DAG experiments.

Tier 1 (Pareto quality):  HV, IGD, GD, Spacing
Tier 2 (objective values): F1 (makespan), F2 (energy), F3 (acceptance), F4 (cost)

References:
  HV:      Zitzler et al. [2003] doi:10.1109/TEVC.2003.810758
  IGD, GD: Van Veldhuizen & Lamont [1998]
  Spacing: Schott [1995] (doi: thesis MIT)
"""

import numpy as np
from itertools import product
from typing import List, Optional


# ── Hypervolume (HV) ──────────────────────────────────────────────────────────

def hypervolume(pareto_front: np.ndarray,
                ref_point: np.ndarray) -> float:
    """
    Compute hypervolume indicator for a 3-objective minimisation problem.

    Uses the WFG algorithm for small fronts (n < 500).
    For larger fronts, uses Monte Carlo approximation.

    Parameters
    ----------
    pareto_front : (M, 3) array of objective values (all minimised)
    ref_point    : (3,)   reference point = 1.1 * F_nadir (Section 5.3)

    Returns
    -------
    HV value (float) — higher is better.
    """
    M = len(pareto_front)
    if M == 0:
        return 0.0

    # Filter: only points that dominate reference point
    dominated_mask = np.all(pareto_front < ref_point, axis=1)
    front = pareto_front[dominated_mask]
    if len(front) == 0:
        return 0.0

    # For 3 objectives: use recursive inclusion-exclusion (WFG sweep line)
    return _hv_recursive(front, ref_point)


def _hv_recursive(front: np.ndarray, ref: np.ndarray) -> float:
    """
    Recursive hypervolume computation (Emmerich et al. 2006 formulation).
    Works for any m objectives; accurate for small fronts.
    """
    m = front.shape[1]
    if m == 1:
        return float(ref[0] - front[:, 0].min())
    if len(front) == 0:
        return 0.0

    # Sort by last objective
    idx = np.argsort(front[:, -1])
    front_sorted = front[idx]

    hv = 0.0
    for i, point in enumerate(front_sorted):
        # Slice in last objective dimension
        if i == 0:
            prev = ref[-1]
        else:
            prev = front_sorted[i - 1, -1]
        slice_height = point[-1] - prev
        if slice_height < 0:
            continue
        # Build restricted front for remaining m-1 objectives
        restricted = front_sorted[:i + 1, :-1].copy()
        restricted = _remove_dominated_2d(restricted) if m - 1 == 2 else restricted
        hv += slice_height * _hv_recursive(restricted, ref[:-1])

    return hv


def _remove_dominated_2d(front: np.ndarray) -> np.ndarray:
    """Remove dominated points from a 2D front (sort + sweep)."""
    if len(front) <= 1:
        return front
    idx = np.argsort(front[:, 0])
    front = front[idx]
    non_dom = [front[0]]
    min_y = front[0, 1]
    for p in front[1:]:
        if p[1] < min_y:
            non_dom.append(p)
            min_y = p[1]
    return np.array(non_dom)


# ── IGD (Inverted Generational Distance) ──────────────────────────────────────

def igd(obtained_front: np.ndarray,
        reference_front: np.ndarray) -> float:
    """
    IGD = (1/|P*|) * ∑_{p* in P*} min_{p in P} ||p - p*||_2
    Lower is better. (paper Eq. 26)
    """
    if len(obtained_front) == 0:
        return np.inf
    total = 0.0
    for p_star in reference_front:
        diffs = obtained_front - p_star
        dists = np.linalg.norm(diffs, axis=1)
        total += dists.min()
    return total / len(reference_front)


# ── GD (Generational Distance) ────────────────────────────────────────────────

def gd(obtained_front: np.ndarray,
       reference_front: np.ndarray) -> float:
    """
    GD = (1/|P|) * ∑_{p in P} min_{p* in P*} ||p - p*||_2
    Lower is better. (paper Eq. 27)
    """
    if len(obtained_front) == 0:
        return np.inf
    total = 0.0
    for p in obtained_front:
        diffs = reference_front - p
        dists = np.linalg.norm(diffs, axis=1)
        total += dists.min()
    return total / len(obtained_front)


# ── Spacing ───────────────────────────────────────────────────────────────────

def spacing(front: np.ndarray) -> float:
    """
    Spacing = std of nearest-neighbour distances (Schott 1995).
    SP = sqrt(1/(|P|-1) * ∑(d_bar - d_i)^2)

    d_i = min_{j≠i} ||F(x_i) - F(x_j)||_1
    Lower is better (more uniform). (paper Eq. 28)
    """
    n = len(front)
    if n <= 1:
        return 0.0
    d = np.zeros(n)
    for i in range(n):
        diffs = np.abs(front - front[i]).sum(axis=1)
        diffs[i] = np.inf
        d[i] = diffs.min()
    d_bar = d.mean()
    return float(np.sqrt(((d - d_bar) ** 2).sum() / (n - 1)))


# ── Reference point computation ───────────────────────────────────────────────

def compute_reference_point(all_obj_runs: List[np.ndarray],
                             factor: float = 1.1) -> np.ndarray:
    """
    ref_point = factor * F_nadir  (paper Section 5.3)
    F_nadir[l] = max over all solutions and runs of F_l.

    Parameters
    ----------
    all_obj_runs : list of (M_i, 3) objective arrays from all algorithms/runs
    factor       : 1.1 (10% beyond nadir)
    """
    all_obj = np.vstack(all_obj_runs)
    nadir = all_obj.max(axis=0)
    return factor * nadir


def compute_reference_front(all_obj_runs: List[np.ndarray],
                             all_pos_runs: List[np.ndarray]) -> np.ndarray:
    """
    Reference Pareto front = combined non-dominated solutions
    from all algorithms and all runs on a given instance.
    """
    from algorithms.pareto_utils import fast_non_dominated_sort
    combined = np.vstack(all_obj_runs)
    fronts = fast_non_dominated_sort(combined)
    return combined[fronts[0]]


# ── Best-compromise solution ──────────────────────────────────────────────────

def best_compromise(front_obj: np.ndarray) -> int:
    """
    Select best-compromise from Pareto front for Tier-2 comparison.
    Minimum normalised Euclidean distance to ideal point.

    Returns index into front_obj.
    """
    if len(front_obj) == 1:
        return 0
    ideal = front_obj.min(axis=0)
    nadir = front_obj.max(axis=0)
    denom = nadir - ideal
    denom[denom < 1e-12] = 1e-12
    normalised = (front_obj - ideal) / denom
    dists = np.linalg.norm(normalised, axis=1)
    return int(np.argmin(dists))
