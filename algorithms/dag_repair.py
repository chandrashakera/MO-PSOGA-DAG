"""
dag_repair.py
DAG feasibility repair and rescheduling operators.

Implements Algorithms 1 (DAG_Repair) and 2 (DAG_Reschedule) from the paper.
Applied to ALL algorithms (proposed and baselines) to ensure fair comparison.

Literature basis:
  - DAG repair via topological sort: Hosny et al. [2024]
    doi:10.1016/j.aej.2024.04.038
  - Rescheduling for acceptance ratio: Wang et al. [2024]
    doi:10.1007/s10586-024-04851-3, Section 3.2
  - EDF optimality: Baker [2005] doi:10.1109/TPDS.2005.88
"""

import numpy as np
from environment.de3c_system import DE3CSystem
from environment.dag_generator import DAGInstance
from environment.task_scheduler import compute_finish_times, FinishTimeResult


def dag_repair(assignment: np.ndarray,
               dag: DAGInstance,
               system: DE3CSystem) -> tuple:
    """
    Algorithm 1: DAG_Repair

    Enforce assignment feasibility constraints:
      1. Connectivity: task must be assigned to a feasible node
         (own device, connected ES, or any cloud server).
      2. Compute finish times in topological order and flag
         tasks that violate deadlines.

    Parameters
    ----------
    assignment : np.ndarray shape (T,) — task-to-node mapping (0-indexed)
    dag        : DAGInstance
    system     : DE3CSystem

    Returns
    -------
    assignment  : repaired assignment (infeasible nodes replaced with cloud)
    result      : FinishTimeResult after repair
    violating   : list of task indices with f_k > d_k
    """
    T = dag.T
    assignment = assignment.copy()

    # ── Step 1: Enforce connectivity constraints (Eqs. 4-5) ──────────────────
    # If assigned node is not feasible for this task, redirect to first
    # available cloud server (always feasible: all devices reach all cloud).
    first_cloud = system.D + system.E   # global index of CS 0
    for k in range(T):
        feasible = system.feasible_nodes_for_task(dag.tasks[k].device)
        if int(assignment[k]) not in feasible:
            assignment[k] = first_cloud

    # ── Step 2: Compute finish times and identify deadline violations ─────────
    result = compute_finish_times(assignment, dag, system)

    violating = [k for k in range(T)
                 if result.finish_times[k] > dag.tasks[k].deadline]

    return assignment, result, violating


def dag_reschedule(assignment: np.ndarray,
                   violating: list,
                   dag: DAGInstance,
                   system: DE3CSystem) -> tuple:
    """
    Algorithm 2: DAG_Reschedule

    Attempt to reschedule deadline-violating tasks to alternative nodes
    that can meet the deadline while satisfying successor constraints.

    Extensions over Wang et al. [2024] rescheduling:
      - Successor-deadline checking: reassigning t_k must not push any
        successor t_j past its deadline (DAG-aware constraint).
      - EDF ordering: process violating tasks in ascending deadline order
        (Baker [2005] optimality result).

    Parameters
    ----------
    assignment : np.ndarray shape (T,) — current assignment (post-repair)
    violating  : list of violating task indices
    dag        : DAGInstance
    system     : DE3CSystem

    Returns
    -------
    assignment : updated assignment (some violations may be resolved)
    result     : FinishTimeResult after rescheduling
    """
    if not violating:
        result = compute_finish_times(assignment, dag, system)
        return assignment, result

    assignment = assignment.copy()

    # Sort violating tasks by deadline (EDF — Baker 2005)
    violating_sorted = sorted(violating, key=lambda k: dag.tasks[k].deadline)

    for k in violating_sorted:
        task = dag.tasks[k]
        current_node = int(assignment[k])
        feasible_nodes = system.feasible_nodes_for_task(task.device)

        # Try each alternative node (first-fit, as in Wang et al. 2024)
        for candidate_node in feasible_nodes:
            if candidate_node == current_node:
                continue

            # Tentative reassignment
            assignment_trial = assignment.copy()
            assignment_trial[k] = candidate_node

            # Evaluate tentative schedule
            trial_result = compute_finish_times(assignment_trial, dag, system)
            f_k_new = trial_result.finish_times[k]

            # Condition 1: task k now meets its deadline
            if f_k_new > task.deadline:
                continue

            # Condition 2 (DAG-aware): no successor deadline violated
            # by the rescheduled task k
            successor_ok = True
            for j in dag.succ[k]:
                if trial_result.finish_times[j] > dag.tasks[j].deadline:
                    # Only reject if this successor was previously OK
                    # (rescheduling of k should not worsen previously OK tasks)
                    orig_result = compute_finish_times(assignment, dag, system)
                    if orig_result.finish_times[j] <= dag.tasks[j].deadline:
                        successor_ok = False
                        break

            if successor_ok:
                assignment = assignment_trial
                break  # first-fit: take first valid node (Wang et al. 2024)

    # Final evaluation after all rescheduling attempts
    result = compute_finish_times(assignment, dag, system)
    return assignment, result


def repair_and_reschedule(assignment: np.ndarray,
                          dag: DAGInstance,
                          system: DE3CSystem) -> tuple:
    """
    Combined repair + reschedule pipeline applied after each PSO/GA update.

    Returns
    -------
    assignment : feasible assignment
    result     : FinishTimeResult
    """
    assignment, result, violating = dag_repair(assignment, dag, system)
    assignment, result = dag_reschedule(assignment, violating, dag, system)
    return assignment, result
