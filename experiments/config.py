"""
config.py
All experimental parameters for MO-PSOGA-DAG paper.

Sources:
  - Algorithm params: Wang et al. [2024] Table 2 (doi:10.1007/s10586-024-04851-3)
  - System params:    Wang et al. [2024] Table 1
  - Stat tests:       Derrac et al. [2011] (doi:10.1016/j.swevo.2011.02.002)
"""

import numpy as np

# ─── DE3C system ──────────────────────────────────────────────────────────────
SYSTEM = dict(
    D=10,    # user devices
    E=5,     # edge servers
    V=10,    # cloud server types
)

# Device parameters  (Wang et al. 2024, Table 1)
DEVICE = dict(
    cores_range=(2, 8),              # cores per device
    capacity_ghz_range=(1.8, 2.5),  # GHz per core
    power_compute_W_GHz=0.5,        # p_i^c  (W / GHz·s)
    power_transmit_W_MB=0.1,        # p_i^r  (W / MB)
)

# Edge server parameters
EDGE = dict(
    cores_range=(4, 32),
    capacity_ghz_range=(1.8, 3.0),
    transfer_rate_mbps_mean=100.0,
    transfer_rate_mbps_std=20.0,
    power_max_W=200.0,              # P_max for linear energy model (Katal 2023)
    power_idle_W=100.0,             # P_idle
)

# Cloud server parameters  (AWS EC2 reference, Wang et al. 2024)
CLOUD = dict(
    cores_range=(1, 8),
    capacity_ghz_range=(1.8, 3.0),
    transfer_rate_mbps_mean=10.0,
    transfer_rate_mbps_std=2.0,
    price_per_sec_range=(0.01, 0.05),  # h_i  monetary units/s
)

# Task parameters  (Wang et al. 2024, Table 1)
TASK = dict(
    compute_ghz_range=(0.5, 1.2),   # c_k  GHz·s
    data_mb_range=(1.5, 6.0),       # m_k  MB
    deadline_s_range=(1.0, 5.0),    # d_k  seconds
)

# DAG structure  (three instances per scale, edge densities)
DAG = dict(
    instances=3,
    edge_density_values=[1.0, 1.5, 2.0],  # |E|/T per instance
)

# Task scales
TASK_SCALES = [50, 100, 200]

# ─── Algorithm parameters (Wang et al. 2024, Table 2) ────────────────────────
ALG = dict(
    N=100,            # population size  — ALL algorithms
    I=100,            # max iterations   — ALL algorithms
    budget=10_000,    # N × I = 10000    — fairness check

    # PSO
    omega_max=1.2,    # inertia weight upper bound
    omega_min=0.4,    # inertia weight lower bound
    c1=2.0,           # cognitive coefficient
    c2=2.0,           # social coefficient

    # GA
    p_c=0.8,          # crossover probability
    p_m=0.1,          # mutation probability

    # NSGA-II / Archive
    archive_size=100, # Pareto archive size (MO-PSOGA-DAG)

    # Initialisation (cloud-biased, Wang et al. 2024 Sec 4.1)
    p_cloud=0.50,     # P(cloud assignment)
    p_edge=0.30,      # total P(edge) — split equally across E servers
    p_device=0.20,    # P(local device)

    # MOEA/D
    T_nb=20,          # neighbourhood size (Zhang & Li 2007)
)

# Fairness assertion
assert ALG['N'] * ALG['I'] == ALG['budget'], "Budget mismatch — check N and I"

# ─── Experiment protocol ──────────────────────────────────────────────────────
EXPERIMENT = dict(
    n_runs=30,        # independent runs per instance
    master_seed=42,   # seed for generating run seeds (Eq. 7 in paper)
    n_instances=3,    # DE3C instances per scale
    # total runs per algorithm per scale = n_instances × n_runs = 90
    # grand total per algorithm = 3 scales × 90 = 270
)

# ─── Pareto metric settings ───────────────────────────────────────────────────
METRICS = dict(
    hv_nadir_factor=1.1,   # ref point = 1.1 × F_nadir (Section 5.3)
    # Reference front: combined non-dominated solutions of all algorithms
)

# ─── Statistical test settings ───────────────────────────────────────────────
STATS = dict(
    alpha=0.05,
    n_comparisons=6,               # pairwise vs 6 baselines
    alpha_corrected=0.05 / 6,      # Bonferroni ≈ 0.0083
    # Cohen d thresholds (Cohen 1988)
    d_small=0.2,
    d_medium=0.5,
    d_large=0.8,
)

# ─── Algorithm names (canonical) ─────────────────────────────────────────────
ALG_NAMES = [
    'MO-PSOGA-DAG',   # proposed
    'NSGA-II',        # Deb et al. 2002
    'MOPSO',          # Coello et al. 2004
    'MOEAD',          # Zhang & Li 2007
    'PSOGA_R',        # Wang et al. 2024
    'FF',             # First Fit heuristic
    'EDF',            # Earliest Deadline First heuristic
]
