"""
seed_table.py
Generate the fixed table of 30 per-run seeds from master seed 42.

  {s_1,...,s_30} = RNG_42.integers(0, 2^31, size=30)   [Eq. 7, paper]

The SAME seed s_ell is applied to ALL algorithms for run ell,
ensuring identical initial populations and DAG instances.
"""

import numpy as np


def make_seed_table(master_seed: int = 42, n_runs: int = 30) -> np.ndarray:
    """Return array of n_runs integer seeds derived from master_seed."""
    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(0, 2**31, size=n_runs)
    return seeds


# Pre-computed table (for import convenience)
SEED_TABLE = make_seed_table(master_seed=42, n_runs=30)


if __name__ == "__main__":
    print("Seed table (30 seeds, master=42):")
    for i, s in enumerate(SEED_TABLE):
        print(f"  Run {i+1:2d}: {s}")
