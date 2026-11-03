# MO-PSOGA-DAG

**Multi-Objective Hybrid PSO-GA Algorithm with DAG-Aware Rescheduling for Task Offloading in Device-Edge-Cloud Collaborative Computing**

## Abstract

This repository contains the implementation of MO-PSOGA-DAG, a multi-objective hybrid Particle Swarm Optimisation and Genetic Algorithm for scheduling DAG-structured workflows across a three-tier Device-Edge-Cloud (DE3C) system. The algorithm simultaneously optimises makespan, energy consumption, and task acceptance ratio using NSGA-II-based Pareto selection with DAG-aware repair and rescheduling operators.

## Requirements

- Python 3.11+
- numpy >= 1.23
- scipy >= 1.9
- pandas >= 1.5
- matplotlib >= 3.5

## Installation

```bash
git clone https://github.com/chandrashakera/MO-PSOGA-DAG.git
cd MO-PSOGA-DAG
pip install -r requirements.txt
```

## Usage

Run all experiments (all algorithms, all task scales):
```bash
python experiments/run_experiments.py --workers 6
```

Run a quick smoke test (EDF and FF baselines, T=50 only, seconds to complete):
```bash
python experiments/run_experiments.py --scale 50 --alg EDF FF --workers 1
```

Available flags:

| Flag | Values | Default | Description |
|---|---|---|---|
| `--scale` | 50 100 200 | all | Task scale(s) to run |
| `--alg` | MO_PSOGA NSGA2 MOPSO MOEAD PSOGA_R EDF FF | all | Algorithm(s) to run |
| `--workers` | int | 1 | Parallel workers |
| `--save_dir` | path | data/ | Output directory |

## Results

Pre-computed results for all 1,890 runs (7 algorithms x 3 task scales x 90 runs) are in `data/raw_results_all.csv`.

Key results at T=200 (30 runs, Wilcoxon + Bonferroni corrected, p < 0.05):

| Metric | vs NSGA-II | vs PSOGA_R |
|---|---|---|
| IGD | -90.4% (p<0.001, d=-3.98) | - |
| Energy (F2) | - | -55.3% (d=-2.34) |
| Makespan (F1) | - | -36.8% (d=-4.61) |

Wilcoxon W/T/L = 38/17/35 across 90 pairwise comparisons.

## Repository Structure

```
MO-PSOGA-DAG/
├── algorithms/          # MO-PSOGA-DAG, NSGA-II, MOPSO, MOEA/D, PSOGA_R, EDF, FF
├── environment/         # DE3C system model and DAG generator
├── experiments/         # Experiment runner, config, seed table
├── metrics/             # IGD, HV, GD, Spacing (Pareto metrics)
├── data/                # Raw results (1,890 runs) and summary statistics
└── requirements.txt
```

## Citation

```bibtex
@article{arrabotu2026mopsoga,
  author    = {Arrabotu, Chandra Shaker and Anita, J. P.},
  title     = {Multi-Objective Hybrid {PSO-GA} Algorithm with {DAG}-Aware
               Rescheduling for Task Offloading in Device-Edge-Cloud
               Collaborative Computing},
  journal   = {Journal of Cloud Computing},
  publisher = {Springer},
  year      = {2026},
  note      = {Under review}
}
```

## License

MIT -- see [LICENSE](LICENSE).
