# MO-PSOGA-DAG: Multi-Objective Hybrid PSO–GA with DAG-Aware Rescheduling

> **Paper:** "Multi-Objective Hybrid PSO–GA Algorithm with DAG-Aware Rescheduling for Task Offloading in Device–Edge–Cloud Collaborative Computing"  
> **Authors:** Chandra Shaker Arrabotu, Anita J P  
> **Institution:** Amrita School of Engineering, Amrita Vishwa Vidyapeetham, Coimbatore, India  
> **Submitted to:** Journal of Cloud Computing (Springer)

---

## Repository Structure

```
MO-PSOGA-DAG/
├── README.md
├── requirements.txt
├── code/
│   ├── run_experiments.py          # Main experiment runner (all algorithms)
│   ├── task_environment_v3_FINAL.py  # DE3C system + DAG task model
│   └── pareto_metrics.py           # HV, IGD, GD, Spacing computation
└── data/
    ├── raw_results_all.csv         # All 1,890 runs (7 algorithms × 3 scales × 90 runs)
    ├── raw_T50_inst1.csv           # Per-instance raw results
    ├── raw_T50_inst2.csv
    ├── raw_T50_inst3.csv
    ├── raw_T100_inst1.csv
    ├── raw_T100_inst2.csv
    ├── raw_T100_inst3.csv
    ├── raw_T200_inst1.csv
    ├── raw_T200_inst2.csv
    ├── raw_T200_inst3.csv
    └── summary_mean_std.csv        # Aggregated mean ± std per algorithm per scale
```

---

## Requirements

- Python 3.10+
- numpy >= 1.23
- scipy >= 1.9
- pandas >= 1.5

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Reproducing the Experiments

### Full experiment (all algorithms, all scales)
```bash
python code/run_experiments.py
```
This reproduces all 1,890 runs using the fixed seed table (master seed = 42).  
**Note:** Full reproduction takes approximately 443 hours (sequential) on an Intel Core i5-12400. To reproduce a single scale:

```bash
python code/run_experiments.py --T 200 --algorithm MO-PSOGA-DAG
```

### Computing Pareto metrics from existing results
```bash
python code/pareto_metrics.py --input data/raw_results_all.csv
```

---

## Experimental Setup Summary

| Parameter | Value |
|---|---|
| Task scales T | 50, 100, 200 |
| Instances per scale | 3 (DAG density 1.0, 1.5, 2.0) |
| Runs per instance | 30 |
| Total runs per algorithm | 270 |
| Population size N | 100 |
| Max iterations I | 100 |
| Master random seed | 42 |

---

## Results Overview (T = 200, primary scale)

| Algorithm | HV | IGD | F1 (s) | F2 (J) | F3 (%) |
|---|---|---|---|---|---|
| **MO-PSOGA-DAG** | **1433.1** | **74.85** | **2.60** | **869.2** | 64.6 |
| NSGA-II | 99.5 | 776.0 | 3.37 | 1398.8 | 88.8 |
| MOPSO | 3030.1 | 109.5 | 2.65 | 1883.7 | 63.2 |
| MOEA/D | 193.7 | 476.5 | 3.70 | 1359.9 | 79.7 |
| PSOGA_R | — | — | 4.11 | 1945.1 | 80.2 |
| FF | — | — | 4.61 | 2017.5 | 68.0 |
| EDF | — | — | 2.41 | 1292.3 | 94.7 |

Bold = best among metaheuristics for that metric.

---

## Citation

If you use this code or data, please cite:

```
Arrabotu, C.S., Anita, J.P.: Multi-Objective Hybrid PSO–GA Algorithm with
DAG-Aware Rescheduling for Task Offloading in Device–Edge–Cloud Collaborative
Computing. Journal of Cloud Computing (under review), 2026.
```

---

## License

This repository is made available for research reproducibility purposes.  
© 2025 Chandra Shaker Arrabotu, Amrita Vishwa Vidyapeetham.
