"""
run_experiments.py — parallel version with CSV bug fix
"""
import sys, os, time, argparse, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from experiments.config    import ALG, TASK_SCALES, EXPERIMENT, SYSTEM, DAG, ALG_NAMES
from experiments.seed_table import SEED_TABLE
from environment.de3c_system   import DE3CSystem
from environment.dag_generator  import generate_all_instances
from algorithms.mo_psoga_dag import run as run_mo
from algorithms.psoga_r      import run as run_pso
from algorithms.baselines    import run_nsga2, run_mopso, run_moead, run_ff, run_edf
from metrics.pareto_metrics  import (hypervolume, igd, gd, spacing,
                                      compute_reference_point,
                                      best_compromise)
from algorithms.pareto_utils import fast_non_dominated_sort

warnings.filterwarnings('ignore')

ALG_FUNCS = {
    'MO-PSOGA-DAG': run_mo,
    'NSGA-II':      run_nsga2,
    'MOPSO':        run_mopso,
    'MOEAD':        run_moead,
    'PSOGA_R':      run_pso,
    'FF':           run_ff,
    'EDF':          run_edf,
}
MO_ALGS = {'MO-PSOGA-DAG', 'NSGA-II', 'MOPSO', 'MOEAD'}


def _fmt(seconds):
    s = int(seconds)
    if s < 60:   return f"{s}s"
    if s < 3600: return f"{s//60}m {s%60}s"
    return f"{s//3600}h {(s%3600)//60}m"


def _worker(args):
    (alg_name, run_idx, seed, T, inst_idx, density,
     sys_seed, dag_seed_base) = args
    sys.path.insert(0, str(_PROJECT_ROOT))
    from environment.de3c_system   import DE3CSystem
    from environment.dag_generator  import generate_all_instances
    system = DE3CSystem(D=SYSTEM['D'], E=SYSTEM['E'], V=SYSTEM['V'],
                        rng=np.random.default_rng(sys_seed))
    instances = generate_all_instances(
        T=T, D=SYSTEM['D'], rng=np.random.default_rng(dag_seed_base),
        densities=DAG['edge_density_values'])
    dag_inst = instances[inst_idx]
    fn  = ALG_FUNCS[alg_name]
    rng = np.random.default_rng(seed)
    t0  = time.time()
    r   = fn(dag=dag_inst, system=system, rng=rng, N=ALG['N'], I=ALG['I'])
    return dict(
        alg=alg_name, run=int(run_idx+1), T=int(T),
        inst=int(inst_idx+1), density=float(density),
        seed=int(seed), elapsed=float(time.time()-t0),
        archive_obj=r['archive_obj'].tolist(),   # ← convert to list for pickling
        archive_f4=r['archive_f4'].tolist(),
    )


def _metrics(rec, ref_point, ref_front):
    aobj = np.array(rec['archive_obj'])   # ← convert back to numpy
    f4   = np.array(rec['archive_f4'])
    alg  = rec['alg']

    if alg in MO_ALGS and len(aobj) > 0:
        hv_v  = float(hypervolume(aobj, ref_point))
        igd_v = float(igd(aobj, ref_front))
        gd_v  = float(gd(aobj, ref_front))
        sp_v  = float(spacing(aobj))
    else:
        hv_v = igd_v = gd_v = sp_v = float('nan')

    bc   = best_compromise(aobj)
    bobj = aobj[bc]
    return {
        'T':         rec['T'],
        'instance':  rec['inst'],
        'density':   rec['density'],
        'algorithm': str(alg),
        'run':       rec['run'],
        'seed':      rec['seed'],
        'elapsed':   rec['elapsed'],
        'HV':        hv_v,
        'IGD':       igd_v,
        'GD':        gd_v,
        'Spacing':   sp_v,
        'F1':        float(bobj[0]),
        'F2':        float(bobj[1]),
        'F3':        float(-bobj[2]),
        'F4':        float(f4[bc]),
    }


def run_experiments(task_scales=None, alg_names=None,
                    save_dir='results/raw', n_workers=None):
    if task_scales is None: task_scales = TASK_SCALES
    if alg_names   is None: alg_names   = ALG_NAMES
    if n_workers   is None: n_workers   = max(1, cpu_count()-1)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    all_records = []

    for T in task_scales:
        print(f"\n{'='*60}")
        print(f"  Scale T={T}  |  Workers={n_workers}")
        print(f"{'='*60}")
        sys_seed      = EXPERIMENT['master_seed'] + T
        dag_seed_base = EXPERIMENT['master_seed'] + T + 1

        for inst_idx in range(EXPERIMENT['n_instances']):
            density    = DAG['edge_density_values'][inst_idx]
            total_jobs = len(alg_names) * EXPERIMENT['n_runs']
            print(f"\n  Instance {inst_idx+1}/3  "
                  f"(density={density}, {total_jobs} jobs)")
            print(f"  Waiting for first job (silence is normal)...")

            jobs = [
                (alg, run_idx, int(SEED_TABLE[run_idx]),
                 T, inst_idx, density, sys_seed, dag_seed_base)
                for alg in alg_names
                for run_idx in range(EXPERIMENT['n_runs'])
            ]

            raw_results = []
            t_start     = time.time()
            alg_times   = {}

            with Pool(processes=n_workers) as pool:
                for i, res in enumerate(pool.imap_unordered(_worker, jobs), 1):
                    raw_results.append(res)
                    alg = res['alg']
                    alg_times.setdefault(alg, []).append(res['elapsed'])
                    elapsed = time.time() - t_start
                    eta     = (elapsed / i) * (total_jobs - i)
                    print(f"  [{i:3d}/{total_jobs}]  "
                          f"elapsed={_fmt(elapsed)}  "
                          f"ETA={_fmt(eta)}  "
                          f"last={alg}({_fmt(res['elapsed'])})")

            # Reference point and front
            all_obj   = [np.array(r['archive_obj']) for r in raw_results]
            ref_pt    = compute_reference_point(all_obj, 1.1)
            combined  = np.vstack(all_obj)
            fronts    = fast_non_dominated_sort(combined)
            ref_front = combined[fronts[0]]
            print(f"\n  Reference front size: {len(ref_front)}")

            # Compute metrics
            inst_records = []
            for rec in raw_results:
                m = _metrics(rec, ref_pt, ref_front)
                all_records.append(m)
                inst_records.append(m)

            # Save this instance — explicit column order, no filtering
            df_inst = pd.DataFrame(inst_records)
            out = f"{save_dir}/raw_T{T}_inst{inst_idx+1}.csv"
            df_inst.to_csv(out, index=False)
            print(f"  Saved {len(df_inst)} rows → {out}")

            # Timing summary
            print(f"\n  Timing per algorithm:")
            for name in alg_names:
                if name in alg_times:
                    t = alg_times[name]
                    print(f"    {name:<15}: avg={_fmt(sum(t)/len(t))}"
                          f"  runs={len(t)}")

    df_all = pd.DataFrame(all_records)
    df_all.to_csv(f'{save_dir}/raw_results_all.csv', index=False)

    summary = (df_all.groupby(['T','algorithm'])
               [['HV','IGD','GD','Spacing','F1','F2','F3','F4']]
               .agg(['mean','std']).round(4))
    summary.to_csv('results/tables/summary_mean_std.csv')
    print(f"\nComplete. {len(df_all)} total records.")
    print(f"Summary → results/tables/summary_mean_std.csv")

    # Print quick results table
    print(f"\n=== MEAN RESULTS SUMMARY ===")
    mean_df = df_all.groupby(['T','algorithm'])[['F1','F2','F3','HV']].mean().round(3)
    print(mean_df.to_string())
    return df_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale',    type=int, nargs='+', default=TASK_SCALES)
    parser.add_argument('--alg',      type=str, nargs='+', default=ALG_NAMES)
    parser.add_argument('--workers',  type=int, default=None)
    parser.add_argument('--save_dir', type=str, default='results/raw')
    args = parser.parse_args()

    n = args.workers or max(1, cpu_count()-1)
    print(f"MO-PSOGA-DAG Experiment Runner")
    print(f"Scales   : {args.scale}")
    print(f"Workers  : {n}")
    print(f"Budget   : N={ALG['N']} x I={ALG['I']} = {ALG['budget']}")
    print(f"Note     : Silence until first job completes is normal.")

    run_experiments(task_scales=args.scale, alg_names=args.alg,
                    n_workers=n, save_dir=args.save_dir)
