#!/usr/bin/env python3
from pathlib import Path
import sys, json, math
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

sys.path.append('04_supplementary_material/src')
from dynamic_prr_experiment import Config, run_one, average_ranks

SCENARIOS = ["FDA1", "dMOP2", "DF1-HF", "FDA1-HS", "dMOP2-HS", "APP-RESOURCE"]
ALGORITHMS = ["NSGA-II", "PPS-NSGA-II", "ABR-NSGA-II", "PRR-NSGA-II"]
DIMS = [10, 30]
RUNS = 5
POP_SIZE = 100
STATES = 20
GENERATIONS = 3
OUT = Path('04_supplementary_material/results/prr_scalability_diag')
OUT.mkdir(parents=True, exist_ok=True)

def task(args):
    dim, scenario, algorithm, run = args
    cfg = Config(runs=RUNS, pop_size=POP_SIZE, variables=dim, states=STATES, generations_per_state=GENERATIONS, workers=1)
    row = run_one(scenario, algorithm, run, cfg)
    row['dimension'] = dim
    return row

def main():
    tasks = [(d, s, a, r) for d in DIMS for s in SCENARIOS for a in ALGORITHMS for r in range(RUNS)]
    workers = min(cpu_count(), 8)
    with Pool(workers) as pool:
        rows = list(pool.imap_unordered(task, tasks, chunksize=1))
    raw = pd.DataFrame(rows).sort_values(['dimension','scenario','algorithm','run']).reset_index(drop=True)
    raw.to_csv(OUT/'scalability_diagnostic_raw.csv', index=False)
    # scenario algorithm means
    scen = raw.groupby(['dimension','scenario','algorithm']).agg(
        HV_mean=('HV','mean'), HV_std=('HV','std'),
        IGD_mean=('IGD','mean'), IGD_std=('IGD','std'),
        Stability_mean=('Stability','mean'),
        DecisionDrift_mean=('DecisionDrift','mean'),
        QDR_mean=('QDR','mean')
    ).reset_index()
    scen.to_csv(OUT/'scalability_diagnostic_by_scenario_algorithm.csv', index=False)
    # overall by dim/alg
    overall = raw.groupby(['dimension','algorithm']).agg(
        HV_mean=('HV','mean'), HV_std=('HV','std'),
        IGD_mean=('IGD','mean'), IGD_std=('IGD','std'),
        Stability_mean=('Stability','mean'),
        DecisionDrift_mean=('DecisionDrift','mean'),
        QDR_mean=('QDR','mean')
    ).reset_index()
    # scenario-level ranks per metric averaged across 6 scenarios for each D
    rank_rows=[]
    for d, sub in scen.groupby('dimension'):
        for metric, asc in [('HV_mean', False), ('IGD_mean', True), ('Stability_mean', False), ('DecisionDrift_mean', True), ('QDR_mean', False)]:
            # rank within each scenario
            ranks=[]
            for s, ss in sub.groupby('scenario'):
                r = ss.set_index('algorithm')[metric].rank(ascending=asc, method='average')
                for alg, val in r.items():
                    ranks.append({'dimension': d, 'algorithm': alg, 'metric': metric.replace('_mean',''), 'rank': float(val)})
            df=pd.DataFrame(ranks)
            for alg, aa in df.groupby('algorithm'):
                rank_rows.append({'dimension': d, 'algorithm': alg, 'metric': metric.replace('_mean',''), 'avg_rank': aa['rank'].mean()})
    ranks = pd.DataFrame(rank_rows)
    ranks_pivot = ranks.pivot_table(index=['dimension','algorithm'], columns='metric', values='avg_rank').reset_index()
    # merge selected metrics for compact table
    compact = overall.merge(ranks_pivot, on=['dimension','algorithm'], how='left')
    compact = compact.rename(columns={'HV':'HV_rank','IGD':'IGD_rank','Stability':'Stability_rank','DecisionDrift':'DecisionDrift_rank','QDR':'QDR_rank'})
    compact.to_csv(OUT/'scalability_diagnostic_summary.csv', index=False)
    # PRR retention/change from D10 to D30 and main competitors
    # Round table for LaTeX
    print('RAW saved', raw.shape)
    print(compact)
    with open(OUT/'scalability_diagnostic_metadata.json','w') as f:
        json.dump({
            'purpose': 'Small D=30 scalability diagnostic; not a full high-dimensional rerun.',
            'dimensions': DIMS, 'scenarios': SCENARIOS, 'algorithms': ALGORITHMS, 'runs': RUNS,
            'pop_size': POP_SIZE, 'states': STATES, 'generations_per_state': GENERATIONS,
            'note': 'Same compact protocol for D=10 and D=30; fixed population size isolates the effect of decision dimension.'
        }, f, indent=2)

if __name__ == '__main__':
    main()
