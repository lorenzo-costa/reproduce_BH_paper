"""
Script to run comparison between old and new simulation functions
"""

from src.helper_functions.simulation_functs import run_simulation
from src.helper_functions.metrics import (
    Power,
    TrueRejections,
    RejectionsNumber,
    FalseDiscoveryRate,
)
from src.helper_functions.methods import (
    Bonferroni,
    BonferroniHochberg,
    BenjaminiHochberg,
)

from src.helper_old.simulation_functs import (run_simulation as run_simulation_old)

from src.helper_old.metrics import (
    Power as PowerOld,
    TrueRejections as TrueRejectionsOld,
    RejectionsNumber as RejectionsNumberOld,
    FalseDiscoveryRate as FalseDiscoveryRateOld,
)
from src.helper_old.methods import (
    Bonferroni as BonferroniOld,
    BonferroniHochberg as BonferroniHochbergOld,
    BenjaminiHochberg as BenjaminiHochbergOld,
)

from src.helper_functions.analyse_functions import aggregate_results

import json
import os
import numpy as np
import yaml
import time
import argparse
import pandas as pd

method_map = {
    "Bonferroni": Bonferroni,
    "BonferroniHochberg": BonferroniHochberg,
    "BenjaminiHochberg": BenjaminiHochberg,
}

method_map_old = {
    "Bonferroni": BonferroniOld,
    "BonferroniHochberg": BonferroniHochbergOld,
    "BenjaminiHochberg": BenjaminiHochbergOld,
}

results_dir = "results/comparison/"

if __name__ == "__main__":
    try: 
        sim_out_new = pd.read_csv("data/timing_new/simulated/full_simulation_results.csv")
        sim_out_new_parallel = pd.read_csv("data/timing_new_parallel/simulated/full_simulation_results.csv")
        sim_out_old = pd.read_csv("data/timing_old/simulated/full_simulation_results.csv")
        sim_out_old_concat = pd.read_csv("data/timing_old_concat/simulated/full_simulation_results.csv")
        print("Loaded existing simulation results.")
        
    except FileNotFoundError:
        print("Running simulations...")

        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
            
        alpha = cfg["alpha"]
        m = cfg["m"]
        m0 = cfg["m0"]
        L = cfg["L"]
        scheme = cfg["scheme"]
        
            
        nsim = 10000
        methods = [method_map[name]() for name in cfg["methods"]]
        methods_old = [method_map_old[name]() for name in cfg["methods"]]
        
        metrics = [Power(), TrueRejections(), RejectionsNumber(), FalseDiscoveryRate()]
        metrics_old = [PowerOld(), TrueRejectionsOld(), RejectionsNumberOld(), FalseDiscoveryRateOld()]
        
        rng1 = np.random.default_rng(cfg["rng_seed"])
        sim_out_new, samples_list = run_simulation(
            nsim=nsim,
            m=m,
            m0_fraction=m0,
            L=L,
            scheme=scheme,
            method=methods,
            alpha=alpha,
            rng=rng1,
            metrics=metrics,
            parallel=False,
        )
        
        rng2 = np.random.default_rng(cfg["rng_seed"])
        sim_out_new_parallel, samples_list_parallel = run_simulation(
            nsim=nsim,
            m=m,
            m0_fraction=m0,
            L=L,
            scheme=scheme,
            method=methods,
            alpha=alpha,
            rng=rng2,
            metrics=metrics,
            parallel=True,
        )
        
        rng3 = np.random.default_rng(cfg["rng_seed"])
        sim_out_old, samples_list_old = run_simulation_old(
            nsim=nsim,
            m=m,
            m0_fraction=m0,
            L=L,
            scheme=scheme,
            method=methods_old,
            alpha=alpha,
            rng=rng3,
            metrics=metrics_old,
            parallel=True,
            old=False
        )
        
        rng4 = np.random.default_rng(cfg["rng_seed"])
        sim_out_old_concat, samples_list_old_concat = run_simulation_old(
            nsim=nsim,
            m=m,
            m0_fraction=m0,
            L=L,
            scheme=scheme,
            method=methods_old,
            alpha=alpha,
            rng=rng4,
            metrics=metrics_old,
            parallel=True,
            old=True,
        )
    
    agg_new = sim_out_new.groupby(['method', 'm', 'm0_fraction', 'scheme', 'L']).mean().reset_index()
    agg_new_parallel = sim_out_new_parallel.groupby(['method', 'm', 'm0_fraction', 'scheme', 'L']).mean().reset_index()
    agg_old = sim_out_old.groupby(['method', 'm', 'm0_fraction', 'scheme', 'L']).mean().reset_index()
    agg_old_concat = sim_out_old_concat.groupby(['method', 'm', 'm0_fraction', 'scheme', 'L']).mean().reset_index()
    
    tolerance = 1e-4
    all_good = True
    for metric in metrics:
        metric = metric.name
        if not np.allclose(agg_new[metric], agg_old[metric], atol=tolerance):
            print(f"Discrepancy found in metric {metric} between new and old simulation!")
            print(np.max(np.abs(agg_new[metric] - agg_old[metric])))
            all_good = False
        if not np.allclose(agg_new_parallel[metric], agg_old[metric], atol=tolerance):
            print(f"Discrepancy found in metric {metric} between new parallel and old simulation!")
            all_good = False
        if not np.allclose(agg_old_concat[metric], agg_old[metric], atol=tolerance):
            print(f"Discrepancy found in metric {metric} between old concatenated and old simulation!")
            all_good = False
    if all_good:
        print(f"All simulation methods produce consistent results up to {tolerance}.")
    