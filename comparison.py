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

import json
import os
import numpy as np
import yaml
import time
import argparse
import subprocess
import pickle

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
    # load config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    alpha = cfg["alpha"]
    m = cfg["m"]
    m0 = cfg["m0"]
    L = cfg["L"]
    scheme = cfg["scheme"]
    rng = np.random.default_rng(cfg["rng_seed"])

    n_sim_list = [1, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4]

    methods = [method_map[name]() for name in cfg["methods"]]
    methods_old = [method_map_old[name]() for name in cfg["methods"]]
    
    metrics = [Power(), TrueRejections(), RejectionsNumber(), FalseDiscoveryRate()]
    metrics_old = [PowerOld(), TrueRejectionsOld(), RejectionsNumberOld(), FalseDiscoveryRateOld()]
    
    times_old = {}
    times_old_concat = {}
    times_new = {}
    times_new_parallel = {}
    
    os.makedirs(results_dir, exist_ok=True)
    
    for nsim in n_sim_list:
        if os.path.exists(os.path.join(results_dir, f"timing_{int(nsim)}.pkl")):
            print(f"Timing for nsim={int(nsim)} already exists, skipping...")
            continue
        nsim = int(nsim)
        print(f"\nRunning simulations with nsim={nsim}")
        
        start_time_new_parallel = np.ones(5)*1e10
        end_time_new_parallel = np.zeros(5)
        for i in range(5):
            start_time_new_parallel[i] = time.time()
            subprocess.run(["python", "-m", "src.run_simulation", "--nsim", str(nsim), "--parallel", "1", "--save", "0"])
            end_time_new_parallel[i] = time.time()
        
        times_new_parallel[nsim] = (end_time_new_parallel - start_time_new_parallel)
        print(f"New simulation function (parallel) took {np.max(end_time_new_parallel - start_time_new_parallel):.2f} seconds.")

        start_time_new = np.ones(5)*1e10
        end_time_new = np.zeros(5)
        for i in range(5):
            start_time_new[i] = time.time()
            subprocess.run(["python", "-m", "src.run_simulation", "--nsim", str(nsim), "--parallel", "0", "--save", "0"])
            end_time_new[i] = time.time()
            
        times_new[nsim] = (end_time_new - start_time_new)
        print(f"New simulation function (sequential) took {np.max(end_time_new - start_time_new):.2f} seconds.")

        start_time_old = np.ones(5)*1e10
        end_time_old = np.zeros(5)
        for i in range(5):
            start_time_old[i] = time.time()
            subprocess.run(["python", "-m", "src.run_simulation_old", "--nsim", str(nsim), "--parallel", "1", "--save", "0", "--old", "0"])
            end_time_old[i] = time.time()
    
        times_old[nsim] = (end_time_old - start_time_old)
        print(f"Old simulation function took {np.max(end_time_old - start_time_old):.2f} seconds.")
        
        start_time_old_concat = np.ones(5)*1e10
        end_time_old_concat = np.zeros(5)
        for i in range(5):
            start_time_old_concat[i] = time.time()
            subprocess.run(["python", "-m", "src.run_simulation_old", "--nsim", str(nsim), "--parallel", "1", "--save", "0", "--old", "1"])
            end_time_old_concat[i] = time.time()
        
        times_old_concat[nsim] = (end_time_old_concat - start_time_old_concat)
        print(f"Old simulation function with concatenation took {np.max(end_time_old_concat - start_time_old_concat):.2f} seconds.")

        with open(os.path.join(results_dir, f"timing_{nsim}.pkl"), "wb") as f:
            pickle.dump(
                {
                    "n_sim_list": nsim,
                    "times_old": end_time_old - start_time_old,
                    "times_old_concat": end_time_old_concat - start_time_old_concat,
                    "times_new": end_time_new - start_time_new,
                    "times_new_parallel": end_time_new_parallel - start_time_new_parallel,
                },
                f,
            )
    
    