"""
Script to run comparison between old and new simulation functions
"""

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

import os
import numpy as np
import yaml
import time
import subprocess
import pickle
import argparse

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

results_dir = "results/"

if __name__ == "__main__":
    # load config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default=None)
    
    args = parser.parse_args()
    target = args.target if args.target is not None else "benchmark"

    alpha = cfg["alpha"]
    m = cfg["m"]
    m0 = cfg["m0"]
    L = cfg["L"]
    scheme = cfg["scheme"]
    rng = np.random.default_rng(cfg["rng_seed"])

    n_sim_list = [1, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4]
    
    # go back to a single one because it takes too long
    n_repeats = 1

    methods = [method_map[name]() for name in cfg["methods"]]
    methods_old = [method_map_old[name]() for name in cfg["methods"]]
    
    metrics = [Power(), TrueRejections(), RejectionsNumber(), FalseDiscoveryRate()]
    metrics_old = [PowerOld(), TrueRejectionsOld(), RejectionsNumberOld(), FalseDiscoveryRateOld()]
    
    times_old = {}
    times_old_concat = {}
    times_new = {}
    times_new_parallel = {}
    times_old_sequential = {}

    if target == "complexity":
        os.makedirs(results_dir+"complexity/", exist_ok=True)
    else:   
        os.makedirs(results_dir+"benchmark/", exist_ok=True)

    for nsim in n_sim_list:
        nsim = int(nsim)
        print(f"\nRunning simulations with nsim={nsim}")
        
        # only time fast, parallel function
        if target == "complexity":
            if os.path.exists(os.path.join(results_dir, f"complexity/complexity_{int(nsim)}.pkl")):
                print(f"Complexity for nsim={int(nsim)} already exists, skipping...")
                continue
            
            times = []
            for i in range(1, n_repeats + 2):
                print(f"Run {i}")
                start = time.time()
                subprocess.run(["python", "-m", "src.run_simulation", 
                                "--nsim", str(nsim), 
                                "--parallel", "0",
                                "--results_dir", "results/complexity/",
                                "--data_dir", "data/complexity/"])
                end = time.time()
                times.append(end - start)

            print(f"Average elapsed time: {sum(times) / len(times):.3f} seconds +/- {np.std(times):.3f} seconds")

            with open(f"{results_dir}complexity/complexity_{nsim}.pkl", "wb") as fh:
                pickle.dump({
                    "nsim": nsim,
                    "times": times,
                }, fh)

        # else run all comparisons
        else:
            if os.path.exists(os.path.join(results_dir, f"benchmark/benchmark_{int(nsim)}.pkl")):
                print(f"Benchmark for nsim={int(nsim)} already exists, skipping...")
                continue

            start_time_new_parallel = np.ones(n_repeats)*1e10
            end_time_new_parallel = np.zeros(n_repeats)
            for i in range(n_repeats):
                start_time_new_parallel[i] = time.time()
                subprocess.run(["python", "-m", "src.run_simulation", 
                                "--nsim", str(nsim), 
                                "--parallel", "1", 
                                "--save", "1",
                                "--results_dir", "results_benchmark_new_parallel/",
                                "--data_dir", "data/benchmark_new_parallel/"])
                end_time_new_parallel[i] = time.time()
            
            times_new_parallel[nsim] = (end_time_new_parallel - start_time_new_parallel)
            print(f"New simulation function (parallel) took {np.max(end_time_new_parallel - start_time_new_parallel):.2f} seconds.\n")

            start_time_new = np.ones(n_repeats)*1e10
            end_time_new = np.zeros(n_repeats)
            for i in range(n_repeats):
                start_time_new[i] = time.time()
                subprocess.run(["python", "-m", "src.run_simulation", 
                                "--nsim", str(nsim), 
                                "--parallel", "0", 
                                "--save", "1",
                                "--results_dir", "results_benchmark_new/",
                                "--data_dir", "data/benchmark_new/"])
                end_time_new[i] = time.time()
                
            times_new[nsim] = (end_time_new - start_time_new)
            print(f"New simulation function (sequential) took {np.max(end_time_new - start_time_new):.2f} seconds.\n")

            start_time_old = np.ones(n_repeats)*1e10
            end_time_old = np.zeros(n_repeats)
            for i in range(n_repeats):
                start_time_old[i] = time.time()
                subprocess.run(["python", "-m", "src.run_simulation_old", 
                                "--nsim", str(nsim), 
                                "--parallel", "1", 
                                "--save", "1", 
                                "--old", "0",
                                "--results_dir", "results/benchmark_old/",
                                "--data_dir", "data/benchmark_old/"])
                end_time_old[i] = time.time()
        
            times_old[nsim] = (end_time_old - start_time_old)
            print(f"Old simulation function took {np.max(end_time_old - start_time_old):.2f} seconds.")

            start_time_old_concat = np.ones(n_repeats)*1e10
            end_time_old_concat = np.zeros(n_repeats)
            for i in range(n_repeats):
                start_time_old_concat[i] = time.time()
                subprocess.run(["python", "-m", "src.run_simulation_old", 
                                "--nsim", str(nsim), 
                                "--parallel", "1", 
                                "--save", "1", 
                                "--old", "1",
                                "--results_dir", "results/benchmark_old_concat/",
                                "--data_dir", "data/benchmark_old_concat/"])
                end_time_old_concat[i] = time.time()
            
            times_old_concat[nsim] = (end_time_old_concat - start_time_old_concat)
            print(f"Old simulation function with concatenation took {np.max(end_time_old_concat - start_time_old_concat):.2f} seconds.\n")
            
            # # running this only for small inputs because it takes too long
            # if nsim <= 1001:
            #     start_time_old_sequential = time.time()
            #     subprocess.run(["python", "-m", "src.run_simulation_old", 
            #                     "--nsim", str(nsim), 
            #                     "--parallel", "0", 
            #                     "--save", "1", 
            #                     "--old", "1",
            #                     "--results_dir", "results_timing_old_sequential/",
            #                     "--data_dir", "data_timing/"])
            #     end_time_old_sequential = time.time()
            #     times_old_sequential[nsim] = np.array(end_time_old_sequential - start_time_old_sequential)
            #     print(f"Old simulation function sequential with concatenation took {end_time_old_sequential - start_time_old_sequential:.2f} seconds.\n")

            with open(os.path.join(results_dir, f"benchmark/benchmark_{nsim}.pkl"), "wb") as f:
                pickle.dump(
                    {
                        "nsim": nsim,
                        "times_old": end_time_old - start_time_old,
                        "times_old_concat": end_time_old_concat - start_time_old_concat,
                        # "times_old_sequential": end_time_old_sequential - start_time_old_sequential,
                        "times_new": end_time_new - start_time_new,
                        "times_new_parallel": end_time_new_parallel - start_time_new_parallel,
                    },
                    f,
                )
    
    