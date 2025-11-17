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
        
    n_sim_list = [1e1, 5e1, 1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 5e4, 1e5, 5e5]
    
    methods = [method_map[name]() for name in cfg["methods"]]
    methods_old = [method_map_old[name]() for name in cfg["methods"]]
    
    metrics = [Power(), TrueRejections(), RejectionsNumber(), FalseDiscoveryRate()]
    metrics_old = [PowerOld(), TrueRejectionsOld(), RejectionsNumberOld(), FalseDiscoveryRateOld()]
    
    times_old = []
    times_old_concat = []
    times_new = []
    times_new_parallel = []
    
    os.makedirs(results_dir, exist_ok=True)
    
    for nsim in n_sim_list:
        if os.path.exists(os.path.join(results_dir, f"timing_{int(nsim)}.json")):
            print(f"Timing for nsim={int(nsim)} already exists, skipping...")
            continue
        nsim = int(nsim)
        print(f"Running simulations with nsim={nsim}")
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--parallel",
            type=int,
            choices=[0, 1],
            help="Whether to run simulations in parallel (1) or sequentially (0). Overrides config file.",
        )
        args = parser.parse_args()
        parallel = bool(int(args.parallel)) if args.parallel is not None else cfg.get("parallel", False)

        start_time_new_parallel = time.time()
        sim_out, samples_list = run_simulation(
            nsim=nsim,
            m=m,
            m0_fraction=m0,
            L=L,
            scheme=scheme,
            method=methods,
            alpha=alpha,
            rng=rng,
            metrics=metrics,
            parallel=False,
        )
        end_time_new_parallel = time.time()
        times_new_parallel.append((end_time_new_parallel - start_time_new_parallel))
        print(f"New simulation function (parallel) took {end_time_new_parallel - start_time_new_parallel:.2f} seconds.")

        start_time_new = time.time()
        sim_out, samples_list = run_simulation(
            nsim=nsim,
            m=m,
            m0_fraction=m0,
            L=L,
            scheme=scheme,
            method=methods,
            alpha=alpha,
            rng=rng,
            metrics=metrics,
            parallel=True,
        )
        end_time_new = time.time()
        times_new.append((end_time_new - start_time_new))
        print(f"New simulation function (sequential) took {end_time_new - start_time_new:.2f} seconds.")

        start_time_old = time.time()
        sim_out_old, samples_list_old = run_simulation_old(
            nsim=nsim,
            m=m,
            m0_fraction=m0,
            L=L,
            scheme=scheme,
            method=methods_old,
            alpha=alpha,
            rng=rng,
            metrics=metrics_old,
            parallel=True,
        )
        end_time_old = time.time()
        times_old.append((end_time_old - start_time_old))
        print(f"Old simulation function took {end_time_old - start_time_old:.2f} seconds.")
        
        start_time_old_concat = time.time()
        sim_out_old, samples_list_old = run_simulation_old(
            nsim=nsim,
            m=m,
            m0_fraction=m0,
            L=L,
            scheme=scheme,
            method=methods_old,
            alpha=alpha,
            rng=rng,
            metrics=metrics_old,
            parallel=True,
            old=True,
        )
        end_time_old_concat = time.time()
        times_old_concat.append((end_time_old_concat - start_time_old_concat))
        print(f"Old simulation function with concatenation took {end_time_old_concat - start_time_old_concat:.2f} seconds.")

        with open(os.path.join(results_dir, f"timing_{nsim}.json"), "w") as f:
            json.dump(
                {
                    "n_sim_list": n_sim_list,
                    "times_old": end_time_old - start_time_old,
                    "times_old_concat": end_time_old_concat - start_time_old_concat,
                    "times_new": end_time_new - start_time_new,
                    "times_new_parallel": end_time_new_parallel - start_time_new_parallel,
                },
                f,
                indent=4,
            )
    
    