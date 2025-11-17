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
        
    nsim = 1000
    methods = [method_map[name]() for name in cfg["methods"]]
    methods_old = [method_map_old[name]() for name in cfg["methods"]]
    
    metrics = [Power(), TrueRejections(), RejectionsNumber(), FalseDiscoveryRate()]
    metrics_old = [PowerOld(), TrueRejectionsOld(), RejectionsNumberOld(), FalseDiscoveryRateOld()]
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parallel",
        type=int,
        choices=[0, 1],
        help="Whether to run simulations in parallel (1) or sequentially (0). Overrides config file.",
    )
    args = parser.parse_args()
    parallel = bool(int(args.parallel)) if args.parallel is not None else cfg.get("parallel", False)

    
    sim_out_new, samples_list = run_simulation(
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
    
    sim_out_new_parallel, samples_list_parallel = run_simulation(
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
        old=False
    )
    
    sim_out_old_concat, samples_list_old_concat = run_simulation_old(
        nsim=nsim,
        m=m,
        m0_fraction=m0,
        L=L,
        scheme=scheme,
        method=methods_old,
        alpha=alpha,
        rng=rng,
        metrics=metrics_old,
        parallel=parallel,
        old=True,
    )
    
    
    